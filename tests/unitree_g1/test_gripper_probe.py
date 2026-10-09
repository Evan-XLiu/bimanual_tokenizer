import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/unitree_g1'))
from gripper_probe import (GripperSource,GripperIO,ACTIVE,probe_config,observation_from_telemetry,
                           gripper_fault,INDICES)
from hold_arms import load_profile,run_hold,OwnershipLost
from arm_execution import Observation

ROOT=Path(__file__).resolve().parents[2]

class GripperTests(unittest.TestCase):
    def base(self):
        return load_profile(ROOT/'artifacts/unitree_g1/hold_shadow_profile.json',False)

    def test_feedback_mapping_and_invalid_values(self):
        telemetry=dict(received=1.,tick=5,motors=[dict(index=i,q=q,dq=0.,motorstate=0) for i,q in [(33,.2),(31,-.007)]])
        self.assertEqual(observation_from_telemetry(telemetry).q,(-.007,.2)+(0.,)*12)
        for key,value in [('q',float('nan')),('dq',float('inf')),('motorstate',1)]:
            bad=dict(telemetry,motors=[dict(x) for x in telemetry['motors']]);bad['motors'][0][key]=value
            with self.assertRaises(ValueError):observation_from_telemetry(bad)
        with self.assertRaises(ValueError):observation_from_telemetry(dict(telemetry,motors=[]))

    def test_range_and_changed_pose_rejection(self):
        _,cfg=self.base()
        for q in [(-.1,0.),(5.,0.),(0.,float('nan'))]:
            with self.assertRaises(ValueError):probe_config(cfg,q+(0.,)*12)
        source=GripperSource((0.,)*14)
        with self.assertRaisesRegex(ValueError,'pose changed'):
            source.initialize(SimpleNamespace(c=cfg,observation=Observation((.01,)+(0.,)*13,0,1,5)),0)

    def test_startup_gain_keeps_protection_and_survives_initialize(self):
        _,cfg=self.base()
        initial=(0.,)*14
        executor=SimpleNamespace(c=cfg,observation=Observation(initial,0,1,5))
        GripperSource(initial,8.).initialize(executor,0)
        self.assertEqual(executor.c.kp,(8.,)*14)
        self.assertEqual(executor.c.tracking_error,(.15,)*14)
        self.assertEqual(executor.c.velocity,(.2,)*14)
        with self.assertRaises(ValueError):probe_config(cfg,initial,7.)

    def simulate(self,stop_at=math.inf,freeze_at=math.inf,foreign_at=math.inf,error_at=math.inf,negative_open=False):
        profile,cfg=self.base();profile['hold_s']=ACTIVE
        if negative_open:profile['hold_s']=6.
        initial=(-.007,-.006)+(0.,)*12;cfg=probe_config(cfg,initial,negative_open=negative_open)
        class IO:
            t=0.;q=initial
            def __init__(self):self.commands=[]
            def check(self):
                if self.t>=foreign_at:raise OwnershipLost('foreign command')
            def sleep(self,dt):self.t+=dt
            def observation(self):
                t=min(self.t,freeze_at);q=self.q
                if self.t>=error_at:q=(q[0]+.16,)+q[1:]
                return Observation(q,t,round(t*10000)+1,5)
            def write(self,c):self.q=c.q;self.commands.append((self.t,c));return True
        io=IO();result=run_hold(io,cfg,profile,lambda:io.t>=stop_at,lambda _:None,
                              lambda:io.t,io.sleep,GripperSource(initial,negative_open=negative_open))
        return io,result

    def test_negative_open_is_bounded_and_does_not_close_on_exit(self):
        io,r=self.simulate(negative_open=True)
        self.assertIsNone(r['error']);self.assertTrue(r['unload_submitted'])
        self.assertLess(io.t,10.5)
        self.assertAlmostEqual(io.commands[-1][1].q[0],-.207,delta=1e-4)
        for (_,a),(_,b) in zip(io.commands,io.commands[1:]):
            self.assertLessEqual(b.q[0]-a.q[0],1e-8)
            self.assertLessEqual(abs(b.q[0]-a.q[0])/.01,.1+1e-8)
        self.assertEqual(io.commands[-1][1].kp,(0.,)*14)
        _,cfg=self.base()
        with self.assertRaises(ValueError):probe_config(cfg,(-.1,)*2+(0.,)*12,negative_open=True)
        with self.assertRaises(ValueError):probe_config(cfg,(0.,)*14,8.,True)
        io,r=self.simulate(error_at=4.,negative_open=True)
        self.assertEqual(r['first_fault']['reason'],'tracking_error')
        self.assertTrue(r['unload_submitted'])

    def test_open_return_rates_and_zero_tail(self):
        io,r=self.simulate();self.assertIsNone(r['error']);self.assertTrue(r['unload_submitted'])
        self.assertAlmostEqual(max(c.q[0] for _,c in io.commands),.493,delta=1e-4)
        self.assertAlmostEqual(io.commands[-1][1].q[0],-.007,delta=1e-4)
        self.assertEqual(io.commands[0][1].q,(-.007,-.006)+(0.,)*12)
        self.assertTrue(all(c.q[2:]==(0.,)*12 for _,c in io.commands))
        self.assertEqual(io.commands[-1][1].kp,(0.,)*14)
        v=[(b.q[0]-a.q[0])/.01 for (_,a),(_,b) in zip(io.commands,io.commands[1:])]
        self.assertLessEqual(max(map(abs,v)),.2+1e-8)
        self.assertLessEqual(max(abs(b-a)/.01 for a,b in zip(v,v[1:])),.4+1e-8)

    def test_stop_stale_tracking_and_ownership(self):
        io,r=self.simulate(stop_at=4.);self.assertIsNone(r['error']);self.assertLess(io.t,7.)
        self.assertGreater(io.commands[-1][1].q[0],-.007)
        io,r=self.simulate(freeze_at=4.);self.assertIsNotNone(r['error']);self.assertTrue(r['unload_submitted'])
        io,r=self.simulate(error_at=4.);self.assertEqual(r['first_fault']['reason'],'tracking_error')
        mapped=gripper_fault(r['first_fault']);self.assertEqual(mapped['tracking_exceeded_gripper_motor_indices'],[31])
        io,r=self.simulate(foreign_at=4.);self.assertFalse(r['unload_submitted']);self.assertTrue(all(t<4. for t,_ in io.commands))

    def test_readonly_cannot_write(self):
        with self.assertRaisesRegex(RuntimeError,'Read-only'):
            GripperIO.write(SimpleNamespace(live=False),None)

    def test_default_cli_never_opens_publisher(self):
        import gripper_probe
        class ReadOnlyIO:
            def __init__(self,interface,live,profile):
                self.assert_readonly=not live
                assert self.assert_readonly
            def preflight(self):pass
            def observation(self):return Observation((0.,)*14,0,1,5)
            def telemetry(self):return None
            def close(self):pass
            def open_writer(self):raise AssertionError('Default CLI opened a publisher')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'session'
            with patch.object(gripper_probe,'GripperIO',ReadOnlyIO), patch.object(sys,'argv',
                    ['gripper_probe.py','--interface','unused','--output',str(out)]):
                self.assertEqual(gripper_probe.main(),0)
            import json
            result=json.loads((out/'summary.json').read_text())
            self.assertEqual(result['command_frames'],0)
            self.assertFalse(result['hardware_motion_enabled'])

    def test_sdk_only_gripper_slots_and_crc(self):
        code=r'''
from gripper_probe import encode_gripper_command
from arm_execution import Command
from unitree_sdk2py.utils.crc import CRC
cmd=Command((.1,.2)+(0.,)*12,(5.,)*14,(.05,)*14,5,'ACTIVE','')
m=encode_gripper_command(cmd)
assert m.motor_cmd[31].q==.1 and m.motor_cmd[33].q==.2
for i,x in enumerate(m.motor_cmd):
    assert x.mode == (1 if i < 29 or i in (31,33) else 0)
    assert x.tau==0 and x.dq==0
    if i in (31,33): assert x.kp==5 and x.kd==.05
    else: assert x.q==0 and x.kp==0 and x.kd==0
assert m.crc==CRC().Crc(m)
'''
        subprocess.run([sys.executable,'-c',code],cwd=ROOT/'scripts/unitree_g1',check=True,timeout=10)

if __name__=='__main__':unittest.main()
