import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/unitree_g1'))
import gripper_cycle as cycle
from arm_execution import Observation
from hold_arms import load_profile, run_hold


class CycleTests(unittest.TestCase):
    def simulate(self,stop_at=float('inf'),open_target=None,stalled=False):
        profile,base=load_profile(cycle.ROOT/'artifacts/unitree_g1/hold_shadow_profile.json',False)
        profile.update(hold_s=cycle.ACTIVE_S if open_target is None else cycle.ABSOLUTE_OPEN_S+cycle.DWELL_S,period_s=.004)
        initial=(-.17,-.21)+(0.,)*12
        closed=(0.,-.03)+(0.,)*12
        class IO:
            t=0.
            q=initial
            def __init__(self):self.commands=[]
            def check(self):pass
            def sleep(self,dt):self.t+=dt
            def observation(self):return Observation(self.q,self.t,round(self.t*10000)+1,5)
            def write(self,c):
                if not stalled:self.q=c.q
                self.commands.append(c)
                return True
        io=IO()
        result=run_hold(io,cycle.cycle_config(base,initial,closed,open_target),profile,
                        lambda:io.t>=stop_at,lambda _:None,lambda:io.t,io.sleep,
                        cycle.CycleSource(initial,closed,open_target))
        return io,result

    def test_absolute_open_is_bounded_and_does_not_close(self):
        io,r=self.simulate(open_target=-5.4)
        self.assertIsNone(r['error'])
        self.assertTrue(r['unload_submitted'])
        self.assertLess(io.t,115.5)
        for j in (0,1):
            q=[c.q[j] for c in io.commands]
            self.assertAlmostEqual(q[-1],-5.4,delta=1e-4)
            self.assertGreaterEqual(min(q),-5.4-1e-8)
            v=[(b-a)/.004 for a,b in zip(q,q[1:])]
            self.assertLessEqual(max(v),1e-8)
            self.assertLessEqual(max(map(abs,v)),.1+1e-8)
            self.assertLessEqual(max(abs(b-a)/.004 for a,b in zip(v,v[1:])),.2+1e-8)
        self.assertEqual(io.commands[-1].kp,(0.,)*14)

    def test_absolute_open_stall_unloads_before_target(self):
        io,r=self.simulate(open_target=-5.4,stalled=True)
        self.assertIsNotNone(r['error'])
        self.assertTrue(r['unload_submitted'])
        self.assertGreater(io.commands[-1].q[0],-1.)
        self.assertEqual(io.commands[-1].kp,(0.,)*14)

    def test_open_more_then_close_with_bounded_motion(self):
        io,r=self.simulate()
        self.assertIsNone(r['error']);self.assertTrue(r['unload_submitted'])
        self.assertLess(io.t,32.5)
        for j,start,end in ((0,-.17,0.),(1,-.21,-.03)):
            q=[c.q[j] for c in io.commands]
            self.assertAlmostEqual(min(q),start-.5,delta=1e-4)
            self.assertLessEqual(max(q),max(start,end)+1e-5)
            self.assertAlmostEqual(q[-1],end,delta=1e-4)
            v=[(b-a)/.004 for a,b in zip(q,q[1:])]
            self.assertLessEqual(max(map(abs,v)),.1+1e-8)
            self.assertLessEqual(max(abs(b-a)/.004 for a,b in zip(v,v[1:])),.2+1e-8)
        self.assertEqual(io.commands[-1].kp,(0.,)*14)

    def test_stop_during_open_does_not_close(self):
        io,r=self.simulate(8.)
        self.assertIsNone(r['error']);self.assertTrue(r['unload_submitted'])
        self.assertLess(io.t,10.5)
        self.assertLess(io.commands[-1].q[0],-.17)
        self.assertTrue(all(c.q==io.commands[-1].q for c in io.commands if c.phase=='RELEASING'))

    def test_rejects_unreviewed_poses_and_changed_start(self):
        _,base=load_profile(cycle.ROOT/'artifacts/unitree_g1/hold_shadow_profile.json',False)
        for initial,closed in [((-.5,)*2,(0.,)*2),((0.,)*2,(.1,)*2)]:
            with self.assertRaises(ValueError):cycle.cycle_config(base,initial+(0.,)*12,closed+(0.,)*12)
        source=cycle.CycleSource((-.2,)*2+(0.,)*12,(0.,)*14)
        with self.assertRaisesRegex(ValueError,'pose changed'):
            source.initialize(SimpleNamespace(observation=Observation((-.1,)*2+(0.,)*12,0,1,5)),0)

    def test_default_is_readonly(self):
        class IO:
            def __init__(self,interface,live,profile):assert not live
            def preflight(self):pass
            def observation(self):return Observation((-.2,)*2+(0.,)*12,0,1,5)
            def telemetry(self):return None
            def close(self):pass
            def open_writer(self):raise AssertionError('Unexpected publisher')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'session'
            with patch.object(cycle,'GripperIO',IO),patch.object(sys,'argv',
                ['gripper_cycle.py','--interface','unused','--output',str(out)]):
                self.assertEqual(cycle.main(),0)
            self.assertEqual(json.loads((out/'summary.json').read_text())['command_frames'],0)
