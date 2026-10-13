from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/unitree_g1'))
from arm_execution import Command, Observation
from hold_arms import load_profile, run_hold
from quest_grippers import GripperFollower, QuestGripperIO, gripper_settings, raw_triggers

ROOT=Path(__file__).resolve().parents[2]


def telemetry(q,t):
    return dict(received=t,motors=[dict(index=i,q=x,dq=0.,motorstate=0) for i,x in zip((31,33),q)])


class GripperQuestTests(unittest.TestCase):
    def setUp(self):
        self.f=GripperFollower(gripper_settings())
        self.start=(-.024,-.061)
        self.f.initialize(telemetry(self.start,0.),0.)
        self.cmd=Command((0.,)*14,(80.,)*14,(3.,)*14,5,'ACTIVE','')

    def test_raw_trigger_values_not_upstream_transformed_values(self):
        self.assertEqual(raw_triggers(dict(raw={'leftState':{'triggerValue':.2},
            'rightState':{'triggerValue':.8}},official_head_yaw={'triggerValue':8})),(.2,.8))
        for value in (float('nan'),-1.,1.1,True):
            with self.assertRaises(ValueError):raw_triggers(dict(raw={'leftState':{'triggerValue':value},'rightState':{'triggerValue':0}}))
        with self.assertRaises(KeyError):raw_triggers(dict(raw={'leftState':{},'rightState':{}}))

    def test_automatic_control_linear_mapping_and_ramp_hold(self):
        packet=self.f.step(replace(self.cmd,phase='ENGAGING'),telemetry(self.start,.01),.01,80.)
        self.assertEqual(packet['q'],self.start)
        self.assertEqual(packet['picked_up'],[False,False])
        # No full squeeze is needed, including inside the old release deadband.
        for k,triggers in enumerate(([0.,.5],[.1,.2],[.15,.25],[1.,0.]),2):
            t=k*.01
            self.f.update(triggers,t,t)
            self.assertEqual(self.f.picked_up,[True,True])
            for i,trigger in enumerate(triggers):
                expected=self.f.c['opened'][i]+trigger*(self.f.c['closed'][i]-self.f.c['opened'][i])
                self.assertAlmostEqual(self.f.target[i],expected)
        packet=self.f.step(self.cmd,telemetry(self.start,.05),.05,80.)
        self.assertGreater(packet['q'][0],self.start[0])
        self.assertLess(packet['q'][1],self.start[1])

    def test_sudden_trigger_changes_are_rate_limited_and_release_freezes(self):
        previous=self.f.q;previous_v=(0.,0.)
        for k in range(1,1801):
            t=k*.01
            triggers=[1.,1.] if k==1 or k>1000 else [0.,0.]
            self.f.update(triggers,t,t)
            packet=self.f.step(self.cmd,telemetry(previous,t),t,80.)
            q=packet['q'];v=tuple((b-a)/.01 for a,b in zip(previous,q))
            self.assertLessEqual(max(map(abs,v)),2.4+1e-8)
            self.assertLessEqual(max(abs(b-a)/.01 for a,b in zip(previous_v,v)),1.2+1e-8)
            self.assertTrue(all(lo-1e-9<=x<=hi+1e-9 for x,lo,hi in zip(q,self.f.lower,self.f.upper)))
            previous,previous_v=q,v
        packet=self.f.step(replace(self.cmd,phase='RELEASING',kp=(40.,)*14),telemetry(previous,18.01),18.01,80.)
        self.assertEqual(packet['q'],previous)
        self.assertEqual(packet['kp'],[2.5,2.5])
        packet=self.f.step(replace(self.cmd,phase='RELEASED',kp=(0.,)*14),telemetry(previous,18.02),18.02,80.)
        self.assertEqual(packet['kp'],[0.,0.]);self.assertEqual(packet['kd'],[0.,0.])

    def test_stale_feedback_latches_but_allows_zero_tail(self):
        self.f.step(self.cmd,telemetry(self.start,.01),.01,80.)
        with self.assertRaisesRegex(ValueError,'Stale gripper feedback'):
            self.f.step(self.cmd,telemetry(self.start,0.),.21,80.)
        self.assertIsNotNone(self.f.fault)
        packet=self.f.step(replace(self.cmd,phase='RELEASED',kp=(0.,)*14),None,.21,80.)
        self.assertEqual(packet['q'],self.start);self.assertEqual(packet['kp'],[0.,0.])

    def test_stale_input_and_invalid_start(self):
        with self.assertRaises(ValueError):self.f.update([0.,0.],0.,.21)
        self.f.step(self.cmd,telemetry(self.start,.01),.01,80.)
        # Even with fresh feedback, absent trigger packets expire.
        with self.assertRaises(ValueError):
            for i in range(2,25):self.f.step(self.cmd,telemetry(self.start,i*.01),i*.01,80.)
        with self.assertRaises(ValueError):GripperFollower(gripper_settings()).initialize(telemetry((-5.5,-5.5),0.),0.)

    def test_full_open_and_close_within_session_at_increased_speed(self):
        previous=self.f.q
        previous_v=(0.,0.)
        self.f.update([1.,1.],.001,.001)
        for k in range(1,1401):
            t=k*.01
            self.f.update([0.,0.] if k<=700 else [1.,1.],t,t)
            packet=self.f.step(self.cmd,telemetry(previous,t),t,80.)
            v=tuple((b-a)/.01 for a,b in zip(previous,packet['q']))
            self.assertLessEqual(max(map(abs,v)),2.4+1e-8)
            self.assertLessEqual(max(abs(b-a)/.01 for a,b in zip(previous_v,v)),1.2+1e-8)
            previous=packet['q']
            previous_v=v
            if k==700:
                for q in previous:self.assertAlmostEqual(q,-5.4,places=6)
        self.assertEqual(self.f.c['opened'],[-5.4,-5.4])
        for q,c in zip(previous,self.f.c['closed']):self.assertAlmostEqual(q,c,places=6)

    def test_gripper_fault_unloads_arm_and_gripper_together(self):
        profile,cfg=load_profile(ROOT/'artifacts/unitree_g1/hold_shadow_profile.json',False)
        profile['hold_s']=6.
        f=self.f
        class IO:
            t=0.
            def __init__(self):self.sent=[]
            def check(self):pass
            def sleep(self,dt):self.t+=dt
            def observation(self):return Observation((0.,)*14,self.t,round(self.t*10000)+1,5)
            def write(self,c):
                q=f.q if self.t<3 else (f.q[0]+.31,f.q[1])
                g=f.step(c,telemetry(q,self.t),self.t,80.)
                self.sent.append((c,g));return True
        io=IO()
        class Source:
            def initialize(self,ex,now):pass
            def update(self,ex,now):
                if ex.phase=='ACTIVE':
                    f.update([1.,1.],now,now)
                    ex.send_targets((0.,)*14,now,now)
        result=run_hold(io,cfg,profile,lambda:False,lambda _:None,lambda:io.t,io.sleep,Source())
        self.assertIn('Gripper tracking_error',result['error'])
        self.assertTrue(result['unload_submitted'])
        self.assertEqual(io.sent[-1][0].kp,(0.,)*14)
        self.assertEqual(io.sent[-1][1]['kp'],[0.,0.])
        self.assertEqual(f.fault['phase'],'ACTIVE')

    def test_contact_does_not_override_trigger_target(self):
        self.f.initialize(telemetry((-3.,-3.),0.),0.)
        self.f.q=(-2.9,-3.)
        self.f.update([1.,.5],.01,.01)
        packet=self.f.step(self.cmd,telemetry((-3.,-3.),.01),.01,80.)
        self.assertEqual(packet['states'][0],'CLOSING')
        self.assertGreater(packet['q'][0],-2.9)
        self.assertEqual(packet['hold_q'],[None,None])
        self.assertEqual(packet['contact_stops'],[0,0])
        for k,trigger in enumerate((.4,.2,0.,.7),2):
            t=k*.01
            self.f.update([trigger,.5],t,t)
            self.assertAlmostEqual(self.f.target[0],-5.4+trigger*(self.f.c['closed'][0]+5.4))

    def blocked_closure(self):
        self.f.initialize(telemetry((-5.4,-5.4),0.),0.)
        actual=(-5.4,-5.4)
        for k in range(1,1001):
            t=k*.01
            self.f.update([1.,1.],t,t)
            packet=self.f.step(self.cmd,telemetry(actual,t),t,80.)
            actual=(min(-2.5,packet['q'][0]),packet['q'][1])
            if packet['closing_blocked'][0]:return t,actual,packet
        self.fail('Closing did not stop')

    def test_blocked_closure_freezes_only_that_side_and_can_open(self):
        now,actual,packet=self.blocked_closure()
        held=packet['q'][0]
        self.assertIsNone(self.f.fault)
        self.assertEqual(packet['states'][0],'CLOSING_BLOCKED')
        for k in range(1,301):
            t=now+k*.01
            self.f.update([1.,1.],t,t)
            # Feedback recovers, but must not automatically squeeze again.
            packet=self.f.step(self.cmd,telemetry((held,actual[1]),t),t,80.)
            self.assertEqual(packet['q'][0],held)
            actual=(held,packet['q'][1])
        self.assertAlmostEqual(actual[1],self.f.c['closed'][1],places=6)
        self.assertEqual(packet['states'][1],'CLOSED')
        # More-open input works even while the original positive error persists.
        t+=.01
        self.f.update([0.,1.],t,t)
        packet=self.f.step(self.cmd,telemetry((held-.31,actual[1]),t),t,80.)
        self.assertLess(packet['q'][0],held)
        self.assertEqual(packet['states'][0],'OPENING')
        self.assertIsNone(self.f.fault)
        t+=.01
        self.f.update([0.,1.],t,t)
        packet=self.f.step(self.cmd,telemetry(packet['q'],t),t,80.)
        self.assertFalse(packet['closing_blocked'][0])

    def test_blocked_closure_preserves_other_faults_and_zero_tail(self):
        for failure in ('opening_error','input','feedback'):
            self.setUp()
            now,actual,packet=self.blocked_closure()
            with self.subTest(failure=failure),self.assertRaises(ValueError):
                for k in range(1,30):
                    t=now+k*.01
                    if failure!='input':self.f.update([1.,1.],t,t)
                    q=(packet['q'][0]+.301,actual[1]) if failure=='opening_error' else actual
                    packet=self.f.step(self.cmd,telemetry(q,now if failure=='feedback' else t),t,80.)
                    actual=(-2.5,packet['q'][1])
            packet=self.f.step(replace(self.cmd,phase='RELEASED',kp=(0.,)*14),None,t+.01,80.)
            self.assertEqual(packet['kp'],[0.,0.])
            self.assertEqual(packet['states'],['FAULT','FAULT'])

    def test_tracking_error_threshold_point_three(self):
        self.assertEqual(self.f.c['tracking_error'],.30)
        self.f.initialize(telemetry((-3.,-3.),0.),0.)
        self.f.update([.5,.5],.01,.01)
        self.f.step(self.cmd,telemetry((-3.29,-3.),.01),.01,80.)
        self.assertIsNone(self.f.fault)
        self.f.update([.5,.5],.02,.02)
        with self.assertRaisesRegex(ValueError,'tracking_error'):
            self.f.step(self.cmd,telemetry((self.f.q[0]+.301,self.f.q[1]),.02),.02,80.)

    def test_status_file_reports_position_control_countdown_and_unload(self):
        import tempfile
        from unittest.mock import patch
        now=.01
        self.f.update([.5,.5],now,now)
        packet=self.f.step(self.cmd,telemetry(self.start,now),now,80.)
        io=QuestGripperIO.__new__(QuestGripperIO)
        io.grippers=self.f;io.live=False;io.active_limit=180.
        io.last_status_time=0.;io.last_display=None
        with tempfile.TemporaryDirectory() as tmp,patch('builtins.print'):
            io.status_path=Path(tmp)/'gripper_status.json'
            row=io.decorate_log(dict(time=now))
            status=json.loads(io.status_path.read_text())
            self.assertEqual(status['states'],['OPENING','OPENING'])
            self.assertEqual(row['gripper_command']['control_mode'],'trigger_position')
            self.assertAlmostEqual(status['remaining_s'],180.)
            self.f.step(replace(self.cmd,phase='RELEASED',kp=(0.,)*14),
                        telemetry(packet['q'],now+.01),now+.01,80.)
            io.decorate_log(dict(time=now+.01))
            status=json.loads(io.status_path.read_text())
            self.assertEqual(status['states'],['RELEASED','RELEASED'])
            self.assertEqual(status['kp'],[0.,0.])

    def test_sdk_single_message_contains_only_arm_and_gripper_gains(self):
        code='''
from arm_execution import Command
from quest_grippers import encode_combined
from unitree_sdk2py.utils.crc import CRC
c=Command((.1,)*14,(80.,)*14,(3.,)*14,5,'ACTIVE','')
m=encode_combined(c,dict(q=(-.2,-.3),kp=[5.,5.],kd=[.05,.05]))
assert m.crc==CRC().Crc(m)
assert m.motor_cmd[31].q==-.2 and m.motor_cmd[33].q==-.3
for i,x in enumerate(m.motor_cmd):
    assert x.tau==0 and x.dq==0
    if 15<=i<=28: assert x.kp==80 and x.q==.1
    elif i in (31,33): assert x.kp==5
    else: assert x.kp==0 and x.kd==0
for i in (29,30,32,34):assert m.motor_cmd[i].mode==0
'''
        subprocess.run([sys.executable,'-c',code],cwd=ROOT/'scripts/unitree_g1',check=True,timeout=10)
