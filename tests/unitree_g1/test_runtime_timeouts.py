import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/unitree_g1'))
from arm_execution import ArmExecution,Command,Observation
from hold_arms import DDSHoldIO,OwnershipLost
from quest_bridge import LIMITS,load_quest_profile,read_packet
from quest_grippers import GripperFollower,gripper_settings
from record_cameras import check_camera_status


class TimeoutTests(unittest.TestCase):
    def test_quest_packet_tolerates_previous_gap_but_still_expires(self):
        self.assertEqual(LIMITS['input_timeout_s'],.2)
        self.assertEqual(LIMITS['feedback_timeout_s'],.2)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'target.json'
            p.write_text(json.dumps(dict(nonce='x',input_time=1.,computed_time=1.01)))
            read_packet(p,'x',1.115207384)
            with self.assertRaises(ValueError):read_packet(p,'x',1.201)

    def test_executor_servo_deadline_doubled(self):
        _,c=load_quest_profile()
        self.assertEqual((c.state_timeout,c.target_timeout,c.max_step_seconds),(.2,.2,.1))
        for gap,fault in ((.075,False),(.101,True)):
            ex=ArmExecution(c,lambda _:True)
            ex.observe(Observation((0.,)*14,0.,1,5));ex.request_control(0.)
            ex.observe(Observation((0.,)*14,gap,2,5));ex.step(gap)
            self.assertEqual(ex.fault,fault)
            if fault:self.assertEqual(ex.first_fault.reason,'servo_deadline')

    def test_gripper_longer_feedback_and_servo_gap(self):
        f=GripperFollower(gripper_settings())
        def feedback(t):
            return dict(received=t,motors=[dict(index=i,q=-3.,dq=0.,motorstate=0) for i in (31,33)])
        f.initialize(feedback(0.),0.)
        f.update([.5,.5],0.,.15)
        f.feedback(feedback(0.),.15)
        cmd=Command((0.,)*14,(80.,)*14,(3.,)*14,5,'ACTIVE','')
        f.step(cmd,feedback(.075),.075,80.)
        with self.assertRaisesRegex(ValueError,'servo deadline'):
            f.step(cmd,feedback(.176),.176,80.)
        with self.assertRaisesRegex(ValueError,'Stale gripper feedback'):
            f.feedback(feedback(0.),.201)

    def test_camera_and_mode_status_expire_at_new_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'status.json'
            p.write_text(json.dumps(dict(time=1.,ready=True,error=None,
                latest={k:1. for k in ('head','left_wrist','right_wrist')})))
            check_camera_status(p,1.5)
            with self.assertRaises(ValueError):check_camera_status(p,1.601)
        import threading
        io=DDSHoldIO.__new__(DDSHoldIO)
        io.monitor_lost=False;io.lock=threading.Lock();io.mode_ok=True;io.mode_time=1.;io.readers=[]
        with patch('hold_arms.time.monotonic',return_value=2.5):io.check()
        with patch('hold_arms.time.monotonic',return_value=2.601),self.assertRaises(OwnershipLost):io.check()
