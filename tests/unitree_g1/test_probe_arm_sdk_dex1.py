import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/unitree_g1'))
from arm_execution import Command
from hold_arms import OwnershipLost
from probe_arm_sdk_dex1 import ProbeIO, encode, request, gripper_feedback, require_passive_baseline
from probe_arm_sdk_dex1 import torque_request, require_stock_torque_baseline
from probe_arm_sdk_dex1 import position_hold_request, guard_position_downstream
from quest_bridge import load_quest_profile
from hold_arms import run_hold
from test_hold_arms import FakeIO


class ProbeTests(unittest.TestCase):
    def test_position_hold_then_tiny_move_and_cleanup(self):
        for t in (1., 1.5, 1.999):
            row = position_hold_request(-2., t, 'ACTIVE')
            self.assertEqual((row['q'], row['kp'], row['kd']), (-2., 5., .05))
        qs = [position_hold_request(-2., 2+i*.001, 'ACTIVE')['q'] for i in range(2001)]
        vs = [(b-a)/.001 for a,b in zip(qs, qs[1:])]
        self.assertAlmostEqual(qs[-1]-qs[0], .01)
        self.assertGreaterEqual(min(vs), 0.)
        self.assertLessEqual(max(vs), .007501)
        self.assertLessEqual(max(abs(b-a)/.001 for a,b in zip(vs, vs[1:])), .015001)
        for phase,t in [('ACTIVE',5.),('FAULT',3.),('RELEASING',3.),('ENGAGING',0.)]:
            row = position_hold_request(-2., t, phase)
            self.assertEqual((row['kp'],row['kd'],row['tau']), (0.,0.,0.))

    def test_position_downstream_accepts_ignored_or_bounded_forwarded_only(self):
        row = {'grippers':{str(i):dict(mode=1.,q=0.,dq=0.,kp=0.,kd=0.,tau=.3) for i in (31,33)}}
        guard_position_downstream(row, -2.)
        row['grippers']['31'].update(q=-1.99,kp=5.,kd=.05)
        guard_position_downstream(row, -2.)
        for side,key,value in [('31','q',0.),('31','kp',6.),('31','tau',.4),('33','q',.01)]:
            original = row['grippers'][side][key]
            row['grippers'][side][key] = value
            with self.assertRaises(ValueError): guard_position_downstream(row, -2.)
            row['grippers'][side][key] = original

    def test_stock_baseline_is_exact_not_general_interlock_bypass(self):
        row={'grippers':{str(i):dict(mode=1.,q=0.,dq=0.,kp=0.,kd=0.,tau=.3) for i in (31,33)}}
        require_stock_torque_baseline(row)
        for key,value in [('tau',.4),('tau',math.nan),('kp',1.),('kd',.05),('mode',0.),('q',1.)]:
            original=row['grippers']['31'][key]
            row['grippers']['31'][key]=value
            with self.assertRaises(ValueError):require_stock_torque_baseline(row)
            row['grippers']['31'][key]=original

    def test_pulse_has_bounded_duration_magnitude_and_zero_cleanup(self):
        for phase in ('ENGAGING','RELEASING','RELEASED','FAULT'):
            self.assertEqual(torque_request(-2.,1.5,phase)['tau'],0.)
        _,config=load_quest_profile()
        cmd=Command((0.,)*14,config.kp,config.kd,5,'ACTIVE','')
        for t in (0.,1.,1.1,1.3,1.7,1.9,2.,4.):
            grip=torque_request(-2.,t,'ACTIVE')
            self.assertTrue(0. <= grip['tau'] <= .1)
            self.assertEqual((grip['kp'],grip['kd']),(0.,0.))
            if t<1. or t>=2.:self.assertEqual(grip['tau'],0.)
            msg=encode(cmd,config,(0.,)*3,grip)
            self.assertEqual(msg.motor_cmd[31].tau,grip['tau'])
            self.assertEqual(msg.motor_cmd[33].mode,0)
            self.assertEqual(msg.motor_cmd[33].tau,0.)

    def test_zero_gains_do_not_hide_existing_feedforward(self):
        for side in ('31','33'):
            row={'grippers':{str(i):dict(kp=0.,kd=0.,tau=0.) for i in (31,33)}}
            require_passive_baseline(row)
            for tau in (.3, -.3, math.nan):
                row['grippers'][side]['tau']=tau
                with self.assertRaisesRegex(ValueError,'Existing downstream'):
                    require_passive_baseline(row)

    def test_executor_cleans_up_on_stop_fault_and_logging_failure(self):
        for kind,failure in [(k,f) for k in ('position','torque-pulse','position-hold')
                             for f in ('none', 'stop', 'guard', 'logging', 'ownership')]:
            profile, config = load_quest_profile()
            profile['hold_s'] = 6.
            class IO(FakeIO):
                def write(self, command):
                    ok = ProbeIO.write(self, command)
                    return ok and super().write(command)
            io=IO();io.config=config;io.waist_q=(0.,)*3;io.initial=(-2.,0.);io.probe_kind=kind
            io.active_since=None;io.fault=None;io.last_probe=None
            io.live=True;io.matched=1;packets=[]
            io.writer=SimpleNamespace(write=lambda msg:packets.append((io.t,msg)))
            def guard(now):
                if failure=='guard' and now>4.:raise ValueError('body fault')
            io.guard=guard
            if failure=='ownership':io.foreign_at=4.
            def log(row):
                if failure=='logging' and io.t>4.:raise OSError('disk failed')
            with patch('probe_arm_sdk_dex1.time.monotonic',side_effect=lambda:io.t):
                result=run_hold(io,config,profile,lambda:failure=='stop' and io.t>4.,
                                log,lambda:io.t,io.sleep)
            self.assertTrue(any((msg.motor_cmd[31].tau if kind=='torque-pulse'
                                 else msg.motor_cmd[31].kp)>0 for _,msg in packets))
            if failure=='ownership':
                self.assertFalse(result['unload_submitted'])
                self.assertTrue(all(t<4. for t,_ in packets))
            else:
                self.assertTrue(result['unload_submitted'])
                last=packets[-1][1]
                self.assertEqual((last.motor_cmd[31].kp,last.motor_cmd[31].kd,last.motor_cmd[29].q),(0.,0.,0.))
                self.assertEqual(last.motor_cmd[31].tau,0.)

    def test_trajectory_bounds_and_speed_acceleration(self):
        dt = .001
        qs = [request(-2., 1+i*dt, 'ACTIVE')['q'] for i in range(3001)]
        vs = [(b-a)/dt for a,b in zip(qs, qs[1:])]
        acc = [(b-a)/dt for a,b in zip(vs, vs[1:])]
        self.assertAlmostEqual(qs[-1]-qs[0], .1)
        self.assertGreaterEqual(min(vs), 0.)
        self.assertLessEqual(max(vs), .050001)
        self.assertLessEqual(max(map(abs, acc)), .066668)

    def test_explicit_off_independent_of_arm_authority(self):
        for phase, elapsed in [('ENGAGING', 0), ('ACTIVE', 4.5),
                               ('RELEASING', 2), ('RELEASED', 3), ('FAULT', 2)]:
            row = request(-2., elapsed, phase)
            self.assertEqual((row['kp'], row['kd']), (0., 0.))
        self.assertEqual(request(-2., .5, 'ACTIVE')['q'], -2.)
        for val in (math.nan, math.inf, -1.):
            with self.assertRaises(ValueError): request(-2., val, 'ACTIVE')

    def test_packet_is_left_only_and_crc_valid(self):
        from unitree_sdk2py.utils.crc import CRC
        _, config = load_quest_profile()
        command = Command((0.,)*14, config.kp, config.kd, 5, 'ACTIVE', '')
        for elapsed in (0., .5, 2., 4.5, 6.):
            grip = request(-2., elapsed, 'ACTIVE')
            msg = encode(command, config, (0.,)*3, grip)
            self.assertEqual(msg.crc, CRC().Crc(msg))
            self.assertEqual(msg.motor_cmd[29].q, 1.)
            self.assertEqual(msg.motor_cmd[31].kp, grip['kp'])
            for i in list(range(12))+[30,32,33,34]:
                m = msg.motor_cmd[i]
                self.assertEqual((m.mode,m.q,m.dq,m.tau,m.kp,m.kd), (0,0,0,0,0,0))

    def test_bad_feedback_rejected(self):
        telemetry = dict(received=1., motors=[dict(index=i,q=-2.,dq=0.,motorstate=0)
                                             for i in (31,33)])
        self.assertEqual(len(gripper_feedback(telemetry, 1.1)), 2)
        with self.assertRaises(ValueError): gripper_feedback(telemetry, 1.3)
        telemetry['motors'][0]['q'] = math.nan
        with self.assertRaises(ValueError): gripper_feedback(telemetry, 1.1)

    def test_failure_cleanup_still_zeros_gripper_but_ownership_loss_blocks(self):
        io = ProbeIO.__new__(ProbeIO)
        _, io.config = load_quest_profile()
        io.initial=(-2.,0.);io.active_since=0.;io.fault=None;io.waist_q=(0.,)*3
        io.live=True;io.matched=1;io.last_probe=None;io.check=lambda:None
        sent=[];io.writer=SimpleNamespace(write=sent.append)
        def bad_guard(now): raise ValueError('feedback failure')
        io.guard=bad_guard
        command=Command((0.,)*14,io.config.kp,io.config.kd,5,'ACTIVE','')
        with patch('probe_arm_sdk_dex1.time.monotonic',return_value=2.):
            with self.assertRaises(ValueError):io.write(command)
            self.assertEqual(sent,[])
            release=Command(command.q,command.kp,command.kd,5,'RELEASING','fault')
            self.assertTrue(io.write(release))
            self.assertEqual(sent[-1].motor_cmd[31].kp,0.)
            self.assertEqual(sent[-1].motor_cmd[31].kd,0.)
            self.assertEqual(sent[-1].motor_cmd[29].q,1.)
            def lost():raise OwnershipLost('lost')
            io.check=lost
            with self.assertRaises(OwnershipLost):io.write(release)
            self.assertEqual(len(sent),1)


if __name__ == '__main__':
    unittest.main()
