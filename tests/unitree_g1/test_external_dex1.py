import math
from pathlib import Path
import sys
import threading
import tempfile
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/unitree_g1'))
from arm_execution import Command
from arm_sdk import ArmSDKIO, encode_arm_sdk_command
from external_dex1 import ExternalFollower, ExternalDex1, ExternalDex1ArmSDKIO, external_settings, encode_external, preopen_external
from hold_arms import OwnershipLost
from quest_bridge import load_quest_profile


def telemetry(now, q=(2., 2.)):
    return dict(received=now, motors=[dict(index=i,q=x,dq=0.,motorstate=0) for i,x in enumerate(q)])


def command(phase='ACTIVE', scale=1.):
    _, c = load_quest_profile()
    return Command((0.,)*14, tuple(scale*k for k in c.kp),tuple(scale*k for k in c.kd),5,phase,'')


class ExternalTests(unittest.TestCase):
    def test_telemetry_captures_external_state_without_later_list_updates(self):
        io=ExternalDex1ArmSDKIO.__new__(ExternalDex1ArmSDKIO)
        io.external=SimpleNamespace(state_lock=threading.Lock(),latest=[dict(q=1.),dict(q=2.)])
        with patch.object(ArmSDKIO,'telemetry',return_value=dict(received=10.,motors=[])):
            before=io.telemetry()
            io.external.latest[0]=dict(q=9.)
        self.assertEqual(before['external_grippers'][0]['q'],1.)

    def test_external_trigger_coordinates_and_no_internal_slots(self):
        f = ExternalFollower(external_settings())
        f.initialize(telemetry(0.),0.)
        f.update([1.,0.],.01,.01)
        row=f.step(command(),telemetry(.01),.01,80.)
        self.assertLess(row['q'][0],2.)
        self.assertGreater(row['q'][1],2.)
        self.assertEqual(row['target'],[0.,5.4])
        msg=encode_external(row,0)
        self.assertEqual(len(msg.cmds),1)
        self.assertEqual((msg.cmds[0].q,msg.cmds[0].kp,msg.cmds[0].kd),(row['q'][0],5.,.05))
        _,c=load_quest_profile()
        arm=encode_arm_sdk_command(command(),c,(0.,)*3)
        for i in list(range(12))+[30,31,32,33,34]:
            m=arm.motor_cmd[i]
            self.assertEqual((m.q,m.kp,m.kd,m.tau,m.mode),(0.,0.,0.,0.,0))

    def test_closing_blocked_sign_is_correct_and_open_request_releases(self):
        f=ExternalFollower(external_settings());f.initialize(telemetry(0.),0.)
        f.update([1.,1.],.01,.01)
        # Command service q=2, feedback q=2.6: closing lag is +.6 in canonical coordinates.
        row=f.step(command(),telemetry(.01,(2.6,2.)),.01,80.)
        self.assertEqual(row['states'][0],'CLOSING_BLOCKED')
        self.assertEqual(row['q'][0],2.)
        f.update([0.,1.],.02,.02)
        row=f.step(command(),telemetry(.02,(2.6,2.)),.02,80.)
        self.assertGreater(row['q'][0],2.)
        self.assertEqual(row['states'][0],'OPENING')

    def test_tracking_fault_disabled_but_closing_obstruction_preserved(self):
        self.assertIsNone(external_settings()['tracking_error'])
        self.assertEqual(external_settings()['closing_error'], .5)
        for side in (0, 1):
            for opening in (False, True):
                for error in (.301, .499, .501, 1.5):
                    f=ExternalFollower(external_settings())
                    f.initialize(telemetry(0.),0.)
                    f.update([0. if opening else 1.]*2,.01,.01)
                    actual=[2.,2.]
                    actual[side] += -error if opening else error
                    row=f.step(command(),telemetry(.01,actual),.01,80.)
                    self.assertIsNone(f.fault)
                    self.assertEqual(row['states'][side]=='CLOSING_BLOCKED', not opening and error > .5)

    def test_input_one_second_and_feedback_point_two_are_independent(self):
        f=ExternalFollower(external_settings())
        f.initialize(telemetry(0.),0.)
        f.update([1.,1.],0.,.999)
        with self.assertRaisesRegex(ValueError,'Stale gripper input'):
            f.update([1.,1.],.001,1.002)
        with self.assertRaisesRegex(ValueError,'Stale gripper feedback'):
            f.feedback(telemetry(0.),.201)
        # Fresh feedback does not refresh the held trigger target.
        f=ExternalFollower(external_settings())
        f.initialize(telemetry(0.),0.)
        f.update([1.,1.],0.,0.)
        actual=[2.,2.]
        for step in range(1,101):
            row=f.step(command(),telemetry(step*.01,actual),step*.01,80.)
            actual=row['q']
        with self.assertRaisesRegex(ValueError,'Stale gripper target'):
            f.step(command(),telemetry(1.01,actual),1.01,80.)

    def test_feedback_and_command_bounds(self):
        f=ExternalFollower(external_settings())
        for q in ((-1.,0.),(6.,0.),(math.nan,0.)):
            with self.assertRaises(ValueError):f.initialize(telemetry(0.,q),0.)
        for key,val in [('q',-1.),('q',6.),('q',math.nan),('kp',6.),('kd',.1)]:
            row=dict(q=[2.,2.],kp=[5.,5.],kd=[.05,.05]);row[key][0]=val
            with self.assertRaises(ValueError):encode_external(row,0)

    def test_receipt_age_is_not_refreshed_by_reads(self):
        x=ExternalDex1.__new__(ExternalDex1)
        x.state_lock=threading.Lock();x.latest=[None,None];x.counts=[0,0];x.feedback_error=None
        m=SimpleNamespace(q=2.,dq=0.,tau_est=0.,lost=0)
        with patch('external_dex1.time.monotonic',return_value=1.):
            x._state(0,SimpleNamespace(states=[m]));x._state(1,SimpleNamespace(states=[m]))
        self.assertEqual(x.snapshot(1.1)['received'],1.)
        with self.assertRaisesRegex(ValueError,'stale'):x.snapshot(1.3)
        x._state(0,SimpleNamespace(states=[]))
        with self.assertRaisesRegex(ValueError,'one motor'):x.snapshot(1.1)

    def test_foreign_traffic_and_monitor_loss_prevent_commands(self):
        x=ExternalDex1.__new__(ExternalDex1);x.monitor_lost=False;x.handles=[1,2]
        sample=lambda handle:SimpleNamespace(sample_info=SimpleNamespace(valid_data=True,publication_handle=handle))
        x.monitors=[SimpleNamespace(take=lambda n:[sample(1)]),SimpleNamespace(take=lambda n:[sample(2)])]
        x.check()
        x.monitors[1]=SimpleNamespace(take=lambda n:[sample(3)])
        with self.assertRaises(OwnershipLost):x.check()
        x.monitor_lost=True
        with self.assertRaises(OwnershipLost):x.check()

    def test_partial_write_cleanup_clears_successful_side(self):
        x=ExternalDex1.__new__(ExternalDex1);x.live=True;x.check=lambda:None
        x.last_submitted=[None,None];x.zero_submitted=[False,False]
        packets=[]
        def broken(msg):raise OSError('right transport')
        x.writers=[SimpleNamespace(write=packets.append),SimpleNamespace(write=broken)]
        with self.assertRaises(OSError):x.write(dict(q=[2.,2.],kp=[5.,5.],kd=[.05,.05]))
        x.zero()
        self.assertEqual(x.zero_submitted,[True,False])
        self.assertEqual((packets[-1].cmds[0].kp,packets[-1].cmds[0].kd),(0.,0.))

    def test_cleanup_attempts_right_even_if_left_fails(self):
        x=ExternalDex1.__new__(ExternalDex1);x.check=lambda:None
        x.last_submitted=[dict(q=2.),dict(q=3.)];x.zero_submitted=[False,False]
        packets=[]
        def broken(msg):raise OSError('left transport')
        x.writers=[SimpleNamespace(write=broken),SimpleNamespace(write=packets.append)]
        with self.assertRaises(RuntimeError):x.zero()
        self.assertEqual(x.zero_submitted,[False,True])
        self.assertEqual(packets[-1].cmds[0].q,3.)

    def test_release_does_not_depend_on_gripper_feedback(self):
        io=ExternalDex1ArmSDKIO.__new__(ExternalDex1ArmSDKIO)
        io.check=lambda:None;io.grippers=ExternalFollower(external_settings())
        io.grippers.initialize(telemetry(0.),0.)
        sent=[]
        io.external=SimpleNamespace(write=sent.append,snapshot=lambda now:(_ for _ in ()).throw(ValueError('stale')))
        with patch.object(ArmSDKIO,'write',return_value=True):
            self.assertTrue(io.write(command('RELEASING',.8)))
        self.assertEqual(sent[-1]['kp'],[0.,0.])
        self.assertEqual(sent[-1]['q'],[2.,2.])

    def test_shadow_creates_no_writers(self):
        x=ExternalDex1.__new__(ExternalDex1);x.live=False;x.writers=[]
        x.check=lambda:None;x.snapshot=lambda now:telemetry(now)
        x.open();x.write(dict(q=[2.,2.],kp=[5.,5.],kd=[.05,.05]))
        self.assertEqual(x.writers,[])

    def test_open_uses_pinned_dds_publication_callback(self):
        x=ExternalDex1.__new__(ExternalDex1)
        x.live=True;x.writers=[];x.handles=[];x.matches=[0,0];x.writer_listeners=[]
        x.participant=object();x.topics=['left','right']
        x.check=lambda:None;x.snapshot=lambda now:telemetry(now)
        def writer(participant, topic, listener):
            listener.on_publication_matched(None,SimpleNamespace(current_count=2))
            return SimpleNamespace()  # No get_matched_subscriptions in pinned SDK.
        with patch('cyclonedds.pub.DataWriter',side_effect=writer),patch('cyclonedds.core.Listener',side_effect=lambda **kw:SimpleNamespace(**kw)),patch('external_dex1.writer_handle',return_value=123):
            x.open()
        self.assertEqual(x.matches,[2,2])
        self.assertEqual(len(x.writers),2)

    def test_full_open_trajectory_is_bounded_and_reaches_endpoint(self):
        f=ExternalFollower(external_settings());f.initialize(telemetry(0.,(0.,0.)),0.)
        previous=0.
        for step in range(1,1001):
            now=step*.01
            f.update([0.,0.],now,now)
            row=f.step(command(),telemetry(now,(previous,previous)),now,80.)
            self.assertTrue(0. <= row['q'][0] <= 5.4)
            self.assertLessEqual(row['q'][0]-previous,f.c['velocity']*.01+1e-9)
            previous=row['q'][0]
        self.assertAlmostEqual(previous,5.4)
        self.assertEqual(f.v,(0.,0.))

    def test_preopen_records_no_control_only_before_writer_attempt(self):
        from unittest.mock import Mock
        for stage in ('preflight', 'open_writer'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                fake = SimpleNamespace(preflight=Mock(), open_writer=Mock(), close=Mock(),
                    external=SimpleNamespace(zero_submitted=[False, False]),
                    external_cleanup_error=None)
                getattr(fake, stage).side_effect = ValueError('Stale body feedback')
                output = Path(directory)/'preopen'
                with patch('external_dex1.ExternalDex1ArmSDKIO', return_value=fake):
                    with self.assertRaisesRegex(RuntimeError, 'Stale body feedback'):
                        preopen_external('unused', output, threading.Event())
                result = json.loads((output/'summary.json').read_text())
                self.assertEqual(result['control_publish_not_started'], stage == 'preflight')
                fake.close.assert_called_once()
                if stage == 'preflight':
                    fake.open_writer.assert_not_called()

    def test_preopen_incomplete_stop_or_failure_blocks_next_stage(self):
        for kind in ('incomplete','stop','error'):
            stopped=threading.Event()
            if kind=='stop':stopped.set()
            opened=[]
            fake=SimpleNamespace(preflight=lambda:None,open_writer=lambda:opened.append(True),
                close=lambda:None,external=SimpleNamespace(zero_submitted=[False,False]),external_cleanup_error=None)
            with tempfile.TemporaryDirectory() as directory:
                output=Path(directory)/'preopen'
                with patch('external_dex1.ExternalDex1ArmSDKIO',return_value=fake),patch('hold_arms.run_hold',return_value=dict(error='fault' if kind=='error' else None,unload_submitted=True)):
                    with self.assertRaises(RuntimeError):preopen_external('unused',output,stopped)
                saved=json.loads((output/'summary.json').read_text())
                self.assertTrue(saved['error'])
                self.assertFalse(saved['unload_submitted'])
                if kind=='stop':self.assertFalse(opened)


if __name__=='__main__':unittest.main()
