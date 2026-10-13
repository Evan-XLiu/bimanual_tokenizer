import json
from functools import wraps
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/unitree_g1'))
from arm_execution import Command
from arm_sdk import ArmSDKIO, STANDING, check_body, encode_arm_sdk_command
from hold_arms import OwnershipLost, run_hold
from quest_bridge import QuestSource, load_quest_profile, open_grippers_before_teleoperation
import test_hold_arms


def sdk_isolated(test):
    @wraps(test)
    def run(self):
        if __name__ == '__main__':
            return test(self)
        result = subprocess.run([sys.executable, __file__, f'ArmSDKTests.{test.__name__}'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
    return run


def body():
    return dict(received=1., imu_rpy=[0., 0., 0.], imu_gyroscope=[0., 0., 0.],
                motors=[dict(index=i, q=.01*i, dq=0., motorstate=0) for i in range(35)])


class ArmSDKTests(unittest.TestCase):
    def setUp(self):
        self.profile, self.config = load_quest_profile()

    @sdk_isolated
    def test_packet_only_populates_upper_body_and_weight_with_crc(self):
        from unitree_sdk2py.utils.crc import CRC
        for weight in (0., .25, 1.):
            cmd = Command((.4,)*14, tuple(weight*x for x in self.config.kp),
                          tuple(weight*x for x in self.config.kd), 5, 'ENGAGING', '')
            msg = encode_arm_sdk_command(cmd, self.config, (.1, -.1, .05))
            self.assertAlmostEqual(msg.motor_cmd[29].q, weight)
            self.assertEqual(msg.crc, CRC().Crc(msg))
            self.assertEqual((msg.mode_machine, msg.mode_pr), (5, 0))
            for i in list(range(12))+list(range(30,35)):
                m = msg.motor_cmd[i]
                self.assertEqual((m.q,m.dq,m.kp,m.kd,m.tau,m.mode), (0.,0.,0.,0.,0.,0))
            self.assertEqual([msg.motor_cmd[i].q for i in range(12,15)], [.1,-.1,.05])
            self.assertEqual(msg.motor_cmd[15].kp, self.config.kp[0])
            self.assertEqual(msg.motor_cmd[12].kp, STANDING['waist_kp'])

    @sdk_isolated
    def test_rejects_nonuniform_or_invalid_authority(self):
        for gain in (float('nan'), -1., 1000., 40.):
            cmd = Command((0.,)*14, (gain,)+self.config.kp[1:], self.config.kd, 5, 'ACTIVE', '')
            with self.assertRaises(ValueError):
                encode_arm_sdk_command(cmd, self.config, (0.,)*3)

    @sdk_isolated
    def test_transport_shadow_and_failed_submission_do_not_claim_live_write(self):
        io=ArmSDKIO.__new__(ArmSDKIO)
        io.config=self.config;io.waist_q=(.1,0.,0.);io.last_sdk=None
        io.check=lambda:None;io.live=False;io.writer=None;io.matched=0
        cmd=Command((.4,)*14,self.config.kp,self.config.kd,5,'ACTIVE','')
        self.assertTrue(io.write(cmd))
        self.assertEqual(io.command_metadata()['arm_sdk_request']['weight'],1.)
        self.assertEqual(io.last_sdk['joint_indices'],list(range(12,29)))
        io.last_sdk=None;io.live=True
        self.assertFalse(io.write(cmd))
        self.assertIsNone(io.last_sdk)
        submitted=[]
        io.writer=SimpleNamespace(write=submitted.append);io.matched=1
        self.assertTrue(io.write(cmd))
        self.assertEqual(len(submitted),1)
        self.assertEqual(submitted[0].motor_cmd[29].q,1.)
        io.check=lambda:(_ for _ in ()).throw(OwnershipLost('mode lost'))
        with self.assertRaises(OwnershipLost):io.write(cmd)
        self.assertEqual(len(submitted),1)

    def test_standing_gate_requires_fresh_finite_stationary_body(self):
        check_body(body(), 1.1, .2, preflight=True)
        cases = [lambda b:b.update(received=.7), lambda b:b.update(received=1.2),
                 lambda b:b.update(imu_rpy=[.16,0.,0.]),
                 lambda b:b.update(imu_gyroscope=[0.,.21,0.]),
                 lambda b:b.update(imu_rpy=[float('nan'),0.,0.]),
                 lambda b:b['motors'][0].update(dq=.16),
                 lambda b:b['motors'][12].update(motorstate=1)]
        for mutate in cases:
            sample=body();mutate(sample)
            with self.assertRaises(ValueError):check_body(sample,1.1,.2,preflight=True)

    def test_active_gate_allows_balance_corrections_but_rejects_tilt_and_waist_drift(self):
        sample=body();sample['motors'][0]['dq']=.2
        check_body(sample,1.1,.2)
        sample['imu_rpy'][0]=.31
        with self.assertRaisesRegex(ValueError,'tilt'):check_body(sample,1.1,.2)
        with self.assertRaisesRegex(ValueError,'Waist'):check_body(body(),1.1,.2,waist_q=(1.,)*3)

    def test_mode_and_ownership_gate_never_requires_quiet_official_lowcmd(self):
        self.assertEqual(ArmSDKIO.command_topic_name, 'rt/arm_sdk')
        self.assertNotIn('rt/lowcmd', ArmSDKIO.monitored_topics)
        io=ArmSDKIO.__new__(ArmSDKIO)
        io.lock=threading.Lock();io.mode_ok=True;io.mode_time=1.;io.mode_evidence={}
        io.monitor_lost=False;io.readers=[];io.own_handle=11
        with patch('arm_sdk.time.monotonic',return_value=1.1):io.check()
        for ok, now in ((False,1.1),(True,2.61)):
            io.mode_ok=ok
            with patch('arm_sdk.time.monotonic',return_value=now),self.assertRaises(OwnershipLost):io.check()
        io.mode_ok=True
        sample=SimpleNamespace(sample_info=SimpleNamespace(valid_data=True,publication_handle=12))
        io.readers=[SimpleNamespace(take=lambda _: [sample])]
        with patch('arm_sdk.time.monotonic',return_value=1.1),self.assertRaises(OwnershipLost):io.check()

    def test_poll_checks_service_and_fsm_without_mode_writes(self):
        for service, fsm_code, fsm, expected in [('ai',0,500,True),('',0,500,False),
                                                ('ai',3102,None,False),('ai',0,0,False),
                                                ('ai',0,706,False),('ai',0,501,True),
                                                ('ai',0,4,False),('ai',0,200,False),('ai',0,801,False)]:
            io=ArmSDKIO.__new__(ArmSDKIO);io.lock=threading.Lock();io.done=threading.Event()
            io.client=SimpleNamespace(CheckMode=lambda:(0,{'name':service}))
            def query():
                io.done.set()
                return fsm_code,fsm
            io.loco=SimpleNamespace(GetFsmId=query)
            io._poll_mode()
            self.assertEqual(io.mode_ok,expected)

    def test_mode_change_during_takeover_is_rejected(self):
        io=ArmSDKIO.__new__(ArmSDKIO)
        io.lock=threading.Lock();io.mode_ok=True;io.mode_time=1.
        io.engaged_fsm=501;io.mode_evidence={'fsm_id':501}
        with patch('arm_sdk.time.monotonic',return_value=1.1):
            io._check_mode()
            io.mode_evidence={'fsm_id':500}
            with self.assertRaisesRegex(OwnershipLost,'changed'):
                io._check_mode()

    @sdk_isolated
    def test_sdk_release_uses_weight_ramp_on_stop_fault_and_logging_failure(self):
        # Drive the real executor and encoder without DDS or hardware. Nominal
        # gains stay present as weight returns to zero, unlike debug unloading.
        fixture=test_hold_arms.HoldTests();fixture.setUp()
        config, profile=fixture.config,fixture.profile
        for failure in ('none','stop','feedback','logging','ownership'):
            class IO(test_hold_arms.FakeIO):
                def write(self, command):
                    self.messages.append(encode_arm_sdk_command(command,config,(0.,)*3))
                    return super().write(command)
            io=IO();io.messages=[]
            if failure=='feedback':io.frozen=True
            if failure=='ownership':io.foreign_at=.15
            def log(row):
                if failure=='logging' and io.t>.05:raise OSError('disk full')
            result=run_hold(io,config,profile,lambda:failure=='stop' and io.t>.15,log,
                            lambda:io.t,io.sleep)
            weights=[m.motor_cmd[29].q for m in io.messages]
            self.assertEqual(weights[0],0.)
            self.assertGreater(max(weights),0.)
            if failure=='ownership':
                self.assertFalse(result['unload_submitted'])
                self.assertTrue(all(t<.15 for t,_ in io.commands))
            else:
                self.assertTrue(result['unload_submitted'])
                self.assertEqual(weights[-1],0.)
                self.assertEqual(io.messages[-1].motor_cmd[15].kp,config.kp[0])

    def test_stability_fault_requests_release_even_without_new_quest_packet(self):
        with tempfile.TemporaryDirectory() as directory:
            source=QuestSource(Path(directory),'x',lambda:1.)
            source.body_checker=lambda:check_body(dict(body(),imu_rpy=[.4,0.,0.]),1.1,.2)
            faults=[]
            executor=SimpleNamespace(observation=SimpleNamespace(received=1.,q=(0.,)*14),
                                     q=(0.,)*14,phase='ACTIVE',_stop=lambda *args:faults.append(args))
            source.update(executor,1.)
            self.assertIn('Body tilt',faults[0][0])
            self.assertTrue(faults[0][1])

    def test_internal_dex1_rejected_before_preopen_or_session_creation(self):
        with self.assertRaisesRegex(ValueError,'Dex1'):
            open_grippers_before_teleoperation(SimpleNamespace(control_mode='arm_sdk'),None,None)
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'session'
            script=Path(__file__).resolve().parents[2]/'scripts/unitree_g1/quest_bridge.py'
            result=subprocess.run([sys.executable,str(script),'--interface','unused','--cert','absent',
                                   '--key','absent','--host-address','127.0.0.1','--output',str(output),
                                   '--grippers','--execute'],capture_output=True,text=True)
            self.assertEqual(result.returncode,2)
            self.assertIn('arm_sdk cannot control internal Dex1',result.stderr)
            self.assertFalse(output.exists())

    def test_raw_sdk_recording_cannot_be_exported_as_16d_motor_actions(self):
        from convert_dataset import plan_episode
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);(p/'session.json').write_text(json.dumps(dict(control_mode='arm_sdk')))
            with self.assertRaisesRegex(ValueError,'raw-only'):
                plan_episode(p,include_failed=True,allow_shadow=True)


if __name__=='__main__':
    unittest.main()
