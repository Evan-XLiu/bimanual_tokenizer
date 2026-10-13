from dataclasses import replace
from pathlib import Path
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/unitree_g1'))
from whole_body_control import BodyState, PolicyResult, WholeBodyControl, BodyFault, encode_whole_body
from quest_bridge import load_quest_profile
from quest_grippers import gripper_settings
from lower_body_policy import load_candidate
from whole_body_runtime import PolicyWorker, RecordingQueue, OfflineServoSession


class WholeBodyTests(unittest.TestCase):
    def setUp(self):
        _, arm = load_quest_profile()
        _, self.leg, _ = load_candidate()
        # Exact arms use the existing profile; generous leg bounds isolate timing.
        lo, hi = [-3.]*15+list(arm.lower), [3.]*15+list(arm.upper)
        self.c = WholeBodyControl(arm, self.leg, (lo, hi), gripper_settings())
        q = self.leg['default_angles']+[0.]*23
        q[31] = q[33] = -5.4
        self.s = BodyState(0., 0, 5, tuple(q), (0.,)*35, (1., 0., 0., 0.), (0.,)*3)
        self.c.initialize(self.s, 0.)
        self.c.update_policy(self.policy(.001, 1))

    def policy(self, stamp, seq):
        return PolicyResult(stamp, stamp, seq, tuple(self.leg['default_angles']), (0.,)*47, (0.,)*12, .001)

    def tick(self, t):
        self.s = replace(self.s, stamp=t, tick=self.s.tick+1)
        if self.c.policy.stamp < t:
            self.c.update_policy(self.policy(t, self.c.policy.sequence+1))
        return self.c.step(self.s, t)

    def active(self):
        self.c.preopen(0.)
        self.tick(.001)
        self.assertEqual(self.c.phase, 'READY')
        self.c.begin_task()

    def test_preopen_and_task_exit_keep_all_body_gains(self):
        self.active()
        for reason in ('operator_release', 'camera_failure', 'recording_failure'):
            self.c.stop_upper(reason)
            frame = self.tick(self.c.last_step+.001)
            self.assertEqual(frame.upper_phase, 'UPPER_HOLD')
            self.assertTrue(all(x > 0 for x in frame.kp[:29]))
            self.assertEqual(frame.q[:12], tuple(self.leg['default_angles']))
            self.assertEqual(len(frame.action16()), 16)

    def test_quest_timeout_only_holds_upper_body(self):
        self.active()
        for i in range(1, 24):
            frame = self.tick(.001+i*.01)
        self.assertEqual(frame.upper_reason, 'upper_input_deadline')
        self.assertIsNone(self.c.body_fault)
        self.assertTrue(all(frame.kp[i] > 0 for i in range(15)))

    def test_gripper_opening_fault_does_not_unload_legs(self):
        self.active()
        q = list(self.s.q)
        q[31] = -4.9  # feedback farther closed than held open target
        self.s = replace(self.s, q=tuple(q))
        frame = self.tick(.002)
        self.assertEqual(frame.upper_phase, 'UPPER_HOLD')
        self.assertIn('gripper', frame.upper_reason)
        self.assertTrue(all(x > 0 for x in frame.kp[:12]))

    def test_stale_policy_latches_body_fault_no_zero_tail(self):
        self.tick(.002)
        # Advancing state does not make old inference fresh.
        self.c.policy = replace(self.c.policy, stamp=-.05, source_stamp=-.05)
        with self.assertRaisesRegex(BodyFault, 'policy_deadline'):
            self.c.step(replace(self.s, stamp=.003, tick=3), .003)
        with self.assertRaises(BodyFault):
            self.c.update_policy(self.policy(.004, 99))

    def test_mode_frozen_tick_tilt_motor_faults(self):
        bad = [replace(self.s, mode_machine=15),
               replace(self.s, stamp=.001),
               replace(self.s, quaternion=(.9, .435889894, 0., 0.)),
               replace(self.s, motorstate=(1,)+(0,)*34)]
        for state in bad:
            with self.subTest(state=state):
                self.c.body_fault = None
                with self.assertRaises(BodyFault):
                    self.c.step(state, .001)

    def test_policy_nan_and_deadline_are_rejected(self):
        for result in [replace(self.policy(.002, 2), target=(float('nan'),)*12),
                       replace(self.policy(.002, 2), inference_s=.021),
                       replace(self.policy(.002, 2), source_stamp=.003)]:
            self.c.body_fault = None
            with self.assertRaises(BodyFault):
                self.c.update_policy(result)

    def test_only_servo_owner_can_compose(self):
        errors = []
        def second_owner():
            try:
                self.c.step(self.s, .001)
            except RuntimeError as exc:
                errors.append(str(exc))
        thread = threading.Thread(target=second_owner)
        thread.start(); thread.join()
        self.assertEqual(len(errors), 1)
        self.assertIn('owner', errors[0])

    def test_recording_queue_full_holds_upper_and_keeps_submitting(self):
        self.active()
        queue = RecordingQueue(capacity=1)
        frames = []
        def emit(frame):
            frames.append(frame)
            return True
        session = OfflineServoSession(self.c, emit, queue, clock=lambda: .002)
        for n in range(2, 5):
            t = n*.001
            self.c.update_policy(self.policy(t, n+1))
            self.s = replace(self.s, tick=n, stamp=t)
            session.tick(self.s, t)
        self.assertEqual(len(frames), 3)
        self.assertEqual(frames[-1].upper_reason, 'recording_queue_full')
        self.assertTrue(all(x > 0 for x in frames[-1].kp[:29]))
        row = queue.queue.get_nowait()
        self.assertEqual(row['observation_before_command']['stamp'], .002)
        self.assertEqual(row['action16'], frames[0].action16())
        self.assertFalse(row['hardware_motion_enabled'])

    def test_failed_submission_does_not_log_or_retry(self):
        queue = RecordingQueue()
        session = OfflineServoSession(self.c, lambda frame: False, queue)
        state = replace(self.s, stamp=.001, tick=1)
        with self.assertRaisesRegex(BodyFault, 'submission_failed'):
            session.tick(state, .001)
        self.assertEqual(session.frames_submitted, 0)
        self.assertTrue(queue.queue.empty())
        with self.assertRaises(BodyFault):
            session.tick(replace(state, stamp=.002, tick=2), .002)

    def test_inference_worker_deadline_fault_is_independent_of_servo(self):
        now = [0.]
        done = threading.Event()
        class SlowPolicy:
            def warmup(self):
                pass
            def step(policy, *args):
                policy.previous = (0.,)*12
                policy.last_observation = (0.,)*47
                now[0] += .021
                done.set()
                return (0.,)*12
        self.c.policy = None
        worker = PolicyWorker(SlowPolicy(), self.c, clock=lambda: now[0])
        worker.start()
        worker.submit(self.s)
        self.assertTrue(done.wait(timeout=1.))
        self.assertTrue(worker.close())
        self.assertIn('invalid_inference_timing', self.c.body_fault)
        self.assertIsNone(self.c.policy)

    def test_packet_contains_body_and_internal_grippers_only(self):
        # Isolate SDK imports from fake modules installed by legacy unit tests.
        if __name__ != '__main__':
            import subprocess
            result = subprocess.run([sys.executable, __file__, 'WholeBodyTests.test_packet_contains_body_and_internal_grippers_only'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            return
        from unitree_sdk2py.utils.crc import CRC
        frame = self.tick(.001)
        msg = encode_whole_body(frame)
        self.assertEqual(msg.crc, CRC().Crc(msg))
        for i in (29, 30, 32, 34):
            m = msg.motor_cmd[i]
            self.assertEqual((m.mode, m.q, m.kp, m.kd), (0, 0, 0, 0))
        for i in list(range(29))+[31, 33]:
            self.assertEqual(msg.motor_cmd[i].q, frame.q[i])
            self.assertGreater(msg.motor_cmd[i].kp, 0)


if __name__ == '__main__':
    unittest.main()
