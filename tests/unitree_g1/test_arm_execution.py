"""Synthetic limits/gains below are test fixtures, never hardware settings."""
import sys
import subprocess
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/unitree_g1'))
from arm_execution import ArmExecution, Command, Config, Observation, encode_debug_command


def config():
    return Config((-2.,)*14, (2.,)*14, (5.,)*14, (1.,)*14,
                  (.2,)*14, (.5,)*14, (.3,)*14, .1, .05, .05, .02)


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.sent = []
        self.executor = ArmExecution(config(), lambda cmd: self.sent.append(cmd) is None)
        self.t = 0.
        self.tick = 0

    def update(self, q=None, mode=5):
        self.tick += 1
        self.executor.observe(Observation(q or self.executor.q or (.4,)*14, self.t, self.tick, mode))

    def advance(self, feedback=True):
        self.t += .01
        if feedback:
            self.update()
        return self.executor.step(self.t)

    def engage(self):
        self.update()
        self.executor.request_control(self.t)
        for _ in range(10):
            self.advance()
        self.assertEqual(self.executor.phase, 'ACTIVE')

    def test_no_constructor_output_and_measured_start(self):
        self.assertEqual(self.sent, [])
        self.engage()
        self.assertEqual(self.sent[0].q, (.4,)*14)
        self.assertEqual(self.sent[0].kp, (0.,)*14)
        self.assertTrue(all(x.q == (.4,)*14 for x in self.sent))
        self.assertEqual(self.sent[-1].kp, (5.,)*14)

    def test_targets_blocked_during_ramp(self):
        self.update()
        self.executor.request_control(0.)
        with self.assertRaises(RuntimeError):
            self.executor.send_targets((0.,)*14, .01, .01)

    def test_limits_velocity_acceleration_and_reversal(self):
        self.engage()
        previous, v = self.executor.q, (0.,)*14
        for index in range(200):
            target = (.7 if index < 100 else .3,)*14
            self.executor.send_targets(target, self.t+.001, self.t+.001)
            command = self.advance()
            speed = tuple((b-a)/.01 for a,b in zip(previous, command.q))
            self.assertLessEqual(max(abs(x) for x in speed), .2+1e-10)
            self.assertLessEqual(max(abs(a-b)/.01 for a,b in zip(speed,v)), .5+1e-8)
            previous, v = command.q, speed

    def test_release_freezes_target_and_unloads_without_home(self):
        self.engage()
        self.executor.release_control()
        for _ in range(11):
            self.advance()
        self.assertEqual(self.executor.phase, 'DISARMED')
        self.assertTrue(all(c.q == (.4,)*14 for c in self.sent))
        self.assertEqual(self.sent[-1].kp, (0.,)*14)
        self.assertEqual(self.sent[-1].kd, (0.,)*14)

    def test_frozen_tick_does_not_refresh_and_fault_latches(self):
        self.engage()
        for _ in range(20):
            self.t += .01
            self.executor.observe(Observation((.4,)*14, self.t, self.tick, 5))
            self.executor.step(self.t)
        self.assertEqual(self.executor.phase, 'FAULT')
        self.assertEqual(self.sent[-1].kp, (0.,)*14)
        self.update()
        with self.assertRaises(RuntimeError):
            self.executor.request_control(self.t)

    def test_invalid_target_unloads(self):
        self.engage()
        for target in ((float('nan'),)*14, (3.,)*14, (0.,)*13):
            fresh = ArmExecution(config(), lambda c: True)
            fresh.observe(Observation((.4,)*14, 0., 1, 5))
            fresh.request_control(0.)
            fresh.phase = 'ACTIVE'
            with self.assertRaises(ValueError):
                fresh.send_targets(target, .001, .001)
            self.assertEqual(fresh.phase, 'RELEASING')
            self.assertTrue(fresh.fault)

    def test_target_timeout_mode_change_and_deadline(self):
        for reason in ('stale_target', 'mode_machine_changed', 'servo_deadline'):
            self.setUp()
            self.engage()
            if reason == 'stale_target':
                for _ in range(7):
                    self.advance()
            elif reason == 'mode_machine_changed':
                self.t += .01
                self.update(mode=6)
                self.executor.step(self.t)
            else:
                self.t += .03
                self.update()
                self.executor.step(self.t)
            self.assertTrue(self.executor.fault)
            self.assertEqual(self.executor.reason, reason)

    def test_transport_failure_not_reported_as_release(self):
        self.update()
        self.executor.emit = lambda c: False
        with self.assertRaises(OSError):
            self.executor.request_control(0.)
        self.assertEqual(self.executor.phase, 'FAULT')
        self.assertEqual(self.executor.reason, 'transport_failure')

    def test_tracking_fault_records_original_phase_and_preserves_first_reason(self):
        for phase in ('ENGAGING', 'ACTIVE', 'RELEASING'):
            with self.subTest(phase=phase):
                self.setUp()
                if phase == 'ENGAGING':
                    self.update()
                    self.executor.request_control(self.t)
                    self.advance()
                else:
                    self.engage()
                    if phase == 'RELEASING':
                        self.executor.release_control()
                scale = self.executor.scale
                self.t += .01
                self.update(q=(.8,)+(.4,)*13)
                self.executor.step(self.t)
                evidence = self.executor.first_fault
                self.assertEqual(evidence.phase, phase)
                self.assertEqual(evidence.reason, 'tracking_error')
                self.assertEqual(evidence.gain_scale, scale)
                self.assertEqual(evidence.tracking_exceeded_arm_indices, (15,))
                self.assertEqual(self.executor.release_reason,
                                 'operator_release' if phase == 'RELEASING' else 'tracking_error')
                self.executor.stop()
                for _ in range(15):
                    self.advance(feedback=False)
                self.assertIs(self.executor.first_fault, evidence)
                self.assertEqual(self.executor.reason, 'tracking_error')
                self.assertEqual(self.executor.phase, 'FAULT')
                self.assertEqual(self.sent[-1].kp, (0.,)*14)
                with self.assertRaises(RuntimeError):
                    self.executor.request_control(self.t)

    def test_release_keeps_freshness_protection(self):
        self.engage()
        self.executor.release_control()
        for _ in range(15):
            self.advance(feedback=False)
        self.assertEqual(self.executor.first_fault.reason, 'stale_state')
        self.assertEqual(self.executor.first_fault.phase, 'RELEASING')
        self.assertEqual(self.executor.phase, 'FAULT')

    def test_missing_or_stale_state(self):
        with self.assertRaises(RuntimeError):
            self.executor.request_control(0.)
        self.update()
        with self.assertRaises(RuntimeError):
            self.executor.request_control(1.)

    def test_config_rejects_invalid_limits(self):
        values = dict(config().__dict__)
        values['velocity'] = (float('nan'),)*14
        with self.assertRaises(ValueError):
            Config(**values)


class BrakingTests(unittest.TestCase):
    def test_old_boundary_failure_and_new_braking(self):
        import math
        from arm_execution import bounded_joint_step
        q = old_q = 0.
        v = old_v = 0.
        old_failed = False
        for _ in range(6000):
            dt = .01
            error = -.2-old_q
            desired = math.copysign(min(.05, math.sqrt(.2*abs(error))), error)
            old_v += max(-.1*dt, min(.1*dt, desired-old_v))
            old_q += old_v*dt
            old_failed |= old_q < -.2
            next_q, next_v = bounded_joint_step(q, -.2, v, .05, .1, -.2, .2, dt)
            self.assertGreaterEqual(next_q, -.2)
            self.assertLessEqual(abs(next_v-v), .1*dt+1e-12)
            q, v = next_q, next_v
        self.assertTrue(old_failed)
        self.assertAlmostEqual(q, -.2, places=10)
        self.assertAlmostEqual(v, 0., places=10)

    def test_long_follow_jitter_reversals_and_both_boundaries(self):
        import random
        from arm_execution import bounded_joint_step
        rng = random.Random(42)
        q = v = 0.
        elapsed = 0.
        minimum = maximum = q
        # Ten simulated minutes, irregular servo periods and abrupt reversals.
        while elapsed < 600.:
            dt = rng.uniform(.002, .049)
            target = (.2 if int(elapsed/12)%2 == 0 else -.2)
            if int(elapsed)%12 == 5:
                target = rng.uniform(-.2, .2)
            qn, vn = bounded_joint_step(q, target, v, .05, .1, -.2, .2, dt)
            self.assertTrue(-.2 <= qn <= .2)
            self.assertLessEqual(abs(vn), .05+1e-12)
            self.assertLessEqual(abs(vn-v), .1*dt+1e-12)
            remaining = .2-qn if vn >= 0 else qn+.2
            self.assertLessEqual(vn*vn/.2, remaining+1e-12)
            minimum, maximum = min(minimum, qn), max(maximum, qn)
            q, v = qn, vn
            elapsed += dt
        self.assertLess(minimum, -.199)
        self.assertGreater(maximum, .199)

    def test_infeasible_state_is_not_silently_clipped(self):
        from arm_execution import bounded_joint_step
        with self.assertRaises(ValueError):
            bounded_joint_step(.2, .2, .05, .05, .1, -.2, .2, .01)


class SDKEncodingTests(unittest.TestCase):
    def test_only_arm_slots_have_gains_and_crc_matches(self):
        import importlib.util
        if importlib.util.find_spec('unitree_sdk2py') is None:
            self.skipTest('Requires pinned SDK environment')
        # Keep DDS imports isolated from virtual-arm tests' import guard.
        code = '''
import sys
def no_network(event, args):
    if event in ('socket.bind', 'socket.connect'):
        raise RuntimeError('No network in encoder test')
sys.addaudithook(no_network)
from arm_execution import Command, encode_debug_command
from unitree_sdk2py.utils.crc import CRC
msg = encode_debug_command(Command((.4,)*14, (5.,)*14, (1.,)*14, 5, 'ACTIVE', ''))
crc = CRC()
assert msg.crc == crc.Crc(msg)
assert msg.crc == crc._crc_py(crc._CRC__PackHGLowCmd(msg))
assert msg.mode_machine == 5
for i, motor in enumerate(msg.motor_cmd):
    assert motor.tau == 0 and motor.dq == 0
    assert motor.kp == (5. if 15 <= i < 29 else 0.)
    assert motor.kd == (1. if 15 <= i < 29 else 0.)
'''
        subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[2] / 'scripts/unitree_g1', check=True)


if __name__ == '__main__':
    unittest.main()
