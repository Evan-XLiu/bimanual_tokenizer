from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/unitree_g1'))
from shadow_lower_body import FeedbackMailbox, check_posture, decode_state


def message(tick=1):
    return NS(tick=tick, mode_machine=5, mode_pr=0,
              motor_state=[NS(q=0., dq=0., motorstate=0) for _ in range(35)],
              imu_state=NS(quaternion=[1., 0., 0., 0.], gyroscope=[0., 0., 0.]))


class ShadowTests(unittest.TestCase):
    def test_imu_order_and_layout(self):
        s = decode_state(message(), 1.)
        self.assertEqual(s.quaternion, (1., 0., 0., 0.))
        self.assertEqual(len(s.q), 35)
        self.assertEqual(check_posture(s), 0.)

    def test_duplicate_does_not_refresh_age(self):
        b = FeedbackMailbox()
        b.accept(message(), 1.)
        b.accept(message(), 1.03)
        with self.assertRaisesRegex(ValueError, 'feedback_deadline'):
            b.latest(1.05)

    def test_reset_latches_but_wrap_is_allowed(self):
        b = FeedbackMailbox()
        b.accept(message(2**32-1), 1.)
        b.accept(message(0), 1.01)
        self.assertEqual(b.latest(1.02).tick, 0)
        b.accept(message(2**32-1), 1.02)
        b.accept(message(1), 1.03)
        with self.assertRaisesRegex(ValueError, 'clock_reset'):
            b.latest(1.03)

    def test_bad_feedback_rejected(self):
        for change in (lambda m: setattr(m, 'mode_pr', 1),
                       lambda m: setattr(m.motor_state[0], 'q', float('nan')),
                       lambda m: setattr(m.imu_state, 'quaternion', [0.]*4)):
            m = message()
            change(m)
            with self.assertRaises(ValueError):
                decode_state(m, 1.)

    def test_tilt_and_motor_fault(self):
        m = message()
        m.imu_state.quaternion = [0., 1., 0., 0.]
        with self.assertRaisesRegex(ValueError, 'body_tilt'):
            check_posture(decode_state(m, 1.))
        m = message()
        m.motor_state[0].motorstate = 1
        with self.assertRaisesRegex(ValueError, 'motor_status'):
            check_posture(decode_state(m, 1.))
