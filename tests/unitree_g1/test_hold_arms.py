import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/unitree_g1'))
from arm_execution import Config, Observation
from hold_arms import OwnershipLost, load_profile, run_hold


class FakeIO:
    def __init__(self):
        self.t = 0.
        self.commands = []
        self.foreign_at = float('inf')
        self.fail_at = float('inf')
        self.frozen = False

    def sleep(self, dt):
        self.t += dt

    def check(self):
        if self.t >= self.foreign_at:
            raise OwnershipLost('foreign command or mode loss')

    def observation(self):
        t = 0 if self.frozen else self.t
        return Observation((.4,)*14, t, round(t*1000)+1, 5)

    def write(self, command):
        if self.t >= self.fail_at:
            return False
        self.commands.append((self.t, command))
        return True


class HoldTests(unittest.TestCase):
    def setUp(self):
        self.io = FakeIO()
        self.config = Config((-2.,)*14, (2.,)*14, (5.,)*14, (1.,)*14,
                             (.2,)*14, (.5,)*14, (.3,)*14, .1, .05, .05, .03)
        self.profile = dict(period_s=.01, hold_s=.1, zero_tail_s=.05)

    def run_case(self, stop=lambda: False, log=lambda row: None):
        return run_hold(self.io, self.config, self.profile, stop, log,
                        lambda: self.io.t, self.io.sleep)

    def test_finite_hold_and_passive_tail(self):
        result = self.run_case()
        self.assertIsNone(result['error'])
        self.assertTrue(result['unload_submitted'])
        self.assertFalse(result['physical_unload_verified'])
        self.assertTrue(all(c.q == (.4,)*14 for _,c in self.io.commands))
        self.assertEqual(self.io.commands[0][1].kp, (0.,)*14)
        self.assertGreaterEqual(sum(c.phase == 'RELEASED' for _,c in self.io.commands), 5)

    def test_signal_during_engagement_or_hold(self):
        for when in (.03, .15):
            self.io = FakeIO()
            result = self.run_case(stop=lambda: self.io.t >= when)
            self.assertTrue(result['unload_submitted'])
            self.assertLess(self.io.t, .3)
            self.assertEqual(self.io.commands[-1][1].kp, (0.,)*14)

    def test_stop_before_engagement_has_no_writes(self):
        result = self.run_case(stop=lambda: True)
        self.assertEqual(self.io.commands, [])
        self.assertFalse(result['unload_submitted'])

    def test_ownership_loss_no_competing_cleanup_writes(self):
        self.io.foreign_at = .15
        result = self.run_case()
        self.assertFalse(result['unload_submitted'])
        self.assertIn('OwnershipLost', result['error'])
        self.assertTrue(all(t < .15 for t,c in self.io.commands))

    def test_transport_failure_never_claims_unload(self):
        self.io.fail_at = .15
        result = self.run_case()
        self.assertFalse(result['unload_submitted'])
        self.assertIn('unload_error', result)

    def test_feedback_freeze_unloads_and_reports_fault(self):
        self.io.frozen = True
        result = self.run_case()
        self.assertTrue(result['unload_submitted'])
        self.assertIsNotNone(result['error'])

    def test_release_fault_evidence_survives_passive_tail(self):
        original = self.io.observation
        def observation():
            state = original()
            if self.io.t >= .25:
                return Observation((.8,)+(.4,)*13, state.received, state.tick, 5)
            return state
        self.io.observation = observation
        rows = []
        result = self.run_case(log=rows.append)
        self.assertEqual(result['error'], 'tracking_error')
        self.assertEqual(result['release_reason'], 'operator_release')
        self.assertEqual(result['first_fault']['phase'], 'RELEASING')
        self.assertTrue(result['unload_submitted'])
        self.assertEqual(rows[-1]['phase'], 'RELEASED')
        self.assertEqual(rows[-1]['first_fault'], result['first_fault'])
        fault_row = next(r for r in rows if r['first_fault'])
        self.assertEqual(fault_row['executor_feedback']['q'][0], .8)

    def test_logging_failure_still_completes_unload(self):
        def log(row):
            if self.io.t > .05:
                raise OSError('disk full')
        result = self.run_case(log=log)
        self.assertTrue(result['unload_submitted'])
        self.assertIn('disk full', result['error'])
        self.assertEqual(self.io.commands[-1][1].kp, (0.,)*14)

    def test_live_rejects_unverified_profile_before_dds(self):
        profile = dict(self.profile, arm_indices=list(range(15,29)),
                       max_initial_speed=.1, controller=self.config.__dict__, hardware_approved=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'profile.json'
            path.write_text(json.dumps(profile))
            load_profile(path, False)
            with self.assertRaisesRegex(ValueError, 'Missing hardware verification'):
                load_profile(path, True)

    def test_candidate_trial_requires_scene_and_rejects_parameter_changes(self):
        root = Path(__file__).resolve().parents[2]
        profile = json.loads((root/'artifacts/unitree_g1/hold_candidate_trial.json').read_text())
        # The real trial profile now contains operator readiness statements.
        # This rejection case must construct its own missing-readiness fixture.
        profile['trial_scene'] = {}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'candidate.json'
            path.write_text(json.dumps(profile))
            with self.assertRaisesRegex(ValueError, 'Missing on-site trial readiness'):
                load_profile(path, True, True)
            profile['trial_scene'] = {key: 'FAKE TEST EVIDENCE' for key in
                ('body_passive_support', 'arm_unload_support', 'independent_stop')}
            path.write_text(json.dumps(profile))
            actual, _ = load_profile(path, True, True)
            self.assertFalse(actual['hardware_approved'])
            profile['hold_s'] = 3
            path.write_text(json.dumps(profile))
            with self.assertRaisesRegex(ValueError, 'differs from reviewed proposal'):
                load_profile(path, True, True)


if __name__ == '__main__':
    unittest.main()
