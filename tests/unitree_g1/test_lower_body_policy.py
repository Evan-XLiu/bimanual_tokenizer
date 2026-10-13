import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/unitree_g1'))
from lower_body_policy import LowerBodyPolicy, gravity, load_candidate


class LowerBodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.config, _ = load_candidate()
        spec = importlib.util.spec_from_file_location('upstream_rotation', cls.root/'deploy/deploy_real/common/rotation_helper.py')
        cls.rotation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.rotation)

    def test_gravity_matches_upstream_for_random_orientations(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            quat = rng.normal(size=4)
            quat /= np.linalg.norm(quat)
            np.testing.assert_allclose(gravity(quat), self.rotation.get_gravity_orientation(quat), atol=4e-7)
        np.testing.assert_array_equal(gravity([1, 0, 0, 0]), [0, 0, -1])

    def test_observation_layout_phase_and_action_history(self):
        p = LowerBodyPolicy()
        q, dq, omega = np.arange(12)*.01, np.arange(12)*.02, np.array([.1, .2, .3])
        cmd = [.2, -.1, .3]
        obs = p.observation(q, dq, [1, 0, 0, 0], omega, cmd)
        expected = np.r_[omega*.25, [0, 0, -1], [.4, -.2, .075],
                         q-p.default, dq*.05, np.zeros(12),
                         np.sin(2*np.pi*.02/.8), np.cos(2*np.pi*.02/.8)]
        np.testing.assert_allclose(obs, expected, atol=1e-7)
        target = p.step(q, dq, [1, 0, 0, 0], omega, cmd)
        np.testing.assert_allclose(target, p.default+p.previous*.25)
        np.testing.assert_array_equal(p.observation(q, dq, [1, 0, 0, 0], omega)[33:45], p.previous)

    def test_invalid_inputs_do_not_advance_policy(self):
        p = LowerBodyPolicy()
        for key, value in [('q', [0]*11), ('dq', [float('nan')]*12),
                           ('quaternion', [0]*4), ('gyro', [float('inf')]*3),
                           ('command', [2, 0, 0])]:
            args = dict(q=p.default, dq=[0]*12, quaternion=[1, 0, 0, 0], gyro=[0]*3)
            args[key] = value
            with self.assertRaises(ValueError):
                p.step(**args)
            self.assertEqual(p.counter, 0)

    def test_invalid_network_output_is_rejected_without_history_change(self):
        p = LowerBodyPolicy()
        for shape in ((1, 11), (1, 12)):
            p.model = lambda obs: p.torch.full(shape, float('nan'))
            with self.assertRaises(ValueError):
                p.step(p.default, [0]*12, [1, 0, 0, 0], [0]*3)
            self.assertEqual(p.counter, 0)
            self.assertFalse(p.previous.any())


if __name__ == '__main__':
    unittest.main()
