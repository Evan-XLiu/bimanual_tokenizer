from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/unitree_g1'))
from g1_sim_model import build_model, mujoco
from lower_body_policy import BODY_NAMES, LowerBodyPolicy
from standing_reference import StandingReference
from stance_odometry import StanceOdometry
from upper_body_balance import UpperBodyBalance


class SimModelTests(unittest.TestCase):
    def test_implicit_actuator_matches_explicit_pd_including_limits(self):
        models = [build_model(servo=s)[0] for s in ('implicit', 'explicit')]
        rng = np.random.default_rng(42)
        _, c, _ = build_model()
        kp, kd = np.r_[c['kps'], c['arm_waist_kps']], np.r_[c['kds'], c['arm_waist_kds']]
        q, dq, target = rng.normal(0, .05, (3, 29))
        forces = []
        for index, m in enumerate(models):
            d = mujoco.MjData(m)
            j = [m.joint(n).id for n in BODY_NAMES]
            d.qpos[m.jnt_qposadr[j]], d.qvel[m.jnt_dofadr[j]] = q, dq
            d.ctrl[:] = target if index == 0 else kp*(target-q)-kd*dq
            mujoco.mj_forward(m, d)
            forces.append(d.qfrc_actuator.copy())
        np.testing.assert_allclose(*forces, atol=1e-12)

    def test_dex1_mass_replaces_rubber_hand_without_double_counting(self):
        base, _, _ = build_model()
        dex, _, meta = build_model(dex1=True)
        self.assertEqual((dex.nu, dex.nq, dex.nv), (33, 40, 39))
        self.assertAlmostEqual(meta['dex1']['mass_per_hand_kg'], .364946)
        # Upstream fused MJCF mass rounds the more precise URDF wrist inertia.
        self.assertAlmostEqual(dex.body_mass.sum()-base.body_mass.sum(), 2*(.364946-.170), delta=2e-6)
        loaded, _, _ = build_model(dex1=True, extra_mass=.2, payload_mass=.5)
        self.assertAlmostEqual(loaded.body_mass.sum()-dex.body_mass.sum(), .9)
        self.assertTrue(np.all(dex.body_inertia[1:] > 0))

    def test_warmup_does_not_change_policy_phase_or_history(self):
        p = LowerBodyPolicy()
        p.warmup()
        self.assertEqual(p.counter, 0)
        self.assertIsNone(p.last_observation)
        self.assertFalse(p.previous.any())

    def test_known_load_compensation_moves_com_towards_nominal(self):
        m, c, _ = build_model(dex1=True, extra_mass=.2, payload_mass=.5)
        balance = UpperBodyBalance(m, c)
        target = balance.target(balance.default, [1., 0., 0., 0.])
        self.assertGreater(target[1], 0.)  # oppose left-heavy load
        self.assertLess(target[2], 0.)    # oppose forward mass shift
        self.assertLessEqual(max(map(abs, target)), .2)
        d = balance.d
        d.qpos[balance.qa[12:15]] = 0.
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        before = np.linalg.norm(d.subtree_com[balance.pelvis, :2]-balance.reference)
        d.qpos[balance.qa[12:15]] = target
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        after = np.linalg.norm(d.subtree_com[balance.pelvis, :2]-balance.reference)
        self.assertLess(after, before*.1)

    def test_standing_frame_rotation_and_pose_timeout(self):
        ref = StandingReference([0, 0], np.pi/2, 0.)
        cmd = ref.update([.1, 0], [0, 0], np.pi/2, .02, .02)
        self.assertAlmostEqual(cmd[0], 0., places=7)
        self.assertGreater(cmd[1], 0.)
        with self.assertRaises(ValueError):
            ref.update([0, 0], [0, 0], 0., .02, .1)

    def test_odometry_uses_support_displacement_without_absolute_pose(self):
        p = np.array([[x, y, -.8] for x in (-.1, .1) for y in (-.15, -.1, .1, .15)])
        estimator = StanceOdometry()
        estimator.update(p, 0.)
        shifted = p.copy(); shifted[:, 0] -= .01
        position, _ = estimator.update(shifted, .02)
        np.testing.assert_allclose(position, [.01, 0.], atol=1e-12)
        with self.assertRaises(ValueError):
            estimator.update(shifted, .02)


if __name__ == '__main__':
    unittest.main()
