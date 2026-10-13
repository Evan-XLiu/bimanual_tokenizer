"""Offline checks of the exported pose and delta conventions."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/unitree_g1'))
from export_kinematics import eef_features, forward, quaternion


class ExportKinematicsTests(unittest.TestCase):
    def test_identity_delta_preserves_absolute_gripper_target(self):
        q = np.linspace(-.2,.2,35)
        fields = eef_features(q,q[15:29],[-2.,-3.])
        delta = fields['action_eef_delta'].reshape(2,8)
        np.testing.assert_allclose(delta[:,:3],0,atol=1e-7)
        np.testing.assert_allclose(delta[:,3:7],[[1,0,0,0]]*2,atol=1e-7)
        np.testing.assert_allclose(delta[:,7],[-2,-3])
        np.testing.assert_allclose(fields['observation.state_eef_absolute'][[7,15]],q[[31,33]])

    def test_delta_reconstructs_target_in_base_frame(self):
        q = np.linspace(-.3,.3,35)
        fields = eef_features(q,np.linspace(.2,-.4,14),[-1.,-2.])
        state = fields['observation.state_eef_absolute'].reshape(2,8)
        target = fields['action_eef_absolute'].reshape(2,8)
        delta = fields['action_eef_delta'].reshape(2,8)
        for s,t,d in zip(state,target,delta):
            np.testing.assert_allclose(s[:3]+d[:3],t[:3],atol=1e-7)
            def rot(x):return Rotation.from_quat(x[[4,5,6,3]])
            np.testing.assert_allclose((rot(d)*rot(s)).as_matrix(),rot(t).as_matrix(),atol=2e-7)
            self.assertAlmostEqual(np.linalg.norm(d[3:7]),1.,places=6)

    def test_measured_waist_rotates_both_poses(self):
        q = np.zeros(35)
        reference = forward(q)
        q[12] = .7
        rotated = forward(q)
        yaw = Rotation.from_euler('z',.7)
        for (p,r),(tp,tr) in zip(reference,rotated):
            np.testing.assert_allclose(tp,yaw.apply(p),atol=1e-12)
            np.testing.assert_allclose(tr.as_matrix(),(yaw*r).as_matrix(),atol=1e-12)

    def test_pi_rotation_and_nonfinite_input(self):
        q = quaternion(Rotation.from_euler('x',np.pi))
        np.testing.assert_allclose(q,[0,1,0,0],atol=1e-12)
        with self.assertRaises(ValueError):forward([float('nan')]*35)


if __name__ == '__main__':unittest.main()
