"""Simulation candidate: model-based waist compensation for known added loads.

Uses joint FK and configured mass, never simulator root translation. It cannot
identify an unknown grasped object's mass and is not a hardware controller.
"""
import numpy as np

from g1_sim_model import build_model, mujoco
from lower_body_policy import BODY_NAMES, finite


class UpperBodyBalance:
    def __init__(self, model, leg_config, elbow=.5):
        self.m = model
        self.d = mujoco.MjData(model)
        self.qa = np.array([model.jnt_qposadr[model.joint(n).id] for n in BODY_NAMES])
        self.default = np.array(leg_config['default_angles']+leg_config['arm_waist_target'])
        self.default[[18, 25]] = elbow
        nominal, _, _ = build_model(dex1=True)
        data = mujoco.MjData(nominal)
        addresses = [nominal.jnt_qposadr[nominal.joint(n).id] for n in BODY_NAMES]
        data.qpos[:3] = 0.
        data.qpos[addresses] = self.default
        mujoco.mj_kinematics(nominal, data)
        mujoco.mj_comPos(nominal, data)
        self.reference = data.subtree_com[nominal.body('pelvis').id, :2].copy()
        self.pelvis = model.body('pelvis').id

    def target(self, body_q, quaternion):
        q = finite(body_q, 29)
        self.d.qpos[:] = self.m.qpos0
        self.d.qpos[:3] = 0.
        self.d.qpos[3:7] = [1., 0., 0., 0.]
        self.d.qpos[self.qa] = self.default
        self.d.qpos[self.qa[15:29]] = q[15:29]
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        com = self.d.subtree_com[self.pelvis, :2].copy()
        jacobian = []
        for index in (13, 14):
            self.d.qpos[self.qa[index]] = .001
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            jacobian.append((self.d.subtree_com[self.pelvis, :2]-com)/.001)
            self.d.qpos[self.qa[index]] = 0.
        correction = np.linalg.solve(np.array(jacobian).T, self.reference-com)
        # Small bounded IMU trim supplements the static CoM approximation.
        # This is a new candidate controller, not an upstream policy feature.
        w, x, y, z = finite(quaternion, 4)
        roll = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
        pitch = np.arcsin(np.clip(2*(w*y-z*x), -1., 1.))
        correction += np.array([.25*roll, -.25*pitch])
        return tuple(np.r_[0., np.clip(correction, -.2, .2)])
