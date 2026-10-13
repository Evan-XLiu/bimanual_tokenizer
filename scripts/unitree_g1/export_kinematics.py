"""Offline pelvis-frame FK using the pinned G1 URDF chain; no robot imports."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

MODEL_PATH = Path(__file__).resolve().parents[2]/'artifacts/unitree_g1/export_kinematics.json'
MODEL = json.loads(MODEL_PATH.read_text())
MODEL_SHA256 = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()


def quaternion(rotation):
    """Unit quaternion in wxyz order, with deterministic hemisphere."""
    q = rotation.as_quat()  # xyzw
    q = q[[3,0,1,2]]
    pivot = next((x for x in q if abs(x) > 1e-12), 1.)
    return q if pivot >= 0 else -q


def forward(full_body_q):
    q = np.asarray(full_body_q, dtype=float)
    if q.shape != (35,) or not np.isfinite(q).all():
        raise ValueError('FK requires 35 finite motor positions')
    poses = []
    for side in ('left','right'):
        position = np.zeros(3)
        rotation = Rotation.identity()
        for joint in MODEL['chains'][side]:
            position += rotation.apply(joint['xyz'])
            rotation = rotation * Rotation.from_euler('xyz',joint['rpy'])
            rotation = rotation * Rotation.from_rotvec(np.asarray(joint['axis'])*q[joint['motor_index']])
        position += rotation.apply(MODEL['tool_offset_xyz_m'])
        poses.append((position,rotation))
    return poses


def eef_features(full_body_q, arm_action, gripper_action):
    """Action FK holds measured waist fixed; delta is target vs current state."""
    state_q = np.asarray(full_body_q,dtype=float)
    target_q = state_q.copy()
    target_q[15:29] = arm_action
    state, target = forward(state_q), forward(target_q)
    observed, absolute, delta = [], [], []
    for i,motor in enumerate((31,33)):
        p,r = state[i]
        tp,tr = target[i]
        observed.extend([*p,*quaternion(r),state_q[motor]])
        absolute.extend([*tp,*quaternion(tr),gripper_action[i]])
        delta.extend([*(tp-p),*quaternion(tr*r.inv()),gripper_action[i]])
    return {key:np.asarray(value,dtype=np.float32) for key,value in (
        ('observation.state_eef_absolute',observed),
        ('action_eef_absolute',absolute),('action_eef_delta',delta))}


def conventions():
    return dict(model_sha256=MODEL_SHA256,base_frame=MODEL['base_frame'],
        tool_definition=MODEL['tool_definition'],tool_offset_xyz_m=MODEL['tool_offset_xyz_m'],
        pose_order='left then right; x,y,z,qw,qx,qy,qz,gripper',
        position_units='m',quaternion='unit wxyz; first nonzero component positive',
        observation='FK of pre-command measured arm and waist positions',
        action='FK of submitted arm targets with the same measured waist positions',
        delta_translation='target minus measured position, expressed in pelvis frame',
        delta_rotation='q_target * inverse(q_measured), Hamilton product, pelvis frame',
        delta_gripper='absolute submitted target in native motor position units',
        tcp_calibrated=False)
