"""Pinned Unitree G1 policy adapter. Pure inference; no DDS or hardware startup.

Observation/action conventions follow the locked upstream deploy_real implementation.
Commands are physical [m/s, m/s, rad/s], NOT normalized joystick values.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / 'artifacts/unitree_g1/lower_body.lock.json'
LEG_NAMES = tuple(f'{side}_{joint}_joint' for side in ('left', 'right')
                  for joint in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll'))
BODY_NAMES = LEG_NAMES + ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint') + tuple(
    f'{side}_{joint}_joint' for side in ('left', 'right') for joint in
    ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw'))


def finite(value, size):
    a = np.asarray(value, dtype=np.float32)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f'Expected {size} finite values')
    return a


def gravity(quaternion):
    w, x, y, z = finite(quaternion, 4)
    if abs(float(np.linalg.norm([w, x, y, z])) - 1) > .001:
        raise ValueError('Expected unit pelvis quaternion in wxyz order')
    return np.array([2*(-z*x+w*y), -2*(z*y+w*x), 1-2*(w*w+z*z)], dtype=np.float32)


def load_candidate():
    lock = json.loads(LOCK.read_text())
    root = ROOT / lock['source_root']
    for name, expected in lock['sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Pinned source hash mismatch: {name}')
    config = yaml.safe_load((root/'deploy/deploy_real/configs/g1.yaml').read_text())
    if (config['num_obs'], config['num_actions'], config['imu_type'], config['control_dt']) != (47, 12, 'pelvis', .02):
        raise ValueError('Unsupported policy convention')
    if config['leg_joint2motor_idx'] != list(range(12)) or config['arm_waist_joint2motor_idx'] != list(range(12, 29)):
        raise ValueError('Unexpected motor mapping')
    return root, config, lock


class LowerBodyPolicy:
    def __init__(self):
        import torch
        root, self.config, self.lock = load_candidate()
        self.torch = torch
        torch.set_num_threads(1)
        self.model = torch.jit.load(str(root/'deploy/pre_train/g1/motion.pt'), map_location='cpu').eval()
        self.previous = np.zeros(12, dtype=np.float32)
        self.counter = 0
        self.default = finite(self.config['default_angles'], 12)
        self.last_observation = None

    def warmup(self, iterations=5):
        """Compile/prime TorchScript without changing phase or previous action."""
        obs = self.observation(self.default, np.zeros(12), [1, 0, 0, 0], np.zeros(3))
        with self.torch.inference_mode():
            for _ in range(iterations):
                output = self.model(self.torch.from_numpy(obs).unsqueeze(0)).cpu().numpy()
                if output.shape != (1, 12):
                    raise ValueError('Unexpected policy output shape')
                finite(output[0], 12)

    def observation(self, q, dq, quaternion, gyro, command=(0., 0., 0.)):
        c = self.config
        cmd = finite(command, 3)
        if np.any(np.abs(cmd) > np.asarray(c['max_cmd'])):
            raise ValueError('Velocity command outside upstream bounds')
        phase = ((self.counter+1)*c['control_dt'] % .8)/.8
        return np.concatenate((finite(gyro, 3)*c['ang_vel_scale'], gravity(quaternion),
                               cmd*np.asarray(c['cmd_scale']),
                               (finite(q, 12)-self.default)*c['dof_pos_scale'],
                               finite(dq, 12)*c['dof_vel_scale'], self.previous,
                               [np.sin(2*np.pi*phase), np.cos(2*np.pi*phase)])).astype(np.float32)

    def step(self, q, dq, quaternion, gyro, command=(0., 0., 0.)):
        obs = self.observation(q, dq, quaternion, gyro, command)
        with self.torch.inference_mode():
            output = self.model(self.torch.from_numpy(obs).unsqueeze(0)).cpu().numpy()
        if output.shape != (1, 12):
            raise ValueError('Unexpected policy output shape')
        action = finite(output[0], 12)
        target = finite(self.default + action*self.config['action_scale'], 12)
        self.previous = action.copy()
        self.last_observation = obs.copy()
        self.counter += 1
        return target
