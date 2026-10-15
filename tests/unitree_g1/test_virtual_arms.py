import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/unitree_g1'))
import virtual_arms as m


def event(t, grip=1., x=0., session='a'):
    pose = np.eye(4)
    pose[0, 3] = x
    return {'kind': 'CONTROLLER_MOVE', 'session': session,
            'host_receive_monotonic_s': t, 'valid': True,
            'raw': {s: pose.flatten(order='F').tolist() for s in ('left', 'right')} |
                   {s+'State': {'squeezeValue': grip, 'triggerValue': 0.} for s in ('left', 'right')}}


class ClutchTests(unittest.TestCase):
    def setUp(self):
        self.c = m.RelativeClutch()
        self.robot = [np.eye(4), np.eye(4)]

    def feed(self, t, **kwargs):
        return self.c.accept(event(t, **kwargs), t, self.robot)

    def engage(self):
        self.feed(1, grip=0)
        self.assertIsNone(self.feed(1.03))

    def test_release_engage_mapping_hold_and_rebase(self):
        self.assertIsNone(self.feed(0.9))
        self.engage()
        result = self.feed(1.06, x=0.02)
        np.testing.assert_allclose(result[0][:3, 3], [0, -0.02, 0])
        self.assertIsNone(self.feed(1.09, grip=0, x=0.4))
        self.assertIsNone(self.feed(1.12, x=0.4))
        result = self.feed(1.15, x=0.41)
        np.testing.assert_allclose(result[1][:3, 3], [0, -0.01, 0], atol=1e-8)

    def test_timeout_and_reconnect_require_release(self):
        self.engage()
        self.c.tick(2)
        self.assertEqual(self.c.state, 'stale')
        self.assertIsNone(self.feed(2.01))
        self.feed(2.04, grip=0)
        self.feed(2.07)
        self.assertIsNone(self.feed(2.10, session='b'))
        self.assertNotEqual(self.c.state, 'following')

    def test_half_grip_engages_and_either_side_below_half_releases(self):
        self.feed(.9,grip=.5)
        self.assertNotEqual(self.c.state,'following')
        self.feed(1.,grip=0.)
        self.feed(1.01,grip=.5)
        self.assertEqual(self.c.state,'following')
        self.assertIsNotNone(self.feed(1.02,grip=.619,x=.01))
        for side in ('left','right'):
            e=event(1.03,grip=1.)
            e['raw'][side+'State']['squeezeValue']=.499
            self.assertIsNone(self.c.accept(e,1.03,self.robot))
            self.assertEqual(self.c.state,'hold')
            self.c=m.RelativeClutch()
            self.feed(1.,grip=0.)
            self.feed(1.01,grip=.5)

    def test_jump_invalid_and_old_backlog(self):
        self.engage()
        self.assertIsNone(self.feed(1.06, x=0.5))
        self.assertEqual(self.c.state, 'pose_jump')
        self.assertIsNone(self.feed(1.09))
        old = event(1.1, grip=0)
        self.assertIsNone(self.c.accept(old, 2, self.robot))
        self.assertEqual(self.c.state, 'stale')
        bad = event(2.1, grip=0)
        bad['raw']['right'] = [0]*16
        self.c.accept(bad, 2.1, self.robot)
        self.assertEqual(self.c.state, 'invalid_input')

    def test_rotation_mapping(self):
        self.engage()
        e = event(1.06)
        a = 0.1
        rotation = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        pose = np.eye(4)
        pose[:3, :3] = rotation
        e['raw']['left'] = pose.flatten(order='F').tolist()
        target = self.c.accept(e, 1.06, self.robot)
        np.testing.assert_allclose(target[0][:3, :3], m.XR_TO_ROBOT @ rotation @ m.XR_TO_ROBOT.T, atol=1e-8)

    def test_relative_rotation_nonidentity_references_and_independent_sides(self):
        from scipy.spatial.transform import Rotation
        initial = [Rotation.from_euler('xyz', [.2, -.3, .4]).as_matrix(),
                   Rotation.from_euler('xyz', [-.1, .4, -.2]).as_matrix()]
        for i, robot in enumerate(self.robot):
            robot[:3, :3] = Rotation.from_euler('xyz', [.3, .2, .1*(i+1)]).as_matrix()
        def rotated_event(t, rotations, grip=1.):
            e = event(t, grip=grip)
            for side, rotation in zip(('left', 'right'), rotations):
                pose = np.eye(4)
                pose[:3, :3] = rotation
                e['raw'][side] = pose.flatten(order='F').tolist()
            return e
        self.c.accept(rotated_event(1., initial, 0.), 1., self.robot)
        self.assertIsNone(self.c.accept(rotated_event(1.03, initial), 1.03, self.robot))
        delta = Rotation.from_rotvec([.1, -.15, .2]).as_matrix()
        current = [delta @ initial[0], initial[1]]
        targets = self.c.accept(rotated_event(1.06, current), 1.06, self.robot)
        np.testing.assert_allclose(targets[0][:3, :3],
            m.XR_TO_ROBOT @ delta @ m.XR_TO_ROBOT.T @ self.robot[0][:3, :3], atol=1e-14)
        np.testing.assert_allclose(targets[1], self.robot[1], atol=1e-14)
        # Release and re-engage at a rotated controller pose resets rotation zero.
        self.c.accept(rotated_event(1.09, current, 0.), 1.09, self.robot)
        self.assertIsNone(self.c.accept(rotated_event(1.12, current), 1.12, self.robot))
        targets = self.c.accept(rotated_event(1.15, current), 1.15, self.robot)
        np.testing.assert_allclose(targets, self.robot, atol=1e-14)

    def test_partial_line(self):
        with tempfile.TemporaryFile(mode='w+') as stream:
            stream.write('{"x":1}\n{"x":')
            stream.seek(0)
            self.assertEqual(m.read_available(stream), [{'x': 1}])
            position = stream.tell()
            stream.seek(0, 2)
            stream.write('2}\n')
            stream.seek(position)
            self.assertEqual(m.read_available(stream), [{'x': 2}])

    def test_guard(self):
        guard = m.NoHardwareImports()
        for name in ('unitree_sdk2py', 'cyclonedds.core', 'teleop.robot_control.robot_arm',
                     'teleop.robot_control.robot_hand_unitree', 'teleop.teleop_hand_and_arm'):
            with self.assertRaises(ImportError):
                guard.find_spec(name)
        self.assertIsNone(guard.find_spec('teleop.robot_control.robot_arm_ik'))


class RealIKTests(unittest.TestCase):
    def test_continuous_two_arm_trajectory(self):
        # SDK encoding tests may already have imported hardware modules in the
        # discovery process. Check the virtual-only boundary in a fresh process.
        result = subprocess.run(
            [sys.executable, '-c',
             'from test_virtual_arms import RealIKTests; '
             'RealIKTests()._check_continuous_two_arm_trajectory()'],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def _check_continuous_two_arm_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            robot = m.VirtualRobot(Path(tmp), False)
            start = robot.q.copy()
            targets = robot.forward()
            for i in range(40):
                shifted = [p.copy() for p in targets]
                shifted[0][0, 3] += 0.03*(i+1)/40
                shifted[1][2, 3] += 0.03*(i+1)/40
                before = robot.q.copy()
                errors = robot.solve(shifted, 1/30)
                self.assertLess(max(errors), 0.03)
                self.assertLessEqual(float(np.max(np.abs(robot.q-before))), 0.05+1e-9)
            self.assertGreater(np.linalg.norm(robot.q[:7]-start[:7]), 0.02)
            self.assertGreater(np.linalg.norm(robot.q[7:]-start[7:]), 0.02)
            self.assertFalse(any(name.startswith(('unitree_sdk2py', 'cyclonedds')) for name in sys.modules))


if __name__ == '__main__':
    unittest.main()
