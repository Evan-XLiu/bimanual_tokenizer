import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('televuer_input', Path(__file__).resolve().parents[2] / 'scripts/unitree_g1/televuer_input.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


class InputTests(unittest.TestCase):
    def test_invalid_pose_and_buttons(self):
        value = dict(left=IDENTITY, right=IDENTITY, leftState={}, rightState={})
        self.assertTrue(m.valid_event('CONTROLLER_MOVE', value))
        value['right'] = [0]*16
        self.assertFalse(m.valid_event('CONTROLLER_MOVE', value))
        value['right'] = IDENTITY
        value['rightState'] = {'triggerValue': float('nan')}
        self.assertFalse(m.valid_event('CONTROLLER_MOVE', value))
        value['rightState'] = {'thumbstickValue': [0]}
        self.assertFalse(m.valid_event('CONTROLLER_MOVE', value))

    def test_timeout_invalid_and_new_connection(self):
        monitor = m.Monitor(1)
        def feed(kind, t, session='a', valid=True):
            monitor.accept(dict(kind=kind, host_receive_monotonic_s=t, session=session, valid=valid))
        self.assertEqual(monitor.status(1)['state'], 'waiting')
        feed('CAMERA_MOVE', 10)
        feed('CONTROLLER_MOVE', 10)
        self.assertTrue(monitor.status(10.5)['receive_fresh'])
        self.assertFalse(monitor.status(11.1)['receive_fresh'])
        feed('CONTROLLER_MOVE', 11.2)
        self.assertFalse(monitor.status(11.2)['receive_fresh'])
        feed('CAMERA_MOVE', 11.2)
        self.assertTrue(monitor.status(11.2)['receive_fresh'])
        feed('CONTROLLER_MOVE', 11.3, valid=False)
        self.assertFalse(monitor.status(11.3)['receive_fresh'])
        feed('CONTROLLER_MOVE', 11.4, session='b')
        self.assertFalse(monitor.status(11.4)['receive_fresh'])
        self.assertFalse(monitor.status(11.4)['motion_enabled'])

    def test_robot_import_guard(self):
        guard = m.NoRobotImports()
        for name in ('unitree_sdk2py', 'cyclonedds.core', 'teleop.robot_control', 'robot_control'):
            with self.assertRaises(ImportError):
                guard.find_spec(name)
        self.assertIsNone(guard.find_spec('televuer'))


if __name__ == '__main__':
    unittest.main()

class OfficialCallbackTests(unittest.TestCase):
    def test_official_callbacks_and_conversion_without_server(self):
        import asyncio
        import multiprocessing as mp
        import queue
        import sys
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp, patch('multiprocessing.Process.start'), patch.object(sys, 'meta_path', sys.meta_path.copy()):
            args = SimpleNamespace(output=Path(tmp), cert=Path(tmp)/'cert', key=Path(tmp)/'key', timeout=1)
            events = queue.Queue()
            tv = m.create_tv(args, events, mp.Value('L', 0))
            session = SimpleNamespace(CURRENT_WS_ID='test')
            head = SimpleNamespace(value={'camera': {'matrix': IDENTITY}})
            controllers = SimpleNamespace(value=dict(left=IDENTITY, right=IDENTITY,
                leftState={'triggerValue': 0.25}, rightState={'triggerValue': 0.75}))
            asyncio.run(tv.on_cam_move(head, session))
            asyncio.run(tv.on_controller_move(controllers, session))
            import json
            self.assertTrue(json.loads(events.get_nowait())['valid'])
            result = json.loads(events.get_nowait())
            self.assertTrue(result['valid'])
            self.assertEqual(result['official_head_yaw']['left_ctrl_triggerValue'], 7.5)
            self.assertEqual(result['official_head_yaw']['right_ctrl_triggerValue'], 2.5)
            controllers.value['right'] = [0]*16
            asyncio.run(tv.on_controller_move(controllers, session))
            result = json.loads(events.get_nowait())
            self.assertFalse(result['valid'])
            self.assertNotIn('official_head_yaw', result)
