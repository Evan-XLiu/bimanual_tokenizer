import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('televuer_input', Path(__file__).resolve().parents[2] / 'scripts/unitree_g1/televuer_input.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


class InputTests(unittest.TestCase):
    def test_invalid_pose_and_buttons(self):
        value = dict(left=IDENTITY, right=IDENTITY, leftState={'triggerValue': 0., 'squeezeValue': 0.},
                     rightState={'triggerValue': 0., 'squeezeValue': 0.})
        self.assertTrue(m.valid_event('CONTROLLER_MOVE', value))
        value['right'] = [0]*16
        self.assertFalse(m.valid_event('CONTROLLER_MOVE', value))
        value['right'] = IDENTITY
        value['rightState'] = {'triggerValue': float('nan')}
        self.assertFalse(m.valid_event('CONTROLLER_MOVE', value))
        value['rightState'] = {'thumbstickValue': [0]}
        self.assertFalse(m.valid_event('CONTROLLER_MOVE', value))

    def test_missing_button_fields_are_not_neutral_inputs(self):
        import copy
        value=dict(left=IDENTITY,right=IDENTITY,
                   leftState={'triggerValue': .2, 'squeezeValue': 1.},
                   rightState={'triggerValue': .3, 'squeezeValue': 1.})
        for side in ('leftState', 'rightState'):
            for key in ('triggerValue','squeezeValue'):
                for invalid in (None, True, b'0', float('inf')):
                    raw=copy.deepcopy(value)
                    raw[side][key]=invalid
                    self.assertFalse(m.valid_event('CONTROLLER_MOVE',raw))
                raw=copy.deepcopy(value)
                del raw[side][key]
                self.assertFalse(m.valid_event('CONTROLLER_MOVE',raw))

    def test_serialization_marks_binary_nonfinite_and_unknown_invalid(self):
        import json
        import numpy as np
        for raw in (b'bad', {'extra': b'bad'}, {'x': float('nan')}, object()):
            record=dict(kind='CONTROLLER_MOVE',host_receive_monotonic_s=1.,
                        session='test',valid=True,raw=raw,official_head_yaw={'x':np.zeros(2)})
            result=json.loads(m.serialize_event(record))
            self.assertFalse(result['valid'])
            self.assertIn('serialization',result['input_error'])
            self.assertNotIn('official_head_yaw',result)
            self.assertEqual(result['host_receive_monotonic_s'],1.)
        result=json.loads(m.serialize_event(dict(valid=True,raw=np.eye(4))))
        self.assertTrue(result['valid'])
        self.assertEqual(result['raw'],np.eye(4).tolist())

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

    def test_gripper_hud_freshness_countdown_and_image_serialization(self):
        import json
        import tempfile
        from vuer.schemas import ImageBackground
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'status.json'
            self.assertIn('UNAVAILABLE',m.gripper_status_lines(path,1)[0])
            status=dict(time=1.,states=['POSITIONED','OPEN'],phase='ACTIVE',
                        hardware_motion_enabled=False,remaining_s=8.,fault=None)
            path.write_text(json.dumps(status))
            lines=m.gripper_status_lines(path,1.1)
            self.assertIn('SHADOW | L: POSITIONED | R: OPEN',lines[0])
            self.assertIn('PLACE OBJECT DOWN',lines[1])
            self.assertIn('STALE',m.gripper_status_lines(path,2)[0])
            self.assertIn('STALE',m.gripper_status_lines(path,0)[0])
            canvas=m.gripper_hud_image(path,1.1)
            self.assertEqual(canvas.getpixel((512,384))[3],0)
            element=ImageBackground(canvas,format='png',key='gripper-status').serialize()
            self.assertEqual(element['key'],'gripper-status')
            status.update(phase='RELEASED',remaining_s=0)
            path.write_text(json.dumps(status))
            self.assertIn('no holding guarantee',m.gripper_status_lines(path,1.1)[1])
            path.write_text('{broken')
            self.assertIn('UNAVAILABLE',m.gripper_status_lines(path,1.1)[0])


if __name__ == '__main__':
    unittest.main()

class OfficialCallbackTests(unittest.TestCase):
    def test_gripper_hud_runs_with_controller_stream_without_robot_imports(self):
        import asyncio
        import multiprocessing as mp
        import queue
        import sys
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp,patch('multiprocessing.Process.start'),patch.object(sys,'meta_path',sys.meta_path.copy()):
            args=SimpleNamespace(output=Path(tmp),cert=Path(tmp)/'cert',key=Path(tmp)/'key',
                                 timeout=1,gripper_status=Path(tmp)/'missing.json')
            tv=m.create_tv(args,queue.Queue(),mp.Value('L',0))
            keys=[]
            async def run():
                ready=asyncio.Event()
                def upsert(element,**kwargs):
                    keys.append(element.serialize()['key'])
                    if 'gripper-status' in keys and 'motionControllers' in keys:ready.set()
                task=asyncio.create_task(tv.main_pass_through(SimpleNamespace(upsert=upsert)))
                try:await asyncio.wait_for(ready.wait(),2.)
                finally:
                    task.cancel()
                    await asyncio.gather(task,return_exceptions=True)
            asyncio.run(run())
            self.assertIn('motionControllers',keys)
            self.assertIn('gripper-status',keys)

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
                leftState={'triggerValue': 0.25, 'squeezeValue': 1.},
                rightState={'triggerValue': 0.75, 'squeezeValue': 1.}))
            asyncio.run(tv.on_cam_move(head, session))
            asyncio.run(tv.on_controller_move(controllers, session))
            import json
            self.assertTrue(json.loads(events.get_nowait())['valid'])
            result = json.loads(events.get_nowait())
            self.assertTrue(result['valid'])
            self.assertEqual(result['official_head_yaw']['left_ctrl_triggerValue'], 7.5)
            self.assertEqual(result['official_head_yaw']['right_ctrl_triggerValue'], 2.5)
            # Actual malformed startup event and binary extension must not kill
            # the callback; a later complete event remains usable.
            for malformed in ({}, b'bad', dict(controllers.value, extra=b'bad')):
                asyncio.run(tv.on_controller_move(SimpleNamespace(value=malformed), session))
                self.assertFalse(json.loads(events.get_nowait())['valid'])
            asyncio.run(tv.on_controller_move(controllers, session))
            self.assertTrue(json.loads(events.get_nowait())['valid'])
            from televuer import TeleVuer
            from unittest.mock import AsyncMock
            with patch.object(TeleVuer, 'on_controller_move', new=AsyncMock(side_effect=ValueError('bad upstream input'))):
                asyncio.run(tv.on_controller_move(controllers, session))
                failed=json.loads(events.get_nowait())
                self.assertFalse(failed['valid'])
                self.assertIn('upstream',failed['input_error'])
            asyncio.run(tv.on_controller_move(controllers, session))
            self.assertTrue(json.loads(events.get_nowait())['valid'])
            controllers.value['right'] = [0]*16
            asyncio.run(tv.on_controller_move(controllers, session))
            result = json.loads(events.get_nowait())
            self.assertFalse(result['valid'])
            self.assertNotIn('official_head_yaw', result)
