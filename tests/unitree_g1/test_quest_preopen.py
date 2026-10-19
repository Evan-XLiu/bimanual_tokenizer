import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/unitree_g1'))
from quest_bridge import open_grippers_before_teleoperation, reconcile_external_preopen_cleanup
from collect_episodes import motion_cleanup_confirmed
from quest_bridge import PreparationTiming, wait_for_preloaded_ik


class PreopenTests(unittest.TestCase):
    def test_preparation_timing_preserves_failed_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            now = [10.]
            timing = PreparationTiming(output, lambda: now[0])
            with timing.stage('camera'):
                now[0] += 2.
            with self.assertRaisesRegex(ValueError, 'failure'):
                with timing.stage('preflight'):
                    now[0] += 3.
                    raise ValueError('failure')
            saved = json.loads((output/'preparation_timing.json').read_text())
            self.assertFalse(saved['ready'])
            self.assertEqual(saved['elapsed_s'], 5.)
            self.assertEqual([s['status'] for s in saved['stages']], ['complete', 'failed'])

    def test_preloaded_ik_still_requires_live_matching_success(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            stop = threading.Event()
            process = SimpleNamespace(is_alive=lambda: True)
            marker = output/'ik_ready.json'
            marker.write_text(json.dumps(dict(nonce='session', time=2.)))
            with patch('quest_bridge.time.monotonic', return_value=3.):
                self.assertEqual(wait_for_preloaded_ik(output, 'session', process, stop, 10.,
                                                     lambda: None)['time'], 2.)
                for cause in ('dead', 'stopped', 'timeout', 'error', 'nonce'):
                    with self.subTest(cause=cause):
                        stop.clear()
                        process.is_alive = lambda: cause != 'dead'
                        if cause == 'stopped': stop.set()
                        error = output/'worker_error.json'
                        if cause == 'error': error.write_text('{}')
                        with self.assertRaises((RuntimeError, ValueError)):
                            wait_for_preloaded_ik(output, 'wrong' if cause == 'nonce' else 'session',
                                                 process, stop, 1. if cause == 'timeout' else 10., lambda: None)
                        if error.exists(): error.unlink()

    def test_preflight_failure_preserves_preopen_release_without_rearming(self):
        released = dict(command_frames=386, arm_sdk_release_submitted=True,
                        external_gripper_zero_submitted=[True, True])
        cases = [
            (released, False, 0, True),
            (released, True, 0, False),  # Partial writer initialization is uncertain.
            (released, False, 1, False),
            ({**released, 'external_gripper_zero_submitted': [True, False]}, False, 0, False),
            ({**released, 'unload_error': 'failed'}, False, 0, False),
            ({}, False, 0, False),
            (dict(control_publish_not_started=True, command_frames=0), False, 0, True),
            (dict(control_publish_not_started=True, command_frames=1), False, 0, False),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output/'preopen').mkdir()
            for preopen, attempted, frames, expected in cases:
                with self.subTest(preopen=preopen, attempted=attempted, frames=frames):
                    (output/'preopen/summary.json').write_text(json.dumps(preopen))
                    result = dict(error='ValueError: Legs/waist not stationary', command_frames=frames)
                    reconcile_external_preopen_cleanup(result, output, True, attempted)
                    self.assertEqual(motion_cleanup_confirmed(result), expected)
                    self.assertEqual(result['error'], 'ValueError: Legs/waist not stationary')
                    self.assertNotIn('physical_handover_verified', result)

    def run_preopen(self, result, code=0, stopped=False):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            (output/'preopen').mkdir()
            (output/'preopen/summary.json').write_text(json.dumps(result))
            process=SimpleNamespace(returncode=code,poll=lambda:code)
            stop=threading.Event()
            if stopped:stop.set()
            args=SimpleNamespace(interface='unused',scene_confirmation='current support ready')
            with patch('quest_bridge.subprocess.Popen',return_value=process) as launch:
                open_grippers_before_teleoperation(args,output,stop)
                self.assertIn('--open-target',launch.call_args.args[0])

    def test_requires_successful_complete_opening(self):
        self.run_preopen(dict(error=None,opening_complete=True))
        for result,code in ((dict(error=None,opening_complete=False),0),
                            (dict(error='tracking_error',opening_complete=True),0),
                            (dict(error=None,opening_complete=True),1)):
            with self.assertRaises(RuntimeError):self.run_preopen(result,code)

    def test_stop_never_proceeds_to_quest(self):
        with self.assertRaisesRegex(RuntimeError,'Stopped'):
            self.run_preopen(dict(error=None,opening_complete=True),stopped=True)
