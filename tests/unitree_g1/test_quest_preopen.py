import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/unitree_g1'))
from quest_bridge import open_grippers_before_teleoperation


class PreopenTests(unittest.TestCase):
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
