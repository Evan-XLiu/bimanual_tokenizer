"""Offline recorder tests: no Quest, robot, SDK or training dependencies."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "quest_record", Path(__file__).resolve().parents[2] / "scripts/unitree_g1/quest_record.py"
)
quest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quest)


def frame(seq=0, timestamp=100):
    pose = {"position": [0., 1., -0.5], "orientation_xyzw": [0., 0., 0., 1.], "emulated_position": False}
    ctrl = {"grip": pose, "target_ray": pose, "buttons": [{"value": 0.3, "pressed": False, "touched": True}], "axes": [0., 0.]}
    return copy.deepcopy({"type": "frame", "seq": seq, "client_time_ms": timestamp,
                          "xr_frame_time_ms": timestamp, "reference_epoch": 0,
                          "visibility": "visible", "head": pose, "left": ctrl, "right": ctrl})


class QuestRecorderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name) / "session"
        self.recorder = quest.Recorder(self.output)
        self.episode = self.recorder.start({"reference_space": "local-floor", "performance_time_origin_ms": 1}, now=0)

    def tearDown(self):
        self.recorder.finish("test_end", now=1)
        self.temp.cleanup()

    def rows(self):
        return [json.loads(line) for line in (self.output / self.episode / "raw.jsonl").read_text().splitlines()]

    def test_preserves_clocks_and_input_not_robot_actions(self):
        self.recorder.batch(self.episode, [frame()], now=.1)
        row = self.rows()[0]
        self.assertEqual(row["host_receive_monotonic_s"], .1)
        self.assertEqual(row["input"]["xr_frame_time_ms"], 100)
        self.assertNotIn("action", row)
        summary = self.recorder.finish("client_end", now=.2)
        self.assertEqual(summary["valid_both_frames"], 1)

    def test_invalid_tracking_not_filled_with_previous_pose(self):
        lost, emulated, hidden = frame(1, 115), frame(2, 130), frame(3, 145)
        lost["left"] = None
        emulated["right"]["grip"]["emulated_position"] = True
        hidden["visibility"] = "hidden"
        self.recorder.batch(self.episode, [frame(), lost, emulated, hidden], now=.1)
        self.assertIsNone(self.rows()[1]["input"]["left"])
        self.assertEqual(self.recorder.finish("client_end", now=.2)["valid_both_frames"], 1)

    def test_bad_batch_is_atomic(self):
        for bad in [frame(2, 115), frame(1, 99), frame(1, 100)]:
            with self.assertRaises(ValueError):
                self.recorder.batch(self.episode, [frame(), bad], now=.1)
            self.assertEqual(self.rows(), [])
        bad = frame(1, 115)
        bad["left"]["grip"]["position"][0] = float("nan")
        with self.assertRaises(ValueError):
            self.recorder.batch(self.episode, [frame(), bad], now=.1)
        self.assertEqual(self.rows(), [])

    def test_duplicate_rejected(self):
        self.recorder.batch(self.episode, [frame()], now=.1)
        with self.assertRaises(ValueError):
            self.recorder.batch(self.episode, [frame()], now=.2)
        self.assertEqual(len(self.rows()), 1)

    def test_timeout_clears_latest_and_blocks_late_frames(self):
        self.recorder.batch(self.episode, [frame()], now=.1)
        status = self.recorder.status(now=2.2)
        self.assertIsNone(status["active"])
        self.assertIsNone(status["latest"])
        self.assertEqual(status["last_summary"]["reason"], "stream_timeout")
        with self.assertRaises(ValueError):
            self.recorder.batch(self.episode, [frame(1, 115)], now=2.3)

    def test_timeout_is_checked_on_ingest(self):
        with self.assertRaises(ValueError):
            self.recorder.batch(self.episode, [frame()], now=3)
        self.assertIsNone(self.recorder.active)

    def test_reference_reset_and_new_episode(self):
        reset = {"type": "event", "seq": 1, "client_time_ms": 105, "name": "reference_reset", "details": {"reference_epoch": 1}}
        after = frame(2, 115)
        after["reference_epoch"] = 1
        self.recorder.batch(self.episode, [frame(), reset, after], now=.1)
        self.assertEqual(self.rows()[2]["input"]["reference_epoch"], 1)
        self.recorder.finish("client_end", now=.2)
        second = self.recorder.start({"reference_space": "local-floor", "performance_time_origin_ms": 1}, now=.3)
        self.assertNotEqual(second, self.episode)
        self.recorder.batch(second, [frame()], now=.4)

    def test_refuses_overwrite_and_second_writer(self):
        with self.assertRaises(FileExistsError):
            quest.Recorder(self.output)
        with self.assertRaises(ValueError):
            self.recorder.start({"reference_space": "local-floor", "performance_time_origin_ms": 1}, now=.1)


if __name__ == "__main__":
    unittest.main()
