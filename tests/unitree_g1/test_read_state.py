"""Run with unittest; no training dependencies, SDK, or robot required."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    "g1_read_state", Path(__file__).resolve().parents[2] / "scripts/unitree_g1/read_state.py"
)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def message(tick=1):
    return SimpleNamespace(
        motor_state=[SimpleNamespace(q=float(i), dq=0.0, tau_est=0.0, motorstate=0) for i in range(35)],
        tick=tick, version=[1, 0], mode_machine=0, mode_pr=0,
    )


class ReadStateTests(unittest.TestCase):
    def test_reference_indices_and_no_fabricated_gripper(self):
        row = reader.snapshot(message(), 1.0)
        self.assertEqual(row["reference_left_q"], list(range(15, 22)))
        self.assertEqual(row["reference_right_q"], list(range(22, 29)))
        self.assertEqual(len(row["motors"]), 35)

    def test_invalid_feedback(self):
        msg = message()
        msg.motor_state[15].q = float("nan")
        with self.assertRaises(ValueError):
            reader.snapshot(msg, 1.0)
        msg.motor_state = msg.motor_state[:23]
        with self.assertRaises(ValueError):
            reader.snapshot(msg, 1.0)

    def test_no_data_and_frozen_tick_timeout(self):
        stats = reader.StreamStats(0)
        with self.assertRaises(TimeoutError):
            stats.check_timeout(2, 2)
        stats.observe(reader.snapshot(message(), 0.1))
        stats.observe(reader.snapshot(message(), 1.9))
        with self.assertRaises(TimeoutError):
            stats.check_timeout(2.2, 2)
        self.assertEqual(stats.duplicates, 1)

    def test_tick_wrap_and_reset(self):
        stats = reader.StreamStats(0)
        stats.observe(reader.snapshot(message(2**32 - 1), 0.1))
        stats.observe(reader.snapshot(message(0), 0.2))
        self.assertEqual(stats.tick_advances, 1)
        with self.assertRaises(ValueError):
            stats.observe(reader.snapshot(message(2**32 - 1), 0.3))

    def run_stream(self, frozen=False, absent=False):
        class Subscriber:
            now = 0.0
            count = 0

            def Read(self, timeout):
                self.now += timeout
                self.count += 1
                return None if absent else message(1 if frozen else self.count)

        sub, rows = Subscriber(), []
        code = reader.monitor(sub, 1.0, 0.4, rows.append, clock=lambda: sub.now)
        return code, rows[-1]

    def test_monitor_success(self):
        code, summary = self.run_stream()
        self.assertEqual(code, 0)
        self.assertAlmostEqual(summary["host_receive_hz"], 10)

    def test_monitor_timeout_exit(self):
        for kwargs in ({"absent": True}, {"frozen": True}):
            code, summary = self.run_stream(**kwargs)
            self.assertEqual(code, 1)
            self.assertEqual(summary["result"], "failed")

    def test_invalid_cli_times(self):
        for value in ("nan", "inf", "-1", "0"):
            with self.assertRaises(reader.argparse.ArgumentTypeError):
                reader.positive_seconds(value)


if __name__ == "__main__":
    unittest.main()
