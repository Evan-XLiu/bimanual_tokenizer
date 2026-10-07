#!/usr/bin/env python3
"""Read G1 LowState only. No command publisher or control-mode client.

Standalone: does not import the model-training environment. JSONL contains
periodic diagnostic snapshots, not a synchronized training dataset.
"""

import argparse
import json
import math
import socket
import sys
import time
from contextlib import nullcontext
from pathlib import Path


def positive_seconds(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def snapshot(msg, received_at):
    motors = [
        {"index": i, "q": float(m.q), "dq": float(m.dq),
         "tau_est": float(m.tau_est), "motorstate": int(m.motorstate)}
        for i, m in enumerate(msg.motor_state)
    ]
    if len(motors) < 29:
        raise ValueError("G1 seven-joint arm reference requires at least 29 motor slots")
    for m in motors:
        if not all(math.isfinite(m[k]) for k in ("q", "dq", "tau_est")):
            raise ValueError(f"non-finite motor feedback at index {m['index']}")
    return {
        "type": "state", "host_receive_monotonic_s": received_at,
        "device_tick_raw": int(msg.tick), "version_raw": list(msg.version),
        "mode_machine_raw": int(msg.mode_machine), "mode_pr_raw": int(msg.mode_pr),
        "motors": motors,
        # Official arm7 example reference only; not a verified hardware profile.
        "reference_left_q": [m["q"] for m in motors[15:22]],
        "reference_right_q": [m["q"] for m in motors[22:29]],
    }


class StreamStats:
    def __init__(self, started):
        self.started = started
        self.first = None
        self.last = None
        self.last_tick = None
        self.last_progress = started
        self.messages = 0
        self.tick_advances = 0
        self.duplicates = 0
        self.max_gap = 0.0

    def observe(self, row):
        now, tick = row["host_receive_monotonic_s"], row["device_tick_raw"]
        if self.last is not None:
            self.max_gap = max(self.max_gap, now - self.last)
            delta = (tick - self.last_tick) % (2**32)
            if delta >= 2**31:
                raise ValueError("device tick moved backwards (reset or out-of-order data)")
            if delta:
                self.tick_advances += 1
                self.last_progress = now
            else:
                self.duplicates += 1
        else:
            self.first = now
            self.last_progress = now
        self.last, self.last_tick = now, tick
        self.messages += 1

    def check_timeout(self, now, timeout):
        if self.last is None:
            if now - self.started >= timeout:
                raise TimeoutError("no rt/lowstate received; check NIC, subnet, DDS and robot")
        elif now - self.last_progress >= timeout:
            raise TimeoutError("state stream stopped or device tick stopped advancing")

    def summary(self):
        elapsed = self.last - self.first if self.messages > 1 else 0.0
        return {"messages": self.messages, "tick_advances": self.tick_advances,
                "duplicate_ticks": self.duplicates,
                "host_receive_hz": (self.messages - 1) / elapsed if elapsed > 0 else 0.0,
                "max_host_receive_gap_s": self.max_gap}


def monitor(subscriber, duration, timeout, emit, clock=time.monotonic):
    started = clock()
    stats = StreamStats(started)
    last_print = started - 1
    try:
        while clock() - started < duration:
            # Check before accepting new data so a long gap is not hidden.
            stats.check_timeout(clock(), timeout)
            msg = subscriber.Read(0.1)
            now = clock()
            stats.check_timeout(now, timeout)
            if msg is not None:
                row = snapshot(msg, now)
                stats.observe(row)
                if now - last_print >= 1:
                    emit(row)
                    last_print = now
        if stats.tick_advances < 2:
            raise TimeoutError("insufficient advancing state samples; increase duration")
    except (ValueError, TimeoutError, KeyboardInterrupt) as exc:
        emit({"type": "summary", "result": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
              "reason": str(exc), **stats.summary()})
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    emit({"type": "summary", "result": "received_advancing_state", **stats.summary()})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True, help="explicit wired NIC, e.g. enp130s0")
    parser.add_argument("--duration", type=positive_seconds, default=30.0)
    parser.add_argument("--timeout", type=positive_seconds, default=2.0,
                        help="diagnostic no-progress timeout, not a robot safety setting")
    parser.add_argument("--output", type=Path, help="new JSONL file; refuses overwrite")
    args = parser.parse_args()
    try:
        if args.interface not in {name for _, name in socket.if_nameindex()}:
            raise ValueError(f"network interface not found: {args.interface}")
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

        # Open before connecting to avoid running if the destination is invalid.
        with args.output.open("x", encoding="utf-8") if args.output else nullcontext() as output:
            def emit(row):
                line = json.dumps(row, allow_nan=False)
                print(line, flush=True)
                if output is not None:
                    output.write(line + "\n")
                    output.flush()

            emit({"type": "metadata", "schema": "g1-read-state-v1", "read_only": True,
                  "interface": args.interface, "topic": "rt/lowstate", "domain_id": 0,
                  "duration_s": args.duration, "timeout_s": args.timeout,
                  "arm_indices_verified_on_hardware": False, "gripper_feedback": "not_read",
                  "note": "1 Hz diagnostic snapshots; receive rate is not servo rate or latency"})
            ChannelFactoryInitialize(0, args.interface)
            subscriber = ChannelSubscriber("rt/lowstate", LowState_)
            subscriber.Init()
            try:
                return monitor(subscriber, args.duration, args.timeout, emit)
            finally:
                subscriber.Close()
    except ImportError as exc:
        print(f"SDK unavailable: {exc}. See docs/workflows/unitree_g1.md", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Read-only diagnostic failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
