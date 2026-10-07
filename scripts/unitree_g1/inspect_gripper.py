#!/usr/bin/env python3
"""Discover gripper endpoints and feedback without creating command publishers."""

import argparse
import json
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True)
    args = parser.parse_args()

    from cyclonedds.builtin import (
        BuiltinDataReader, BuiltinTopicDcpsPublication, BuiltinTopicDcpsSubscription,
    )
    from cyclonedds.domain import DomainParticipant
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, LowState_

    ChannelFactoryInitialize(0, args.interface)
    participant = DomainParticipant(0)
    readers = [("publisher", BuiltinDataReader(participant, BuiltinTopicDcpsPublication)),
               ("subscriber", BuiltinDataReader(participant, BuiltinTopicDcpsSubscription))]
    lock = threading.Lock()
    feedback = {}

    def callback(topic):
        def receive(msg):
            motors = msg.states if hasattr(msg, "states") else msg.motor_state
            indices = (31, 33) if topic == "rt/lowstate" else range(len(motors))
            row = {"motor_count": len(motors), "q": {str(i): float(motors[i].q) for i in indices if i < len(motors)},
                   "dq": {str(i): float(motors[i].dq) for i in indices if i < len(motors)}}
            if hasattr(msg, "tick"):
                row["tick"] = int(msg.tick)
            with lock:
                row["samples"] = feedback.get(topic, {}).get("samples", 0) + 1
                feedback[topic] = row
        return receive

    subscribers = []
    seen = set()
    try:
        for topic, msg_type in (
            ("rt/dex1/right/state", MotorStates_), ("rt/dex1/left/state", MotorStates_),
            ("rt/dex3/right/state", HandState_), ("rt/dex3/left/state", HandState_),
            ("rt/lowstate", LowState_),
        ):
            sub = ChannelSubscriber(topic, msg_type)
            sub.Init(callback(topic), 10)
            subscribers.append(sub)
        end = time.monotonic() + 6
        while time.monotonic() < end:
            for kind, reader in readers:
                for sample in reader.take(100):
                    if not hasattr(sample, "topic_name"):
                        continue
                    topic = sample.topic_name
                    if not any(word in topic for word in ("dex", "gripper", "lowcmd", "arm_sdk")):
                        continue
                    # Ignore our own state subscriptions in the report.
                    if kind == "subscriber" and topic.endswith("/state"):
                        continue
                    seen.add((kind, topic, sample.type_name))
            time.sleep(0.05)
        with lock:
            print(json.dumps({"read_only": True, "endpoints": sorted(seen), "feedback": feedback}, indent=2))
    finally:
        for sub in subscribers:
            sub.Close()


if __name__ == "__main__":
    main()
