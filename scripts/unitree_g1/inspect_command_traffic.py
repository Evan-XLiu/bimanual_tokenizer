#!/usr/bin/env python3
"""Observe command-topic traffic; creates subscribers only, never a publisher."""
import argparse
import json
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface', required=True)
    args = parser.parse_args()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    ChannelFactoryInitialize(0, args.interface)
    lock = threading.Lock()
    topics = ('rt/lowcmd', 'rt/arm_sdk', 'rt/user_lowcmd')
    stats = {topic: {'samples': 0} for topic in topics}

    def callback(topic):
        def receive(msg):
            now = time.monotonic()
            with lock:
                row = stats[topic]
                row.setdefault('first_receive_monotonic_s', now)
                row['last_receive_monotonic_s'] = now
                row['samples'] += 1
                row['last_mode_machine'] = int(msg.mode_machine)
                row['last_nonzero_gain_indices'] = [i for i, m in enumerate(msg.motor_cmd) if m.kp != 0 or m.kd != 0]
                row['last_nonzero_feedforward_indices'] = [i for i, m in enumerate(msg.motor_cmd) if m.tau != 0]
        return receive

    subscribers = []
    try:
        for topic in topics:
            subscriber = ChannelSubscriber(topic, LowCmd_)
            subscriber.Init(callback(topic), 10)
            subscribers.append(subscriber)
        started = time.monotonic()
        time.sleep(8)
        with lock:
            print(json.dumps({'read_only': True, 'duration_s': time.monotonic()-started,
                              'topics': stats, 'limitation': 'No observed samples is not proof of exclusive ownership.'}, indent=2))
    finally:
        for subscriber in subscribers:
            subscriber.Close()


if __name__ == '__main__':
    main()
