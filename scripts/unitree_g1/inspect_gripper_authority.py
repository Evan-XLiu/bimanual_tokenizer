"""Read-only DDS writer/participant attribution and gripper command sampling."""
import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    from cyclonedds.builtin import (BuiltinDataReader, BuiltinTopicDcpsParticipant,
                                   BuiltinTopicDcpsPublication, BuiltinTopicDcpsSubscription)
    from cyclonedds.domain import DomainParticipant
    from cyclonedds.topic import Topic
    from cyclonedds.sub import DataReader
    ChannelFactoryInitialize(0, args.interface)
    participant = DomainParticipant(0)
    builtins = {name: BuiltinDataReader(participant, topic) for name, topic in (
        ('participants', BuiltinTopicDcpsParticipant),
        ('publishers', BuiltinTopicDcpsPublication),
        ('subscribers', BuiltinTopicDcpsSubscription))}
    readers = {name: DataReader(participant, Topic(participant, name, LowCmd_))
               for name in ('rt/lowcmd', 'rt/arm_sdk', 'rt/user_lowcmd')}
    discovered = {name: {} for name in builtins}
    stats = {}
    started = time.monotonic()
    while time.monotonic()-started < 8:
        for kind, reader in builtins.items():
            for sample in reader.take(256):
                if not sample.sample_info.valid_data:
                    continue
                if kind != 'participants' and not any(s in sample.topic_name for s in
                        ('lowcmd', 'arm_sdk', 'loco', 'sport', 'dex', 'motion_switcher')):
                    continue
                row = dict(key=str(sample.key), instance_handle=sample.sample_info.instance_handle,
                           qos=repr(sample.qos))
                if kind != 'participants':
                    row.update(participant_key=str(sample.participant_key), topic=sample.topic_name)
                discovered[kind][str(sample.key)] = row
        for topic, reader in readers.items():
            for sample in reader.take(512):
                if not sample.sample_info.valid_data:
                    continue
                key = topic+':'+str(sample.sample_info.publication_handle)
                row = stats.setdefault(key, dict(topic=topic,
                    publication_handle=sample.sample_info.publication_handle, samples=0,
                    first_received=time.monotonic(), gripper_signatures={}))
                row['samples'] += 1
                row['last_received'] = time.monotonic()
                signature = json.dumps({str(i):{k:getattr(sample.motor_cmd[i], k)
                    for k in ('mode','q','dq','kp','kd','tau')} for i in (31,33)}, sort_keys=True)
                row['gripper_signatures'][signature] = row['gripper_signatures'].get(signature, 0)+1
        time.sleep(.01)
    result = dict(read_only=True, command_frames_sent=0, elapsed_s=time.monotonic()-started,
                  discovery=discovered, traffic=stats,
                  limitation='DDS metadata is self-reported, not authenticated process identity. '
                  'Command readers use default history: samples are a subset, not a rate measurement.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    attribution = []
    for row in stats.values():
        matches = [p for p in discovered['publishers'].values()
                   if p['instance_handle'] == row['publication_handle']]
        attribution.append(dict(traffic=row, writers=[dict(endpoint=m,
            participant=discovered['participants'].get(m['participant_key'])) for m in matches]))
    print(json.dumps(dict(output=str(args.output), attribution=attribution,
                          limitation=result['limitation']), indent=2))


if __name__ == '__main__':
    main()
