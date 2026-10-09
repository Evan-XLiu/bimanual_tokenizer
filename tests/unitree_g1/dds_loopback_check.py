"""Manual transport check: loopback-only, domain 213, non-robot test topic."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/unitree_g1'))
from hold_arms import writer_handle
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.qos import Qos, Policy

ChannelFactoryInitialize(213, 'lo')
participant = DomainParticipant(213)
topic = Topic(participant, 'rt/codex_hold_transport_test', LowCmd_)
reader = DataReader(participant, topic, qos=Qos(Policy.History.KeepAll, Policy.ResourceLimits(max_samples=2048)))
writers = [DataWriter(participant, topic), DataWriter(participant, topic)]
time.sleep(.3)
for writer in writers:
    writer.write(unitree_hg_msg_dds__LowCmd_())
deadline = time.monotonic()+2
samples = []
while time.monotonic() < deadline and len(samples) < 2:
    samples.extend(reader.take(10))
    time.sleep(.01)
actual = [sample.sample_info.publication_handle for sample in samples]
for writer in writers:
    expected = writer_handle(writer)
    assert expected in actual, (expected, actual)
assert writer_handle(writers[0]) != writer_handle(writers[1])
print('PASS: own and foreign DDS writer handles match received sample metadata')
