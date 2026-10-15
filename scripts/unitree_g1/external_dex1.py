"""External serial2dds Dex1 transport alongside official locomotion/arm_sdk.

Independent MotorCmds topics, never LowCmd slots 31/33. Physical coordinates
are 0 closed, +5.4 open (official calibrated service convention). A private
sign transform reuses the existing closing-obstruction follower. No calibration
or remote service startup is performed. CLI is strictly read-only.
"""
import argparse
import json
import math
from pathlib import Path
import time
import threading

from arm_sdk import ArmSDKIO
from hold_arms import OwnershipLost, writer_handle
from quest_grippers import GripperFollower, QuestGripperIO

SIDES = ('left', 'right')


def external_settings():
    return dict(indices=[0, 1], opened=[-5.4, -5.4], closed=[0., 0.],
                kp=5., kd=.05, velocity=5.0, acceleration=5.0,
                tracking_error=None, closing_error=.50,
                timeout=.2, input_timeout=1.0, servo_timeout=.1, position_tolerance=.01,
                backend='external-dex1', internal_coordinate='negative of service q',
                physical_opened=[5.4, 5.4], physical_closed=[0., 0.],
                reference='https://github.com/unitreerobotics/dex1_1_service',
                calibration_verified=False)


class ExternalFollower(GripperFollower):
    def feedback(self, telemetry, now):
        q, dq = super().feedback(telemetry, now)
        return tuple(-x for x in q), tuple(-x for x in dq)

    def step(self, *args):
        packet = super().step(*args)
        self.last_sent = dict(packet, q=[-q for q in packet['q']],
                             target=[-q for q in packet['target']],
                             coordinate='external_service_q', backend='external-dex1')
        return self.last_sent


def encode_external(packet, side):
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
    values = [packet[k][side] for k in ('q', 'kp', 'kd')]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite external Dex1 command')
    q, kp, kd = values
    if not -.01 <= q <= 5.41 or not 0 <= kp <= 5. or not 0 <= kd <= .05:
        raise ValueError('External Dex1 command outside candidate bounds')
    m = unitree_go_msg_dds__MotorCmd_()
    m.mode, m.q, m.kp, m.kd = 1, q, kp, kd
    m.dq = m.tau = 0.
    return MotorCmds_(cmds=[m])


class ExternalDex1:
    def __init__(self, participant, live):
        from cyclonedds.topic import Topic
        from cyclonedds.sub import DataReader
        from cyclonedds.qos import Qos, Policy
        from cyclonedds.core import Listener
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_, MotorCmds_
        from unitree_sdk2py.core.channel import ChannelSubscriber
        self.participant, self.live = participant, live
        self.states, self.topics, self.monitors = [], [], []
        self.writers, self.handles = [], []
        self.matches, self.writer_listeners = [0, 0], []
        self.latest = [None, None]
        self.state_lock = threading.Lock()
        self.feedback_error = None
        self.counts = [0, 0]
        self.monitor_lost = False
        self.last_submitted = [None, None]
        self.zero_submitted = [False, False]
        self.listener = Listener(on_sample_lost=self._lost, on_sample_rejected=self._lost)
        for i, side in enumerate(SIDES):
            subscriber = ChannelSubscriber(f'rt/dex1/{side}/state', MotorStates_)
            self.states.append(subscriber)
            subscriber.Init(lambda msg, i=i: self._state(i, msg), 1)
            topic = Topic(participant, f'rt/dex1/{side}/cmd', MotorCmds_)
            self.topics.append(topic)
            self.monitors.append(DataReader(participant, topic,
                qos=Qos(Policy.History.KeepAll, Policy.ResourceLimits(max_samples=2048)), listener=self.listener))

    def _lost(self, *_):
        self.monitor_lost = True

    def check(self):
        if self.monitor_lost:
            raise OwnershipLost('External Dex1 command monitor lost samples')
        for i, reader in enumerate(self.monitors):
            samples = reader.take(256)
            for msg in samples:
                info = msg.sample_info
                if info.valid_data and (len(self.handles) <= i or info.publication_handle != self.handles[i]):
                    raise OwnershipLost('Foreign external Dex1 command traffic')
            if len(samples) == 256:
                raise OwnershipLost('External Dex1 command monitor backlog')

    def _state(self, i, msg):
        with self.state_lock:
            try:
                if len(msg.states) != 1:
                    raise ValueError('External Dex1 requires one motor per side')
                m = msg.states[0]
                if not all(math.isfinite(v) for v in (m.q, m.dq, m.tau_est)) or m.lost:
                    raise ValueError('Invalid external Dex1 feedback')
                self.latest[i] = dict(index=i, q=float(m.q), dq=float(m.dq), tau_est=float(m.tau_est),
                                      motorstate=0, received=time.monotonic(),
                                      status_limitation='Service may not populate motor fault fields')
                self.counts[i] += 1
            except Exception as exc:
                self.feedback_error = str(exc)

    def snapshot(self, now):
        with self.state_lock:
            latest = list(self.latest)
            error = self.feedback_error
        if error:
            raise ValueError(error)
        if any(m is None or not 0 <= now-m['received'] <= .2 for m in latest):
            raise ValueError('Missing or stale external Dex1 left/right state')
        return dict(received=min(m['received'] for m in latest), motors=latest,
                    backend='external-dex1')

    def open(self):
        self.check()
        self.snapshot(time.monotonic())
        if not self.live:
            return
        from cyclonedds.pub import DataWriter
        from cyclonedds.core import Listener
        for i, topic in enumerate(self.topics):
            def matched(entity, status, i=i):
                self.matches[i] = status.current_count
            listener = Listener(on_publication_matched=matched)
            self.writer_listeners.append(listener)
            writer = DataWriter(self.participant, topic, listener=listener)
            self.writers.append(writer)
            self.handles.append(writer_handle(writer))
        # Establish discovery without issuing a command.
        deadline = time.monotonic()+2.
        while not all(count >= 2 for count in self.matches):
            # One subscription is our own command monitor. Require another;
            # fresh state is checked separately, matching is not an ACK.
            self.check()
            self.snapshot(time.monotonic())
            if time.monotonic() > deadline:
                raise RuntimeError('No external Dex1 command subscriber')
            time.sleep(.01)

    def write(self, packet):
        self.check()
        messages = [encode_external(packet, i) for i in range(2)]
        if not self.live:
            return
        if len(self.writers) != 2:
            raise RuntimeError('External Dex1 publishers not open')
        # DDS writes are sequential, not an atomic transaction. Record each side
        # only after success; cleanup independently attempts both sides.
        for i, (writer, msg) in enumerate(zip(self.writers, messages)):
            writer.write(msg)
            self.last_submitted[i] = dict(q=msg.cmds[0].q, kp=msg.cmds[0].kp, kd=msg.cmds[0].kd)
            self.zero_submitted[i] = msg.cmds[0].kp == msg.cmds[0].kd == 0.

    def zero(self):
        """Best-effort independent cleanup, including partial write failures."""
        self.check()  # Never compete with another external gripper controller.
        errors = []
        for i, writer in enumerate(self.writers):
            previous = self.last_submitted[i]
            if previous is None:
                continue
            try:
                packet = dict(q=[previous['q']]*2, kp=[0.]*2, kd=[0.]*2)
                writer.write(encode_external(packet, i))
                self.last_submitted[i] = dict(q=previous['q'], kp=0., kd=0.)
                self.zero_submitted[i] = True
            except Exception as exc:
                errors.append(f'{SIDES[i]}: {exc}')
        if errors:
            raise RuntimeError('; '.join(errors))

    def close(self):
        self.writers.clear()
        for subscriber in self.states:
            subscriber.Close()
        self.states.clear()
        self.monitors.clear()
        self.topics.clear()
        self.participant = None


class ExternalDex1ArmSDKIO(ArmSDKIO):
    def __init__(self, interface, live, profile, config):
        self.external = None
        self.grippers = ExternalFollower(external_settings())
        self.full_arm_kp = profile['controller']['kp'][0]
        self.active_limit = profile['hold_s']
        self.status_path, self.last_status_time, self.last_display = None, 0., None
        self.external_cleanup_error = None
        super().__init__(interface, live, profile, config)
        try:
            self.external = ExternalDex1(self.participant, live)
        except BaseException:
            self.close()
            raise

    def check(self):
        super().check()
        if self.external:
            self.external.check()

    def check_readiness(self):
        super().check_readiness()
        if self.external:
            self.grippers.initialize(self.external.snapshot(time.monotonic()), time.monotonic())

    def open_writer(self):
        super().open_writer()
        self.external.open()

    def initialize_grippers(self, now):
        self.grippers.initialize(self.external.snapshot(now), now)

    def set_gripper_targets(self, values, stamp, now):
        self.grippers.update(values, stamp, now)

    def write(self, command):
        self.check()
        if command.phase in ('ENGAGING', 'ACTIVE'):
            self.check_active_body()
            telemetry = self.external.snapshot(time.monotonic())
            packet = self.grippers.step(command, telemetry, time.monotonic(), self.full_arm_kp)
        else:
            # Exit promptly zeros external gains; it does not depend on stale
            # external feedback, Quest packets, or the arm authority ramp.
            packet = dict(q=[-q for q in self.grippers.q], kp=[0.]*2, kd=[0.]*2,
                          states=[command.phase]*2, phase=command.phase,
                          backend='external-dex1', coordinate='external_service_q')
            self.grippers.last_sent = packet
        self.external.write(packet)
        return super().write(command)

    def command_metadata(self):
        return dict(super().command_metadata(), external_gripper_feedback=self.external.latest,
                    external_gripper_submitted=self.external.last_submitted)

    def telemetry(self):
        telemetry = super().telemetry()
        if telemetry is not None and self.external is not None:
            with self.external.state_lock:
                telemetry = dict(telemetry, external_grippers=list(self.external.latest))
        return telemetry

    decorate_log = QuestGripperIO.decorate_log

    def close(self):
        if self.external:
            try:
                # Dedicated external commands can be cleared even if locomotion
                # changed mode. External ownership is checked separately.
                if self.live:
                    self.external.zero()
            except Exception as exc:
                self.external_cleanup_error = str(exc)
            finally:
                self.external.close()
        super().close()


def preopen_external(interface, output, stop):
    """Finite opening with arms/waist held via arm_sdk, official legs untouched."""
    from hold_arms import run_hold
    from quest_bridge import load_quest_profile
    profile, config = load_quest_profile()
    profile['hold_s'] = 10.
    output.mkdir(parents=True, exist_ok=False)
    io = None
    result = dict(error='initialization_failed', command_frames=0, opening_complete=False)
    writer_open_attempted = False

    class OpenSource:
        complete = False

        def initialize(self, executor, now):
            self.q = executor.observation.q
            io.initialize_grippers(now)

        def update(self, executor, now):
            io.check_active_body()
            if executor.phase == 'ACTIVE':
                io.set_gripper_targets([0., 0.], now, now)
                executor.send_targets(self.q, now, now)
                feedback, _ = io.grippers.feedback(io.external.snapshot(now), now)
                if (all(abs(q+5.4) <= .001 for q in io.grippers.q)
                        and all(abs(v) <= .001 for v in io.grippers.v)
                        and all(abs(q+5.4) <= .15 for q in feedback)):
                    self.complete = True
                    executor.release_control()

    source = OpenSource()
    try:
        io = ExternalDex1ArmSDKIO(interface, True, profile, config)
        io.preflight()
        if stop.is_set():
            raise RuntimeError('Stopped before external preopen')
        writer_open_attempted = True
        io.open_writer()
        with (output/'commands.jsonl').open('x', buffering=1) as log:
            result = run_hold(io, config, profile, stop.is_set,
                lambda row: log.write(json.dumps(io.decorate_log(row), allow_nan=False)+'\n'),
                target_source=source)
        result['opening_complete'] = source.complete
        result['arm_sdk_release_submitted'] = result.pop('unload_submitted', False)
        if not source.complete or stop.is_set():
            result['error'] = result.get('error') or 'External preopen incomplete or stopped'
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if io:
            io.close()
            result.update(external_gripper_zero_submitted=list(io.external.zero_submitted),
                          external_gripper_cleanup_error=io.external_cleanup_error)
            result['error'] = result.get('error') or io.external_cleanup_error
        result.update(control_publish_not_started=not writer_open_attempted,
                      unload_submitted=False, physical_gripper_release_verified=False,
                      physical_handover_verified=False, backend='external-dex1')
        (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    if result['error']:
        raise RuntimeError('External preopen failed: '+result['error'])


def main():
    parser = argparse.ArgumentParser(description='Read-only arm_sdk + external Dex1 preflight')
    parser.add_argument('--interface', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from quest_bridge import load_quest_profile
    profile, config = load_quest_profile()
    io = None
    result = dict(read_only=True, command_frames=0, preflight_passed=False)
    try:
        io = ExternalDex1ArmSDKIO(args.interface, False, profile, config)
        io.preflight()
        result['preflight_passed'] = True
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if io:
            result.update(mode=io.mode_evidence, external_feedback=io.external.latest,
                          external_counts=io.external.counts)
            io.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        print(json.dumps(result, indent=2))
    return 0 if result['preflight_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
