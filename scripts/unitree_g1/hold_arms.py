#!/usr/bin/env python3
"""Finite current-pose hold experiment. Default is shadow (NO motor publisher)."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import signal
import threading
import time

from arm_execution import ArmExecution, Command, Config, Observation, encode_debug_command


class OwnershipLost(RuntimeError):
    pass


def writer_handle(writer):
    # Pinned CycloneDDS 0.10.2 Entity.get_instance_handle uses int(c_uint64),
    # which raises on Python 3.10. Keep the workaround local to this adapter.
    import ctypes
    from cyclonedds.internal import dds_c_t
    handle = dds_c_t.instance_handle()
    code = writer._get_instance_handle(writer._ref, ctypes.byref(handle))
    if code != 0:
        raise RuntimeError(f'Cannot identify DDS writer: {code}')
    # This binding declares a signed int64 whereas SampleInfo exposes uint64.
    return handle.value & ((1 << 64) - 1)


def load_profile(path, live, candidate_trial=False):
    profile = json.loads(path.read_text())
    config = Config(**profile['controller'])
    if profile['arm_indices'] != list(range(15, 29)):
        raise ValueError('Only verified arm7 ordering 15..28 is supported')
    for key in ('period_s', 'hold_s', 'zero_tail_s', 'max_initial_speed'):
        if not isinstance(profile[key], (int, float)) or not math.isfinite(profile[key]) or profile[key] <= 0:
            raise ValueError(f'Invalid {key}')
    if profile['period_s'] >= config.max_step_seconds or profile['hold_s'] > 10:
        raise ValueError('Invalid servo period or hold duration (maximum 10s)')
    if profile['zero_tail_s'] < profile['period_s'] * 3:
        raise ValueError('Require at least three passive tail frames')
    if live and candidate_trial:
        # Explicit experimental route requested by the operator; never promote
        # candidate parameters to "verified". Restricted to this finite hold.
        baseline = json.loads((Path(__file__).resolve().parents[2] /
            'artifacts/unitree_g1/hold_shadow_profile.json').read_text())
        for key in ('controller', 'arm_indices', 'joint_names', 'mode_machine',
                    'period_s', 'hold_s', 'zero_tail_s', 'max_initial_speed', 'source'):
            if profile.get(key) != baseline[key]:
                raise ValueError(f'Candidate trial differs from reviewed proposal: {key}')
        if profile.get('hardware_approved') is not False:
            raise ValueError('Candidate trial must remain unverified')
        for key in ('body_passive_support', 'arm_unload_support', 'independent_stop'):
            evidence = profile.get('trial_scene', {}).get(key)
            if not isinstance(evidence, str) or not evidence.strip():
                raise ValueError(f'Missing on-site trial readiness: {key}')
    elif live:
        for key in ('joint_mapping', 'gains_and_limits', 'body_passive_support',
                    'arm_unload_support', 'independent_stop', 'debug_exit_behavior'):
            evidence = profile.get('verified', {}).get(key)
            if not isinstance(evidence, str) or not evidence.strip():
                raise ValueError(f'Missing hardware verification evidence: {key}')
        if profile.get('hardware_approved') is not True:
            raise ValueError('Profile is not approved for hardware')
    return profile, config


def run_hold(io, config, profile, stop_requested, log, clock=time.monotonic, sleep=time.sleep,
             target_source=None):
    """io.check owns mode/traffic interlocks; no publishing after ownership loss.

    Signal requests ramp down. Transport/ownership loss is an unconfirmed stop,
    never a successful unload. The io owner closes DDS after this returns.
    """
    last_command = None
    submitted = 0
    logging_failed = False

    def emit(command):
        nonlocal last_command, submitted, logging_failed
        io.check()
        if io.write(command) is not True:
            raise OSError('DDS write failed')
        last_command = command
        submitted += 1
        if not logging_failed:
            try:
                try:
                    feedback = asdict(io.observation())
                except Exception:
                    feedback = None
                log({'type': 'command', 'time': clock(), 'feedback': feedback,
                     'executor_feedback': (None if executor.observation is None else
                                           asdict(executor.observation)),
                     'release_reason': executor.release_reason,
                     'telemetry': io.telemetry() if hasattr(io, 'telemetry') else None,
                     'first_fault': (None if executor.first_fault is None else
                                     asdict(executor.first_fault)), **asdict(command)})
            except Exception:
                logging_failed = True
                raise
        return True

    executor = ArmExecution(config, emit)
    started = clock()
    hold_started = None
    result = {'unload_submitted': False, 'physical_unload_verified': False}
    error = None
    try:
        io.check()
        executor.observe(io.observation())
        if stop_requested():
            raise RuntimeError('Stopped before engagement')
        if target_source is not None:
            target_source.initialize(executor, clock())
        executor.request_control(clock())
        initial_q = executor.q
        while executor.phase not in ('DISARMED', 'FAULT'):
            sleep(profile['period_s'])
            now = clock()
            io.check()
            try:
                executor.observe(io.observation())
            except (ValueError, TypeError) as exc:
                error = f'invalid_feedback: {exc}'
            if stop_requested() and executor.phase != 'RELEASING':
                executor.stop()
            if target_source is not None and executor.phase in ('ENGAGING', 'ACTIVE'):
                target_source.update(executor, now)
            if executor.phase == 'ACTIVE':
                if hold_started is None:
                    hold_started = now
                if now-hold_started >= profile['hold_s']:
                    executor.release_control()
                elif target_source is None:
                    executor.send_targets(initial_q, now, now)
            executor.step(now)
            if now-started > 4*config.ramp_seconds + profile['hold_s'] + 2:
                raise TimeoutError('Experiment exceeded hard deadline')
        if executor.fault:
            error = error or executor.reason
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        # Unexpected exceptions still attempt a bounded unload from LAST SENT
        # gains, only while ownership checks pass. Never restore a service.
        if last_command is not None:
            try:
                begin = clock()
                base = last_command
                while any(last_command.kp + last_command.kd):
                    sleep(profile['period_s'])
                    fraction = max(0., 1-(clock()-begin)/config.ramp_seconds)
                    emit(Command(base.q, tuple(fraction*x for x in base.kp),
                                 tuple(fraction*x for x in base.kd), base.mode_machine,
                                 'RELEASING', 'finally_unload'))
                tail = clock()
                while clock()-tail < profile['zero_tail_s']:
                    sleep(profile['period_s'])
                    emit(Command(base.q, (0.,)*14, (0.,)*14, base.mode_machine,
                                 'RELEASED', 'passive_tail'))
                result['unload_submitted'] = True
            except BaseException as exc:
                result['unload_error'] = f'{type(exc).__name__}: {exc}'
                error = error or result['unload_error']
        result.update(error=error, command_frames=submitted, phase=executor.phase,
                      elapsed_s=clock()-started, release_reason=executor.release_reason,
                      first_fault=(None if executor.first_fault is None else
                                   asdict(executor.first_fault)))
    return result


class DDSHoldIO:
    """Read-only until open_writer. Mode polling is off the servo thread."""
    def __init__(self, interface, live, profile):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.topic import Topic
        from cyclonedds.sub import DataReader
        from cyclonedds.qos import Qos, Policy
        from cyclonedds.core import Listener
        ChannelFactoryInitialize(0, interface)
        self.live, self.profile = live, profile
        self.lock = threading.Lock()
        self.latest = None
        self.latest_telemetry = None
        self.feedback_error = None
        self.writer = None
        self.subscriber = None
        self.mode_ok, self.mode_time = False, 0.
        self.done = threading.Event()
        self.mode_thread = None
        self.readers = []
        self.participant = DomainParticipant(0)
        self.command_type = LowCmd_
        self.command_topic = Topic(self.participant, 'rt/lowcmd', LowCmd_)
        self.own_handle = None
        self.matched = 0
        self.monitor_lost = False
        def lost(*_):
            self.monitor_lost = True
        self.monitor_listener = Listener(on_sample_lost=lost, on_sample_rejected=lost)
        try:
            for name in ('rt/lowcmd', 'rt/arm_sdk', 'rt/user_lowcmd'):
                topic = self.command_topic if name == 'rt/lowcmd' else Topic(self.participant, name, LowCmd_)
                self.readers.append(DataReader(self.participant, topic,
                    qos=Qos(Policy.History.KeepAll, Policy.ResourceLimits(max_samples=2048)),
                    listener=self.monitor_listener))
            self.client = MotionSwitcherClient()
            self.client.SetTimeout(.5)
            self.client.Init()
            self.subscriber = ChannelSubscriber('rt/lowstate', LowState_)
            self.subscriber.Init(self._state, 10)
            self.mode_thread = threading.Thread(target=self._poll_mode, daemon=True)
            self.mode_thread.start()
        except BaseException:
            self.close()
            raise

    def _state(self, msg):
        now = time.monotonic()
        with self.lock:
            try:
                if msg.mode_machine != self.profile['mode_machine'] or msg.mode_pr != 0:
                    raise ValueError('Unexpected machine variant or non-PR feedback')
                if any(m.motorstate != 0 for m in msg.motor_state):
                    raise ValueError('Nonzero motorstate')
                q = tuple(float(msg.motor_state[i].q) for i in range(15, 29))
                dq = tuple(float(msg.motor_state[i].dq) for i in range(15, 29))
                if not all(math.isfinite(v) for v in q+dq):
                    raise ValueError('Nonfinite feedback')
                if self.latest is not None:
                    delta = (msg.tick-self.latest[0].tick) % 2**32
                    if delta == 0:
                        return
                    if delta >= 2**31:
                        raise ValueError('Device tick reset')
                self.latest = (Observation(q, now, int(msg.tick), int(msg.mode_machine)), dq)
                self.latest_telemetry = {
                    'received': now, 'tick': int(msg.tick),
                    'motors': [{'index': i, 'mode': int(m.mode),
                                'vol': float(m.vol), 'temperature': list(m.temperature),
                                'q': float(m.q), 'dq': float(m.dq),
                                'tau_est': float(m.tau_est), 'motorstate': int(m.motorstate)}
                               for i, m in enumerate(msg.motor_state)]}
            except Exception as exc:
                self.feedback_error = str(exc)

    def _poll_mode(self):
        while not self.done.is_set():
            try:
                code, value = self.client.CheckMode()
                ok = code == 0 and isinstance(value, dict) and value.get('name') == ''
            except Exception:
                ok = False
            with self.lock:
                self.mode_ok, self.mode_time = ok, time.monotonic()
            self.done.wait(.2)

    def observation(self):
        with self.lock:
            if self.latest is None:
                raise RuntimeError('No state')
            if self.feedback_error:
                raise ValueError(self.feedback_error)
            return self.latest[0]

    def telemetry(self):
        with self.lock:
            return self.latest_telemetry

    def check(self):
        if self.monitor_lost:
            raise OwnershipLost('Command monitor lost or rejected samples')
        with self.lock:
            if not self.mode_ok or time.monotonic()-self.mode_time > .8:
                raise OwnershipLost('Debug mode not confirmed or mode query stale')
        for reader in self.readers:
            # Bound work; a saturated queue is itself a stop condition.
            samples = reader.take(256)
            for sample in samples:
                info = sample.sample_info
                if info.valid_data and (self.own_handle is None or info.publication_handle != self.own_handle):
                    raise OwnershipLost('Foreign command traffic observed')
            if len(samples) == 256:
                raise OwnershipLost('Command monitor backlog')

    def preflight(self):
        end = time.monotonic()+3
        time.sleep(.8)
        while time.monotonic() < end:
            self.check()
            state = self.observation()
            if not 0 <= time.monotonic()-state.received <= self.profile['controller']['state_timeout']:
                raise RuntimeError('Stale preflight state')
            with self.lock:
                if max(abs(x) for x in self.latest[1]) > self.profile['max_initial_speed']:
                    raise RuntimeError('Arms not stationary')
            time.sleep(.02)

    def open_writer(self):
        self.check()
        if not self.live:
            return
        from cyclonedds.pub import DataWriter
        from cyclonedds.core import Listener
        def matched(entity, status):
            self.matched = status.current_count
        self.listener = Listener(on_publication_matched=matched)
        self.writer = DataWriter(self.participant, self.command_topic, listener=self.listener)
        self.own_handle = writer_handle(self.writer)
        deadline = time.monotonic()+2
        while self.matched == 0:
            self.check()
            if time.monotonic() > deadline:
                raise RuntimeError('No command subscriber matched')
            time.sleep(.01)

    def write(self, command):
        if not self.live:
            return True
        if self.writer is None or self.matched == 0:
            return False
        self.writer.write(encode_debug_command(command))
        return True  # Submission only, not robot acknowledgement.

    def close(self):
        self.done.set()
        if self.mode_thread is not None:
            self.mode_thread.join(timeout=1.)
        if self.subscriber is not None:
            self.subscriber.Close()
        self.writer = None
        self.readers.clear()
        self.command_topic = None
        self.participant = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface', required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Publish ONLY with a verified hardware profile')
    parser.add_argument('--candidate-trial', action='store_true',
                        help='Explicit finite trial of reviewed candidate profile, with scene readiness recorded')
    parser.add_argument('--preflight-only', action='store_true', help='Read-only checks; no hold or publisher')
    args = parser.parse_args()
    if args.preflight_only and args.execute:
        parser.error('--preflight-only cannot be used with --execute')
    if args.candidate_trial and not args.execute:
        parser.error('--candidate-trial requires --execute')
    profile, config = load_profile(args.profile, args.execute, args.candidate_trial)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'profile.json').write_text(json.dumps(profile, indent=2)+'\n')
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    io = None
    result = {'error': 'initialization_failed', 'unload_submitted': False}
    try:
        io = DDSHoldIO(args.interface, args.execute, profile)
        io.preflight()
        if stop.is_set():
            raise RuntimeError('Stopped during preflight')
        if args.preflight_only:
            state = io.observation()
            ArmExecution(config, lambda _: False)._check_limits(state.q)
            result = {'error': None, 'preflight_passed': True, 'observation': asdict(state),
                      'unload_submitted': False, 'command_frames': 0}
        else:
            io.open_writer()
            with (args.output/'commands.jsonl').open('x') as stream:
                def log(row):
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                    stream.flush()
                result = run_hold(io, config, profile, stop.is_set, log)
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if io is not None:
            io.close()
        result['hardware_motion_enabled'] = args.execute
        result['candidate_trial'] = args.candidate_trial
        result['scope'] = 'hardware experiment' if args.execute else 'shadow: no motor publisher'
        result['physical_unload_verified'] = False
        if not args.execute:
            result['unload_submitted'] = False
        (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
    return 1 if result['error'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
