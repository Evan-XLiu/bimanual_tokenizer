"""Finite LEFT internal-Dex1 compatibility probe, NOT a teleoperation backend.

Default is read-only preflight. Only --execute creates an arm_sdk publisher.
Never publishes lowcmd, changes FSM, calibrates, or retries. Slot29 release is
not assumed to release Dex1: slot31 gains are explicitly zeroed first.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import threading
import time

from arm_sdk import ArmSDKIO, encode_arm_sdk_command
from hold_arms import run_hold
from quest_bridge import load_quest_profile
from quest_grippers import gripper_settings

DELTA = .10
ACTIVE_SECONDS = 6.
TINY_DELTA = .01


def position_hold_request(initial, elapsed, phase):
    """Opt-in stock-baseline trial: ramp, hold, tiny move, hold, zero gains."""
    if not math.isfinite(initial) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('Invalid position-hold time/pose')
    if phase != 'ACTIVE' or elapsed >= 5.:
        return dict(q=initial, kp=0., kd=0., tau=0., stage='OFF')
    if elapsed < 1.:
        return dict(q=initial, kp=5.*elapsed, kd=.05*elapsed, tau=0., stage='GAIN_RAMP')
    if elapsed < 2.:
        return dict(q=initial, kp=5., kd=.05, tau=0., stage='CURRENT_HOLD')
    u = min(1., (elapsed-2.)/2.)
    return dict(q=initial+TINY_DELTA*(3*u*u-2*u*u*u), kp=5., kd=.05,
                tau=0., stage='TINY_MOVE' if u < 1 else 'TARGET_HOLD')


def guard_position_downstream(downstream, initial):
    right = downstream['grippers']['33']
    require_stock_torque_baseline({'grippers': {'33': right}})
    left = downstream['grippers']['31']
    if (left['mode'] != 1 or abs(left['dq']) > 1e-6
            or not -1e-6 <= left['kp'] <= 5.00001
            or not -1e-6 <= left['kd'] <= .050001
            or not -.01 <= left['tau'] <= .31):
        raise ValueError('Unexpected downstream left position effort')
    # Unforwarded q=0 is allowed only with no position gain.
    if left['kp'] > 1e-6 and not initial-.005 <= left['q'] <= initial+TINY_DELTA+.005:
        raise ValueError('Unexpected downstream left position target')


def torque_request(initial, elapsed, phase):
    """One 1s pulse: .2s rise, .6s plateau, .2s fall, never more than .10.

    Outside the pulse explicitly request zero effort. This might leave the
    observed stock .30 torque in place; it is not a physical unload claim.
    """
    if not math.isfinite(initial) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('Invalid pulse time/pose')
    tau = 0.
    if phase == 'ACTIVE' and 1. <= elapsed < 2.:
        tau = .1*min(1., (elapsed-1.)/.2, (2.-elapsed)/.2)
    return dict(q=initial, kp=0., kd=0., tau=tau,
                stage='TORQUE_PULSE' if tau > 0 else 'ZERO_REQUEST')


def request(initial, elapsed, phase):
    if not math.isfinite(initial) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('Invalid probe trajectory')
    # 1s gain ramp at initial pose, 3s cubic move, .5s hold, 1.5s off.
    if phase != 'ACTIVE' or elapsed >= 4.5:
        return dict(q=initial, kp=0., kd=0., stage='OFF')
    if elapsed < 1.:
        return dict(q=initial, kp=5.*elapsed, kd=.05*elapsed, stage='GAIN_RAMP')
    u = min(1., (elapsed-1.)/3.)
    return dict(q=initial+DELTA*(3*u*u-2*u*u*u), kp=5., kd=.05,
                stage='MOVE' if u < 1 else 'HOLD')


def encode(command, config, waist, grip):
    from unitree_sdk2py.utils.crc import CRC
    msg = encode_arm_sdk_command(command, config, waist)
    m = msg.motor_cmd[31]
    m.mode, m.q, m.kp, m.kd = 1, grip['q'], grip['kp'], grip['kd']
    m.tau = grip.get('tau', 0.)
    # ONLY left31 is populated; right33 remains untouched.
    msg.crc = CRC().Crc(msg)
    return msg


def gripper_feedback(telemetry, now):
    if telemetry is None or not 0 <= now-telemetry['received'] <= .2:
        raise ValueError('Stale gripper state')
    motors = {m['index']: m for m in telemetry['motors']}
    selected = [motors[i] for i in (31, 33)]
    for m in selected:
        if m['motorstate'] or not all(math.isfinite(m[k]) for k in ('q', 'dq')):
            raise ValueError('Invalid gripper state')
    return selected


def require_passive_baseline(downstream):
    if any(not math.isfinite(m[k]) or abs(m[k]) > 1e-6
           for m in downstream['grippers'].values() for k in ('kp', 'kd', 'tau')):
        raise ValueError('Existing downstream gripper effort; probe refused')


def require_stock_torque_baseline(downstream):
    """Explicit trial accepts only the observed stock signature, not any effort."""
    for m in downstream['grippers'].values():
        for key, expected in dict(mode=1., q=0., dq=0., kp=0., kd=0., tau=.3).items():
            if not math.isfinite(m[key]) or abs(m[key]-expected) > 1e-5:
                raise ValueError('Stock .30 torque baseline changed; probe refused')


class ProbeIO(ArmSDKIO):
    def __init__(self, *args, probe_kind='position'):
        if probe_kind not in ('position', 'torque-pulse', 'position-hold'):
            raise ValueError('Invalid probe kind')
        self.probe_kind = probe_kind
        self.downstream = None
        self.low_sub = None
        self.initial = None
        self.active_since = None
        self.fault = None
        self.last_probe = None
        super().__init__(*args)
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        self.low_sub = ChannelSubscriber('rt/lowcmd', LowCmd_)
        self.low_sub.Init(self.on_lowcmd, 10)

    def on_lowcmd(self, msg):
        row = dict(received=time.monotonic(), grippers={
            str(i): {k: float(getattr(msg.motor_cmd[i], k))
                     for k in ('mode', 'q', 'dq', 'kp', 'kd', 'tau')}
            for i in (31, 33)})
        with self.lock:
            self.downstream = row

    def downstream_snapshot(self, now):
        with self.lock:
            row = self.downstream
        if row is None or not 0 <= now-row['received'] <= .2:
            raise ValueError('No fresh downstream lowcmd for observation')
        if not all(math.isfinite(x) for m in row['grippers'].values() for x in m.values()):
            raise ValueError('Invalid downstream command')
        return row

    def check_readiness(self):
        super().check_readiness()
        motors = gripper_feedback(self.telemetry(), time.monotonic())
        settings = gripper_settings()
        if any(abs(m['dq']) > .1 for m in motors):
            raise ValueError('Grippers not stationary')
        if not -5.4 <= motors[0]['q'] <= settings['closed'][0]-DELTA-.05:
            raise ValueError('Left gripper lacks clearance for small closure')
        downstream = self.downstream_snapshot(time.monotonic())
        if self.probe_kind in ('torque-pulse', 'position-hold'):
            require_stock_torque_baseline(downstream)
        else:
            require_passive_baseline(downstream)

    def guard(self, now):
        self.check_active_body()
        motors = gripper_feedback(self.telemetry(), now)
        downstream = self.downstream_snapshot(time.monotonic())
        if self.probe_kind == 'position-hold':
            guard_position_downstream(downstream, self.initial[0])
        if self.probe_kind == 'torque-pulse':
            for m in downstream['grippers'].values():
                if abs(m['kp']) > 1e-6 or abs(m['kd']) > 1e-6 or not -.01 <= m['tau'] <= .45:
                    raise ValueError('Unexpected downstream gripper effort during pulse')
            if abs(downstream['grippers']['33']['tau']-.3) > .01:
                raise ValueError('Uncommanded right gripper torque changed')
        displacement_limit = .05 if self.probe_kind == 'position-hold' else .15
        if abs(motors[0]['q']-self.initial[0]) > displacement_limit:
            raise ValueError(f'Left gripper displacement exceeds {displacement_limit}')
        if abs(motors[1]['q']-self.initial[1]) > .05:
            raise ValueError('Uncommanded right gripper moved')
        if max(abs(m['dq']) for m in motors) > .3:
            raise ValueError('Gripper speed exceeds .3')
        if self.last_probe and self.last_probe['kp'] > 0:
            if abs(motors[0]['q']-self.last_probe['q']) > .15:
                raise ValueError('Gripper tracking error exceeds .15')

    def write(self, command):
        self.check()
        now = time.monotonic()
        if command.phase == 'ACTIVE' and self.active_since is None:
            self.active_since = now
        elapsed = 0. if self.active_since is None else now-self.active_since
        # Cleanup must remain possible after a body/feedback failure. Ownership
        # checks above still prohibit all writes after loss of mode/authority.
        if command.phase in ('ENGAGING', 'ACTIVE'):
            self.guard(now)
        trajectory = (torque_request if getattr(self, 'probe_kind', 'position') == 'torque-pulse'
                      else request)
        if getattr(self, 'probe_kind', 'position') == 'position-hold':
            trajectory = position_hold_request
        grip = trajectory(self.initial[0], elapsed,
                       'FAULT' if self.fault else command.phase)
        msg = encode(command, self.config, self.waist_q, grip)
        if self.live:
            if self.writer is None or not self.matched:
                return False
            self.writer.write(msg)
        self.last_probe = dict(grip, weight=msg.motor_cmd[29].q,
                               elapsed_active_s=elapsed)
        return True

    def command_metadata(self):
        with self.lock:
            downstream = self.downstream
        return dict(probe_request=self.last_probe, downstream_lowcmd=downstream)

    def close(self):
        if self.low_sub is not None:
            self.low_sub.Close()
            self.low_sub = None
        super().close()


class HoldSource:
    def __init__(self, io):
        self.io = io

    def initialize(self, executor, now):
        self.q = executor.observation.q
        motors = gripper_feedback(self.io.telemetry(), now)
        self.io.initial = tuple(m['q'] for m in motors)

    def update(self, executor, now):
        try:
            self.io.guard(now)
            if executor.phase == 'ACTIVE':
                executor.send_targets(self.q, now, now)
        except Exception as exc:
            self.io.fault = str(exc)
            executor._stop('probe_guard: '+str(exc), True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--probe-kind', choices=('position', 'torque-pulse', 'position-hold'), default='position')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    profile, config = load_quest_profile()
    pulse = args.probe_kind == 'torque-pulse'
    tiny = args.probe_kind == 'position-hold'
    profile['hold_s'] = 4. if pulse else ACTIVE_SECONDS
    source = Path(__file__).read_bytes()
    (args.output/'source.py').write_bytes(source)
    (args.output/'session.json').write_text(json.dumps(dict(
        candidate_only=True, hardware_approved=False, left_index=31, probe_kind=args.probe_kind,
        delta=0. if pulse else TINY_DELTA if tiny else DELTA, pulse_peak_tau=.1 if pulse else None,
        pulse_s=1. if pulse else None, active_s=profile['hold_s'],
        peak_velocity=None if pulse else .0075 if tiny else .05,
        peak_acceleration=None if pulse else .015 if tiny else 2*DELTA/3,
        kp=0. if pulse else 5., kd=0. if pulse else .05, control_mode='arm_sdk', execute=args.execute,
        profile=profile, source_sha256=hashlib.sha256(source).hexdigest(),
        scope='Single left probe, no Quest or right actuation. Release not assumed from weight. '
              'Torque pulse and position-hold accept only observed stock .30 baseline by explicit operator authorization; '
              'zero requests may leave stock torque in place.',
        authorization='User explicitly requested limited compatibility trial this turn.'), indent=2)+'\n')
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    io = None
    result = dict(error=None, command_frames=0)
    try:
        io = ProbeIO(args.interface, args.execute, profile, config, probe_kind=args.probe_kind)
        io.preflight()
        result['preflight_passed'] = True
        result['mode'] = io.mode_evidence
        result['initial_grippers'] = gripper_feedback(io.telemetry(), time.monotonic())
        result['baseline_lowcmd'] = io.downstream_snapshot(time.monotonic())
        if args.execute:
            if stop.is_set():
                raise RuntimeError('Stopped before opening publisher')
            io.open_writer()
            with (args.output/'commands.jsonl').open('x', buffering=1) as log:
                result.update(run_hold(io, config, profile, stop.is_set,
                    lambda row: log.write(json.dumps(row, allow_nan=False)+'\n'),
                    target_source=HoldSource(io)))
            result['arm_sdk_release_submitted'] = bool(result.pop('unload_submitted', False))
            result['gripper_zero_gain_submitted'] = bool(io.last_probe and
                io.last_probe['kp'] == 0 and io.last_probe['kd'] == 0)
            result['gripper_zero_request_submitted'] = bool(result['gripper_zero_gain_submitted']
                and io.last_probe.get('tau', 0.) == 0.)
            result['last_request'] = io.last_probe
            result['final_telemetry'] = io.telemetry()
            result['final_lowcmd'] = io.downstream
    except Exception as exc:
        result['error'] = type(exc).__name__+': '+str(exc)
    finally:
        if io is not None:
            result['last_mode'] = io.mode_evidence
            result['last_downstream_lowcmd'] = io.downstream
            try:
                result['last_gripper_feedback'] = gripper_feedback(io.telemetry(), time.monotonic())
            except Exception as exc:
                result['final_feedback_error'] = str(exc)
            io.close()
        result.update(unload_submitted=False, physical_handover_verified=False,
                      physical_gripper_release_verified=False, hardware_motion_enabled=args.execute)
        (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k != 'final_telemetry'}, indent=2))
    return 1 if result['error'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
