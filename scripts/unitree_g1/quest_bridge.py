#!/usr/bin/env python3
"""Finite Quest/G1 candidate trial. Default: real feedback, NO motor publisher.

The IK subprocess is forbidden from importing DDS. The parent alone owns the
bounded executor. Optional Dex1 commands share the same publisher. No replay
input, automatic rearming or mode changes. Each run requires a fresh gesture.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from arm_execution import vector
from hold_arms import DDSHoldIO, load_profile, run_hold
from quest_grippers import QuestGripperIO, gripper_settings, raw_triggers, trigger_pair

ROOT = Path(__file__).resolve().parents[2]
LIMITS = dict(active_s=60., input_timeout_s=.1, feedback_timeout_s=.1,
              joint_excursion_rad=.20, velocity_rad_s=.05, acceleration_rad_s2=.1,
              translation_scale=1., workspace_m=.10, ik_position_error_m=.015,
              tracking_error_rad=.10, workspace_shape='box_above_start',
              orientation_mode='free')
IK_COST = dict(translation=500., rotation=0., smooth=.1)


def load_quest_profile():
    # Quest-only trial override; preserve the original hold/probe profile.
    profile, config = load_profile(ROOT/'artifacts/unitree_g1/hold_shadow_profile.json', False)
    profile['hold_s'] = LIMITS['active_s']
    tracking = (LIMITS['tracking_error_rad'],)*14
    profile['controller']['tracking_error'] = list(tracking)
    return profile, replace(config, tracking_error=tracking)


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, allow_nan=False)+'\n')
    temp.replace(path)


def read_packet(path, nonce, now):
    packet = json.loads(path.read_text())
    if packet.get('nonce') != nonce:
        raise ValueError('Wrong session target')
    for key in ('input_time', 'computed_time'):
        stamp = packet[key]
        if not math.isfinite(stamp) or not 0 <= now-stamp <= LIMITS['input_timeout_s']:
            raise ValueError('Stale or future Quest target')
    return packet


def configure_measured_pose_ik(robot):
    # Keep upstream kinematics and solver, but remove its absolute-zero bias.
    # With targets at measured FK, all remaining costs have a zero minimum at
    # the measured joint pose. Scope is this worker only, not the virtual demo.
    ik = robot.ik
    # Position-only task: omit the rotation expression entirely (not 0*cost),
    # so its derivatives cannot constrain motion or introduce rotation NaNs.
    ik.opti.minimize(IK_COST['translation'] * ik.translational_cost +
                     IK_COST['smooth'] * ik.smooth_cost)
    ik.trial_reference = ik.opti.parameter(14)
    # Numerical margin keeps the nonlinear solver inside the executor's exact
    # envelope, without expanding either the hardware or acceptance bounds.
    radius = LIMITS['joint_excursion_rad'] - 1e-6
    ik.opti.subject_to(ik.opti.bounded(ik.trial_reference-radius,
                                     ik.var_q, ik.trial_reference+radius))


def bounded_ik(robot, requested, references, previous, feedback, evidence):
    """Project position requests, with orientation freely chosen by the IK.

    Backtracking is bounded to five fresh solves, including revalidation of the
    last accepted target. Input timestamps are preserved through all attempts.
    """
    import numpy as np
    targets = [t.copy() for t in requested]
    limited = []
    for target, reference in zip(targets, references):
        delta = target[:3, 3] - reference[:3, 3]
        # Model-frame box relative to engagement: preserve horizontal bounds,
        # allow combined lift/lateral motion, and do not request below start.
        radius = LIMITS['workspace_m']
        bounded = np.clip(delta, [-radius, -radius, 0.], [radius, radius, radius])
        limited.append(bool(np.any(delta != bounded)))
        target[:3, 3] = reference[:3, 3] + bounded
        # Upstream takes SE(3) inputs; rotation here is just a neutral placeholder.
        # The position-only objective does not use it or controller rotation.
        target[:3, :3] = reference[:3, :3]
    evidence.update(requested_target=[t.tolist() for t in requested],
                    workspace_target=[t.tolist() for t in targets],
                    workspace_limited=limited, orientation_mode='free', attempts=[])
    seed = np.asarray(vector(feedback['q']))
    initial = np.asarray(vector(feedback['initial_q']))
    robot.ik.opti.set_value(robot.ik.trial_reference, initial)
    for fraction in (1., .5, .25, .125, 0.):
        trial = [t.copy() for t in targets]
        for t, base in zip(trial, previous):
            t[:3, 3] = base[:3, 3] + fraction * (t[:3, 3]-base[:3, 3])
        robot.ik.smooth_filter._data_queue = []
        candidate, _ = robot.ik.solve_ik(*trial, seed.copy(), np.zeros(14))
        q = vector(candidate)
        attempt = dict(fraction=fraction, candidate_q=q)
        evidence['attempts'].append(attempt)
        if not robot.ik.opti.stats().get('success'):
            raise ValueError('IK failed')
        # A solver violating its constraints is a fault, not operator saturation.
        if (np.any(np.abs(np.asarray(q)-initial) > LIMITS['joint_excursion_rad'])
                or np.any(np.asarray(q) < robot.model.lowerPositionLimit)
                or np.any(np.asarray(q) > robot.model.upperPositionLimit)):
            raise ValueError('IK violates joint constraints')
        robot.q = np.asarray(q)
        achieved = robot.forward()
        if not all(np.isfinite(a).all() for a in achieved):
            raise ValueError('Nonfinite IK forward kinematics')
        errors = [float(np.linalg.norm(a[:3, 3]-b[:3, 3])) for a,b in zip(achieved, trial)]
        # Diagnostic change from engagement orientation, not an acceptance error.
        angles = [math.acos(float(np.clip((np.trace(a[:3, :3] @ b[:3, :3].T)-1)/2, -1, 1)))
                  for a,b in zip(achieved, references)]
        attempt.update(ik_position_error_m=errors, ik_rotation_change_rad=angles)
        if max(errors) <= LIMITS['ik_position_error_m']:
            evidence.update(target=[t.tolist() for t in trial], candidate_q=q,
                            ik_position_error_m=errors, ik_rotation_change_rad=angles,
                            ik_limited=fraction < 1., accepted_fraction=fraction)
            return q, errors, trial
    raise ValueError('IK cannot revalidate bounded target')


class QuestSource:
    def __init__(self, output, nonce, clock=time.monotonic, gripper_io=None):
        self.output, self.nonce = output, nonce
        self.clock = clock
        self.initial = None
        self.last_input = None
        self.session = None
        self.gripper_io = gripper_io

    def snapshot(self, io, phase, q=None):
        state = io.observation()
        write_json(self.output/'feedback.json', dict(nonce=self.nonce,
            time=state.received, q=list(state.q), command_q=list(q or state.q),
            phase=phase, initial_q=self.initial))

    def initialize(self, executor, now):
        packet = read_packet(self.output/'target.json', self.nonce, now)
        if packet['state'] != 'following' or packet['phase'] != 'WAITING':
            raise ValueError('Fresh release/press required before engagement')
        if self.gripper_io is not None:
            trigger_pair(packet.get('triggers'))
            self.gripper_io.initialize_grippers(self.clock())
        self.session = packet['session']
        self.initial = executor.observation.q
        self.last_input = packet['input_time']
        c = executor.c
        radius = LIMITS['joint_excursion_rad']
        executor.c = replace(c,
            lower=tuple(max(lo, q-radius) for lo, q in zip(c.lower, self.initial)),
            upper=tuple(min(hi, q+radius) for hi, q in zip(c.upper, self.initial)),
            velocity=(LIMITS['velocity_rad_s'],)*14,
            acceleration=(LIMITS['acceleration_rad_s2'],)*14)

    def update(self, executor, now):
        state = executor.observation
        write_json(self.output/'feedback.json', dict(nonce=self.nonce,
            time=state.received, q=list(state.q), command_q=list(executor.q),
            phase=executor.phase, initial_q=self.initial))
        try:
            if (self.output/'worker_error.json').exists():
                failure = json.loads((self.output/'worker_error.json').read_text())
                raise ValueError('IK worker: '+failure['error'])
            packet = read_packet(self.output/'target.json', self.nonce, self.clock())
            if packet['session'] != self.session:
                raise ValueError('Quest connection changed')
            if packet['state'] in ('ready', 'hold', 'operator_release'):
                executor.release_control()
                return
            if packet['state'] != 'following':
                raise ValueError('Quest stopped: '+packet['state'])
            if packet['input_time'] < self.last_input:
                raise ValueError('Nonmonotonic Quest target')
            if packet['input_time'] == self.last_input:
                return  # A cached packet must never renew target validity.
            self.last_input = packet['input_time']
            if executor.phase == 'ACTIVE':
                # Reject packets computed against a previous execution phase.
                if packet['phase'] != 'ACTIVE' or packet['input_time'] <= executor.target_time:
                    return
                q = vector(packet['q'])
                if self.gripper_io is not None:
                    self.gripper_io.set_gripper_targets(packet.get('triggers'),packet['input_time'],self.clock())
                executor.send_targets(q, packet['input_time'], self.clock())
        except (ValueError, KeyError, TypeError, OSError) as exc:
            executor._stop('quest_input: '+str(exc), True)


def worker(output, nonce, done, with_grippers=False):
    # The spawn context isolates VirtualRobot's NoHardwareImports guard from
    # the parent DDS process. All IK results are proposals, never DDS commands.
    import numpy as np
    from virtual_arms import VirtualRobot, RelativeClutch, read_available
    clutch = RelativeClutch(LIMITS['input_timeout_s'], LIMITS['translation_scale'],
                            workspace_m=None)
    stream = None
    terminal = None
    engaged = False
    previous_phase = None
    evidence = None
    previous_targets = None
    last_boundary = None
    try:
        with tempfile.TemporaryDirectory(prefix='g1-quest-ik-') as tmp:
            robot = VirtualRobot(Path(tmp), False)
            configure_measured_pose_ik(robot)
            expected = json.loads((ROOT/'artifacts/unitree_g1/hold_shadow_profile.json').read_text())['joint_names']
            if list(robot.model.names)[1:] != expected:
                raise ValueError('IK/executor joint ordering mismatch')
            write_json(output/'ik_ready.json', {'nonce': nonce})
            with (output/'proposals.jsonl').open('x', buffering=1) as log:
                while not done.is_set():
                    if terminal is not None:
                        done.wait(.01)
                        continue  # A release/fault cannot be overwritten by later input.
                    path = output/'input/events.jsonl'
                    if stream is None and path.exists():
                        stream = path.open()
                    if not (output/'feedback.json').exists() or stream is None:
                        done.wait(.01)
                        continue
                    feedback = json.loads((output/'feedback.json').read_text())
                    if feedback['nonce'] != nonce:
                        raise ValueError('Feedback session mismatch')
                    events = read_available(stream)
                    if len(events) > 128:
                        raise ValueError('Input backlog')
                    for event in events:
                        if event['kind'] != 'CONTROLLER_MOVE':
                            continue
                        now = time.monotonic()
                        if not 0 <= now-feedback['time'] <= LIMITS['feedback_timeout_s']:
                            if engaged:
                                raise ValueError('Stale robot feedback in IK')
                            continue
                        phase = feedback['phase']
                        robot.q = np.asarray(vector(feedback['q']))
                        robot_poses = robot.forward()
                        # Once gains are up, rebase to the hands and measured
                        # robot pose at that moment; movement during ramp is ignored.
                        if phase == 'ACTIVE' and previous_phase != 'ACTIVE':
                            clutch.reference = None
                            previous_targets = None
                        previous_phase = phase
                        targets = clutch.accept(event, now, robot_poses)
                        q = vector(feedback['command_q'])
                        errors = None
                        if targets is not None and phase == 'ACTIVE':
                            evidence = dict(input_time=event['host_receive_monotonic_s'],
                                feedback=feedback, target=[t.tolist() for t in targets])
                            q, errors, previous_targets = bounded_ik(
                                robot, targets, clutch.robot_reference,
                                previous_targets if previous_targets is not None else clutch.robot_reference,
                                feedback, evidence)
                            boundary = (any(evidence['workspace_limited']), evidence['ik_limited'])
                            if boundary != last_boundary:
                                print(f'Operation boundary: workspace_limited={boundary[0]}, '
                                      f'ik_limited={boundary[1]}', flush=True)
                                last_boundary = boundary
                        state = clutch.state
                        if engaged and state != 'following':
                            terminal = state
                        if state == 'following':
                            engaged = True
                        packet = dict(nonce=nonce, session=event['session'],
                            input_time=event['host_receive_monotonic_s'], computed_time=time.monotonic(),
                            phase=phase, state=state, q=q, ik_position_error_m=errors)
                        if with_grippers:
                            packet['triggers']=raw_triggers(event)
                        if errors is not None:
                            packet['ik_evidence'] = evidence
                        log.write(json.dumps(packet, allow_nan=False)+'\n')
                        write_json(output/'target.json', packet)
                        if terminal is not None:
                            break
                    done.wait(.002)
    except BaseException as exc:
        write_json(output/'worker_error.json', {'error': f'{type(exc).__name__}: {exc}',
                                               'last_ik_evidence': evidence})
    finally:
        if stream:
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface', required=True)
    parser.add_argument('--cert', type=Path, required=True)
    parser.add_argument('--key', type=Path, required=True)
    parser.add_argument('--host-address', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--grippers', action='store_true', help='Enable tested-range Dex1 trigger control in the single publisher')
    parser.add_argument('--scene-confirmation', help='Current operator statement, not historical profile readiness')
    parser.add_argument('--wait-seconds', type=float, default=300.)
    args = parser.parse_args()
    if not math.isfinite(args.wait_seconds) or not 0 < args.wait_seconds <= 600:
        parser.error('wait-seconds must be in (0, 600]')
    if args.execute and not (args.scene_confirmation or '').strip():
        parser.error('--execute requires current --scene-confirmation')
    for path in (args.cert, args.key):
        if not path.is_file():
            parser.error('TLS file missing')
    # Preserve candidate gains and hardware_approved=false; log Quest overrides.
    profile, config = load_quest_profile()
    grip_settings = gripper_settings() if args.grippers else None
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    nonce = uuid.uuid4().hex
    (output/'source').mkdir()
    source_hashes = {}
    for name in ('quest_bridge.py', 'quest_grippers.py', 'arm_execution.py', 'hold_arms.py', 'virtual_arms.py', 'televuer_input.py'):
        data = (ROOT/'scripts/unitree_g1'/name).read_bytes()
        (output/'source'/name).write_bytes(data)
        source_hashes[name] = hashlib.sha256(data).hexdigest()
    write_json(output/'session.json', dict(nonce=nonce, limits=LIMITS, ik_cost=IK_COST, profile=profile,
        grippers=grip_settings,
        hardware_motion_enabled=args.execute, scene_confirmation=args.scene_confirmation,
        interface=args.interface, host_address=args.host_address,
        source_sha256=source_hashes,
        scope='candidate translation-only trial; no collision or general teleoperation acceptance'))
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    context = multiprocessing.get_context('spawn')
    done = context.Event()
    io = input_proc = ik_proc = input_log = None
    result = dict(error='initialization_failed', command_frames=0, unload_submitted=False)
    source = QuestSource(output, nonce)
    try:
        io = (QuestGripperIO(args.interface,args.execute,profile,grip_settings) if args.grippers
              else DDSHoldIO(args.interface, args.execute, profile))
        source.gripper_io=io if args.grippers else None
        io.preflight()
        source.snapshot(io, 'WAITING')
        input_log = (output/'input.log').open('x')
        input_proc = subprocess.Popen([sys.executable, str(ROOT/'scripts/unitree_g1/televuer_input.py'),
            '--cert', str(args.cert.resolve()), '--key', str(args.key.resolve()),
            '--output', str(output/'input'), '--host-address', args.host_address,
            '--timeout', str(LIMITS['input_timeout_s'])], stdout=input_log, stderr=subprocess.STDOUT)
        ik_proc = context.Process(target=worker, args=(output, nonce, done, args.grippers))
        ik_proc.start()
        print(f'Quest: https://{args.host_address}:8012/?ws=wss://{args.host_address}:8012', flush=True)
        print('MODE: '+('LIVE CANDIDATE' if args.execute else 'SHADOW: NO MOTOR PUBLISHER'), flush=True)
        print('Release both grips, then hold both. Keep still for 2s ramp; small translations only. '
              'Position control; end-effector orientation free (controller rotation is not a target). '
              'Workspace: forward/lateral +/-10 cm, up 0..10 cm; initial joints +/-0.20 rad. '
              'Release either grip to end this run. No automatic rearm.', flush=True)
        if args.grippers:
            print('Dex1: after ramp, squeeze each INDEX trigger fully once to pick up control; '
                  'then release trigger=open, squeeze=close. Side grips remain the stop/clutch. '
                  'Grippers unload with arms; no object holding on exit.',flush=True)
        deadline = time.monotonic()+args.wait_seconds
        while not stop.is_set():
            now = time.monotonic()
            if now > deadline:
                raise TimeoutError('No fresh Quest engagement before wait deadline')
            io.check()
            if input_proc.poll() is not None or not ik_proc.is_alive():
                raise RuntimeError('Input or IK process exited; inspect session logs')
            source.snapshot(io, 'WAITING')
            try:
                packet = read_packet(output/'target.json', nonce, time.monotonic())
            except (FileNotFoundError, ValueError):
                packet = None
            if packet and packet['state'] == 'following':
                # Fresh stationary feedback and original profile limits required
                # again immediately before any publisher is opened.
                state = io.observation()
                if not 0 <= time.monotonic()-state.received <= config.state_timeout:
                    raise ValueError('Stale engagement feedback')
                if any(not lo <= q <= hi for q,lo,hi in zip(state.q,config.lower,config.upper)):
                    raise ValueError('Engagement pose outside profile')
                with io.lock:
                    if max(abs(x) for x in io.latest[1]) > profile['max_initial_speed']:
                        raise ValueError('Arms not stationary at engagement')
                io.open_writer()
                with (output/'commands.jsonl').open('x', buffering=1) as log:
                    result = run_hold(io, config, profile, stop.is_set,
                        lambda row: log.write(json.dumps(io.decorate_log(row) if args.grippers else row,
                                                       allow_nan=False)+'\n'), target_source=source)
                break
            time.sleep(.01)
        else:
            result['error'] = 'operator_stop_before_engagement'
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        done.set()
        if ik_proc:
            ik_proc.join(timeout=3)
            if ik_proc.is_alive():
                ik_proc.terminate()
                ik_proc.join(timeout=2)
        if input_proc and input_proc.poll() is None:
            input_proc.terminate()
            try:
                input_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                input_proc.kill()
                input_proc.wait()
        if input_log:
            input_log.close()
        if io:
            if args.grippers:
                result['gripper_fault']=io.grippers.fault
                result['last_gripper_command']=io.grippers.last_sent
            io.close()
        result.update(hardware_motion_enabled=args.execute, physical_unload_verified=False)
        if not args.execute:
            result['unload_submitted'] = False
        write_json(output/'summary.json', result)
        print(json.dumps(result, indent=2), flush=True)
    return 1 if result['error'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
