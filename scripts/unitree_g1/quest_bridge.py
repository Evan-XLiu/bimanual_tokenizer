#!/usr/bin/env python3
"""Finite Quest/G1 candidate trial. Default: real feedback, NO motor publisher.

The IK subprocess is forbidden from importing DDS. Default transport: arm_sdk,
with official locomotion already in operator-confirmed Regular standing mode.
Debug lowcmd is explicit legacy opt-in; internal Dex1 is debug-only. No replay
input, automatic rearming or mode changes. Each run requires a fresh gesture.
"""
import argparse
from contextlib import contextmanager
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
from arm_sdk import ArmSDKIO, STANDING
from hold_arms import DDSHoldIO, load_profile, run_hold
from quest_grippers import QuestGripperIO, gripper_settings, raw_triggers, trigger_pair
from record_cameras import check_camera_status, load_config

ROOT = Path(__file__).resolve().parents[2]
LIMITS = dict(active_s=180., input_timeout_s=1.0, feedback_timeout_s=.2,
              joint_limits='urdf', velocity_rad_s=1.5, acceleration_rad_s2=3.0,
              translation_scale=1., workspace_m=.50, ik_position_error_m=.03,
              tracking_error_rad=.50, workspace_shape='box_centered_on_start',
              orientation_mode='relative', side_grip_threshold=.5)
IK_COST = dict(translation=500., rotation=1., smooth=.1)


def load_quest_profile():
    # Quest-only trial override; preserve the original hold/probe profile.
    profile, config = load_profile(ROOT/'artifacts/unitree_g1/hold_shadow_profile.json', False)
    profile['hold_s'] = LIMITS['active_s']
    profile['max_initial_speed'] = .20
    tracking = (LIMITS['tracking_error_rad'],)*14
    profile['controller']['tracking_error'] = list(tracking)
    timeouts=dict(state_timeout=LIMITS['feedback_timeout_s'],
                  target_timeout=LIMITS['input_timeout_s'],max_step_seconds=.1)
    profile['controller'].update(timeouts)
    return profile, replace(config, tracking_error=tracking, **timeouts)


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, allow_nan=False)+'\n')
    temp.replace(path)


class PreparationTiming:
    def __init__(self, output, clock=time.monotonic):
        self.output, self.clock = output, clock
        self.started = clock()
        self.stages = []
        self.save(False)

    def save(self, ready):
        write_json(self.output/'preparation_timing.json', dict(
            started=self.started, elapsed_s=self.clock()-self.started,
            ready=ready, stages=self.stages))

    @contextmanager
    def stage(self, name):
        row = dict(name=name, started=self.clock(), status='running')
        self.stages.append(row)
        self.save(False)
        try:
            yield
        except BaseException:
            row['status'] = 'failed'
            raise
        else:
            row['status'] = 'complete'
        finally:
            row['elapsed_s'] = self.clock()-row['started']
            self.save(False)
            print(f'Preparing: {name} {row["status"]} ({row["elapsed_s"]:.2f}s)', flush=True)


def wait_for_preloaded_ik(output, nonce, process, stop, deadline, snapshot):
    while True:
        if (stop.is_set() or not process.is_alive()
                or (output/'worker_error.json').exists() or time.monotonic() > deadline):
            raise RuntimeError('IK initialization failed before routing VR input')
        if (output/'ik_ready.json').exists():
            ready = json.loads((output/'ik_ready.json').read_text())
            if ready.get('nonce') != nonce:
                raise ValueError('IK readiness session mismatch')
            return ready
        snapshot()
        time.sleep(.02)


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
    import casadi
    ik = robot.ik
    # Squared matrix distance is smooth even at zero rotation error. Upstream's
    # symbolic log3 produced NaN derivatives in axis-aligned rotation tests.
    # Half the Frobenius norm squared is 4*sin(theta/2)^2, locally theta^2.
    orientation_cost = casadi.Function('quest_orientation_cost',
        [ik.cq, ik.cTf_l, ik.cTf_r],
        [sum(.5 * casadi.sumsqr(ik.cdata.oMf[frame].rotation - target[:3, :3])
             for frame, target in ((ik.L_hand_id, ik.cTf_l), (ik.R_hand_id, ik.cTf_r)))])
    ik.rotation_cost = orientation_cost(ik.var_q, ik.param_tf_l, ik.param_tf_r)
    ik.opti.minimize(IK_COST['translation'] * ik.translational_cost +
                     IK_COST['rotation'] * ik.rotation_cost +
                     IK_COST['smooth'] * ik.smooth_cost)
    # Keep the upstream URDF bounds; no engagement-relative joint envelope.


def bounded_ik(robot, requested, references, previous, feedback, evidence):
    """Project position requests while preserving controller-relative orientation.

    Each arm backtracks independently, with at most nine fresh paired solves.
    Both arms are revalidated on every solve, including at the previous pose.
    Input timestamps are preserved through all attempts.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    targets = [t.copy() for t in requested]
    limited = []
    for target, reference in zip(targets, references):
        delta = target[:3, 3] - reference[:3, 3]
        # Model-frame box centered on engagement, symmetric on all three axes.
        radius = LIMITS['workspace_m']
        bounded = np.clip(delta, -radius, radius)
        limited.append(bool(np.any(delta != bounded)))
        target[:3, 3] = reference[:3, 3] + bounded
    evidence.update(requested_target=[t.tolist() for t in requested],
                    workspace_target=[t.tolist() for t in targets],
                    workspace_limited=limited, orientation_mode='relative', attempts=[])
    seed = np.asarray(vector(feedback['q']))
    ladder = (1., .5, .25, .125, 0.)
    levels = [0, 0]  # left, right
    for _ in range(9):
        fractions = [ladder[level] for level in levels]
        trial = [t.copy() for t in targets]
        for t, base, fraction in zip(trial, previous, fractions):
            t[:3, 3] = base[:3, 3] + fraction * (t[:3, 3]-base[:3, 3])
            # Shortest SO(3) arc; zero revalidates the entire previous pose.
            if fraction == 0.:
                t[:3, :3] = base[:3, :3]
            elif fraction != 1.:
                delta = Rotation.from_matrix(t[:3, :3] @ base[:3, :3].T)
                t[:3, :3] = Rotation.from_rotvec(fraction * delta.as_rotvec()).as_matrix() @ base[:3, :3]
        robot.ik.smooth_filter._data_queue = []
        candidate, _ = robot.ik.solve_ik(*trial, seed.copy(), np.zeros(14))
        q = vector(candidate)
        # Legacy scalar is diagnostic only: the minimum of the two fractions.
        attempt = dict(fraction=min(fractions), fractions=fractions, candidate_q=q)
        evidence['attempts'].append(attempt)
        if not robot.ik.opti.stats().get('success'):
            raise ValueError('IK failed')
        # A solver violating its constraints is a fault, not operator saturation.
        if (np.any(np.asarray(q) < robot.model.lowerPositionLimit)
                or np.any(np.asarray(q) > robot.model.upperPositionLimit)):
            raise ValueError('IK violates joint constraints')
        robot.q = np.asarray(q)
        achieved = robot.forward()
        if not all(np.isfinite(a).all() for a in achieved):
            raise ValueError('Nonfinite IK forward kinematics')
        errors = [float(np.linalg.norm(a[:3, 3]-b[:3, 3])) for a,b in zip(achieved, trial)]
        # Soft orientation objective; residuals are diagnostic, not a hard gate.
        rotation_errors = [float(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude())
                           for a,b in zip(achieved, trial)]
        angles = [math.acos(float(np.clip((np.trace(a[:3, :3] @ b[:3, :3].T)-1)/2, -1, 1)))
                  for a,b in zip(achieved, references)]
        attempt.update(ik_position_error_m=errors, ik_rotation_change_rad=angles,
                       ik_rotation_error_rad=rotation_errors)
        if max(errors) <= LIMITS['ik_position_error_m']:
            evidence.update(target=[t.tolist() for t in trial], candidate_q=q,
                            ik_position_error_m=errors, ik_rotation_change_rad=angles,
                            ik_rotation_error_rad=rotation_errors,
                            ik_limited=any(f < 1. for f in fractions),
                            ik_limited_sides=[f < 1. for f in fractions],
                            accepted_fraction=min(fractions), accepted_fractions=fractions)
            return q, errors, trial
        failing = [i for i, error in enumerate(errors)
                   if error > LIMITS['ik_position_error_m']]
        if any(levels[i] == len(ladder)-1 for i in failing):
            break
        for i in failing:
            levels[i] += 1
    raise ValueError('IK cannot revalidate bounded target')


class QuestSource:
    def __init__(self, output, nonce, clock=time.monotonic, gripper_io=None):
        self.output, self.nonce = output, nonce
        self.clock = clock
        self.initial = None
        self.last_input = None
        self.session = None
        self.gripper_io = gripper_io
        self.camera_status = None
        self.body_checker = None
        self.input_service = None

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
        executor.c = replace(c,
            velocity=(LIMITS['velocity_rad_s'],)*14,
            acceleration=(LIMITS['acceleration_rad_s2'],)*14)

    def update(self, executor, now):
        state = executor.observation
        write_json(self.output/'feedback.json', dict(nonce=self.nonce,
            time=state.received, q=list(state.q), command_q=list(executor.q),
            phase=executor.phase, initial_q=self.initial))
        try:
            if self.input_service is not None:
                from continuous_input import check_service
                check_service(self.input_service)
            if self.body_checker is not None:
                self.body_checker()
            if self.camera_status is not None:
                check_camera_status(self.camera_status, self.clock())
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
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
            executor._stop('quest_input: '+str(exc), True)


def worker(output, nonce, done, with_grippers=False, trial_limits=None):
    # The spawn context isolates VirtualRobot's NoHardwareImports guard from
    # the parent DDS process. All IK results are proposals, never DDS commands.
    if trial_limits is not None:
        LIMITS.update(trial_limits)
    import numpy as np
    from virtual_arms import VirtualRobot, RelativeClutch, read_available
    clutch = RelativeClutch(LIMITS['input_timeout_s'], LIMITS['translation_scale'],
                            workspace_m=None, grip_threshold=LIMITS['side_grip_threshold'])
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
            write_json(output/'ik_ready.json', {'nonce': nonce, 'time': time.monotonic()})
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
                            # Invalid/released input has already stopped the clutch.
                            # Do not dereference missing fields or invent open commands.
                            packet['triggers'] = raw_triggers(event) if state == 'following' else None
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


def open_grippers_before_teleoperation(args, output, stop):
    """Run a finite opening in its own DDS lifetime, before any arm publisher."""
    if getattr(args, 'control_mode', 'debug') != 'debug':
        raise ValueError('Internal Dex1 pre-opening cannot run with official locomotion')
    if stop.is_set():
        raise RuntimeError('Stopped before pre-opening; Quest will not start')
    destination=output/'preopen'
    settings=gripper_settings()
    print(f"Pre-open: both Dex1 to -5.4, velocity={settings['velocity']}, acceleration={settings['acceleration']}; "
          'finish on target/feedback, then unload. Arms/body passive. Quest starts only after success.',flush=True)
    with (output/'preopen.log').open('x') as log:
        process=subprocess.Popen([sys.executable,str(ROOT/'scripts/unitree_g1/gripper_cycle.py'),
            '--interface',args.interface,'--output',str(destination),'--execute',
            '--open-target','-5.4','--scene-confirmation',args.scene_confirmation],
            stdout=log,stderr=subprocess.STDOUT)
        interrupted=False
        while process.poll() is None:
            if stop.is_set() and not interrupted:
                process.terminate()  # SIGTERM requests the child's bounded unload.
                interrupted=True
            time.sleep(.05)
    if stop.is_set():
        raise RuntimeError('Stopped during pre-opening; Quest will not start')
    result=json.loads((destination/'summary.json').read_text())
    if process.returncode!=0 or result.get('error') is not None or result.get('opening_complete') is not True:
        raise RuntimeError('Gripper pre-opening failed; inspect preopen/summary.json')


def reconcile_external_preopen_cleanup(result, output, preopen_started, writer_open_attempted):
    """Carry earlier release evidence only if the Quest writer was never opened."""
    if writer_open_attempted or result.get('command_frames') != 0:
        return
    if result.get('external_gripper_cleanup_error') or result.get('unload_error'):
        return
    if not preopen_started:
        result['control_publish_not_started'] = True
        return
    try:
        preopen = json.loads((output/'preopen/summary.json').read_text())
    except (OSError, ValueError):
        return
    if not isinstance(preopen, dict):
        return
    if preopen.get('external_gripper_cleanup_error') or preopen.get('unload_error'):
        return
    if (preopen.get('control_publish_not_started') is True
            and preopen.get('command_frames') == 0):
        result['control_publish_not_started'] = True
    elif (preopen.get('arm_sdk_release_submitted') is True
            and preopen.get('external_gripper_zero_submitted') == [True, True]):
        result.update(arm_sdk_release_submitted=True,
                      external_gripper_zero_submitted=[True, True],
                      release_evidence_source='preopen/summary.json',
                      quest_writer_open_attempted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface', required=True)
    parser.add_argument('--cert', type=Path, required=True)
    parser.add_argument('--key', type=Path, required=True)
    parser.add_argument('--host-address', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--active-seconds', type=float, default=180.,
                        help='ACTIVE duration, at most 180s; takeover/release are additional')
    parser.add_argument('--control-mode', choices=('arm_sdk', 'debug'), default='arm_sdk',
                        help='arm_sdk preserves official leg control; debug is legacy full-body lowcmd')
    parser.add_argument('--grippers', action='store_true', help='Enable Dex1 trigger control using the selected backend')
    parser.add_argument('--gripper-backend', choices=('internal', 'external-dex1'), default='internal',
                        help='External Dex1 uses independent serial2dds topics with arm_sdk')
    parser.add_argument('--scene-confirmation', help='Current operator statement, not historical profile readiness')
    parser.add_argument('--wait-seconds', type=float, default=300.)
    parser.add_argument('--record', action='store_true', help='Record three RGB cameras and causal state/action logs')
    parser.add_argument('--task', help='Task instruction recorded with this episode')
    parser.add_argument('--outcome', choices=('unknown','success','failure'), default='unknown',
                        help='Operator task label for automatic BiCAT/G1 export; default unknown')
    parser.add_argument('--camera-host', default='unitree@192.168.123.164')
    parser.add_argument('--camera-ssh-key', type=Path, default=Path.home()/'.ssh/g1_camera_ed25519')
    parser.add_argument('--camera-config', type=Path, default=ROOT/'artifacts/unitree_g1/camera_capture_profile.json')
    parser.add_argument('--shared-input',type=Path,help='Persistent VR service owned by collect_episodes.py')
    parser.add_argument('--defer-export',action='store_true',help='Leave raw attempt pending operator s/f decision')
    args = parser.parse_args()
    if not math.isfinite(args.active_seconds) or not 0 < args.active_seconds <= 180:
        parser.error('--active-seconds must be in (0, 180]')
    LIMITS['active_s'] = args.active_seconds
    external_grippers = args.grippers and args.gripper_backend == 'external-dex1'
    if args.gripper_backend == 'external-dex1' and (not args.grippers or args.control_mode != 'arm_sdk'):
        parser.error('external-dex1 requires --grippers --control-mode arm_sdk')
    if args.control_mode == 'arm_sdk' and args.grippers and not external_grippers:
        parser.error('arm_sdk cannot control internal Dex1; --grippers is supported only in --control-mode debug')
    if args.record and (not (args.task or '').strip() or not args.grippers):
        parser.error('--record requires a nonempty --task and --grippers for BiCAT/G1 export')
    if not math.isfinite(args.wait_seconds) or not 0 < args.wait_seconds <= 600:
        parser.error('wait-seconds must be in (0, 600]')
    if args.execute and not (args.scene_confirmation or '').strip():
        parser.error('--execute requires current --scene-confirmation')
    for path in (args.cert, args.key):
        if not path.is_file():
            parser.error('TLS file missing')
    # Preserve candidate gains and hardware_approved=false; log Quest overrides.
    profile, config = load_quest_profile()
    from external_dex1 import ExternalDex1ArmSDKIO, external_settings, preopen_external
    grip_settings = external_settings() if external_grippers else gripper_settings() if args.grippers else None
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    nonce = uuid.uuid4().hex
    (output/'source').mkdir()
    source_hashes = {}
    for name in ('quest_bridge.py', 'arm_sdk.py', 'external_dex1.py', 'quest_grippers.py', 'gripper_cycle.py', 'gripper_probe.py', 'arm_execution.py', 'hold_arms.py', 'virtual_arms.py', 'televuer_input.py', 'record_cameras.py', 'convert_dataset.py', 'export_recording.py', 'export_kinematics.py', 'continuous_input.py', 'collect_episodes.py'):
        data = (ROOT/'scripts/unitree_g1'/name).read_bytes()
        (output/'source'/name).write_bytes(data)
        source_hashes[name] = hashlib.sha256(data).hexdigest()
    write_json(output/'session.json', dict(nonce=nonce, limits=LIMITS, ik_cost=IK_COST, profile=profile,
        control_mode=args.control_mode,
        command_topic='rt/arm_sdk' if args.control_mode == 'arm_sdk' else 'rt/lowcmd',
        standing_gate=STANDING if args.control_mode == 'arm_sdk' else None,
        grippers=grip_settings, preopen_required=bool(args.execute and args.grippers),
        gripper_backend=args.gripper_backend if args.grippers else None,
        recording=dict(enabled=args.record, task=args.task, fps=30,
                       shapes=({r: spec['shape'] for r,spec in load_config(args.camera_config)['cameras'].items()} if args.record else {}),
                       cameras=['head','left_wrist','right_wrist'],
                       action_semantics=('arm_sdk_requested_absolute_position' if args.control_mode == 'arm_sdk'
                                         else 'submitted_absolute_position'),
                       export_supported=bool(args.grippers and (args.control_mode == 'debug' or external_grippers)),
                       dataset_format='BiCAT-G1 / LeRobot v3.0', auto_export=not args.defer_export,
                       joint_dimensions=16, outcome=args.outcome),
        hardware_motion_enabled=args.execute, scene_confirmation=args.scene_confirmation,
        interface=args.interface, host_address=args.host_address,
        source_sha256=source_hashes,
        scope='candidate relative-pose trial; BiCAT/G1 export uses SDK requests and external service coordinates; no hardware acceptance'))
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    context = multiprocessing.get_context('spawn')
    done = context.Event()
    io = input_proc = ik_proc = input_log = None
    camera_proc = camera_log = None
    external_preopen_started = False
    writer_open_attempted = False
    result = dict(error='initialization_failed', command_frames=0, unload_submitted=False)
    source = QuestSource(output, nonce)
    source.input_service = args.shared_input
    timing = PreparationTiming(output)
    try:
        # IK only builds a model and proposals; it waits for feedback/input and
        # cannot publish DDS commands. Keep VR routing closed until all gates pass.
        ik_started = time.monotonic()
        ik_deadline = ik_started+60
        ik_proc = context.Process(target=worker, args=(output, nonce, done, args.grippers, dict(LIMITS)))
        ik_proc.start()
        with timing.stage('cameras'):
            if args.record:
                camera_log=(output/'cameras.log').open('x')
                camera_proc=subprocess.Popen([sys.executable,str(ROOT/'scripts/unitree_g1/record_cameras.py'),
                    '--output',str(output/'cameras'),'--host',args.camera_host,
                    '--ssh-key',str(args.camera_ssh_key),'--config',str(args.camera_config)],stdout=camera_log,stderr=subprocess.STDOUT)
                source.camera_status=output/'cameras/status.json'
                camera_deadline=time.monotonic()+30
                while True:
                    if stop.is_set() or camera_proc.poll() is not None or time.monotonic()>camera_deadline:
                        raise RuntimeError('Camera recording failed before engagement; inspect cameras.log')
                    try:
                        check_camera_status(source.camera_status,time.monotonic())
                        break
                    except (OSError,ValueError,KeyError,TypeError):time.sleep(.05)
        if external_grippers:
            with timing.stage('robot_initialization_and_preflight'):
                if stop.is_set():
                    raise RuntimeError('Stopped before Quest initialization')
                io = ExternalDex1ArmSDKIO(args.interface, args.execute, profile, config)
                io.preflight()
        with timing.stage('gripper_preopen'):
            if args.execute and args.grippers:
                if external_grippers:
                    external_preopen_started = True
                    preopen_external(args.interface, output/'preopen', stop, prepared_io=io)
                else:
                    open_grippers_before_teleoperation(args,output,stop)
        with timing.stage('robot_readiness_recheck' if external_grippers else 'robot_initialization_and_preflight'):
            if stop.is_set():
                raise RuntimeError('Stopped before Quest initialization')
            io = (io if external_grippers
                  else ArmSDKIO(args.interface, args.execute, profile, config) if args.control_mode == 'arm_sdk'
                  else QuestGripperIO(args.interface,args.execute,profile,grip_settings) if args.grippers
                  else DDSHoldIO(args.interface, args.execute, profile))
            if args.control_mode == 'arm_sdk':
                source.body_checker = io.check_active_body
            source.gripper_io=io if args.grippers else None
            if args.grippers:io.status_path=output/'gripper_status.json'
            io.record_observation=args.record
            if external_grippers:
                io.check()
                io.check_readiness()
            else:
                io.preflight()
            source.snapshot(io, 'WAITING')
        with timing.stage('input_service'):
            if args.shared_input:
                from continuous_input import check_service
                check_service(args.shared_input)
            else:
                input_log = (output/'input.log').open('x')
                input_proc = subprocess.Popen([sys.executable, str(ROOT/'scripts/unitree_g1/televuer_input.py'),
                    '--cert', str(args.cert.resolve()), '--key', str(args.key.resolve()),
                    '--output', str(output/'input'), '--host-address', args.host_address,
                    '--timeout', str(LIMITS['input_timeout_s'])]+
                    (['--gripper-status',str(output/'gripper_status.json')] if args.grippers else []),
                    stdout=input_log, stderr=subprocess.STDOUT)
        with timing.stage('ik_remaining_wait'):
            ik_ready = wait_for_preloaded_ik(output, nonce, ik_proc, stop, ik_deadline,
                                           lambda: source.snapshot(io, 'WAITING'))
        timing.stages.append(dict(name='ik_parallel_load', started=ik_started,
                                  elapsed_s=ik_ready['time']-ik_started, status='complete',
                                  overlaps_other_stages=True))
        with timing.stage('input_routing'):
            if args.shared_input:
                from continuous_input import set_route
                set_route(args.shared_input,output/'input',nonce)
        timing.save(True)
        if args.shared_input:
            write_json(output/'collection_ready.json',dict(
                nonce=nonce,time=time.monotonic(),preparation_s=time.monotonic()-timing.started))
        print(f'Quest: https://{args.host_address}:8012/?ws=wss://{args.host_address}:8012', flush=True)
        print('MODE: '+('LIVE CANDIDATE' if args.execute else 'SHADOW: NO MOTOR PUBLISHER'), flush=True)
        print(f'Transport: {args.control_mode}; '+
              ('official locomotion owns legs; waist holds engagement pose; '
               + ('external Dex1 topics enabled; ' if external_grippers else 'no Dex1 commands; ') +
               'exit returns upper-body authority, not whole-body unload.' if args.control_mode == 'arm_sdk'
               else 'debug lowcmd: legs/body passive.'), flush=True)
        print('Release both grips, then hold both. Keep still for 2s ramp; small translations and rotations. '
              'Position and controller-relative orientation control (soft IK orientation objective). '
              f'Workspace: forward/lateral/vertical +/-{LIMITS["workspace_m"]*100:g} cm from engagement; URDF joint limits. '
              'Release either grip to end this run. No automatic rearm.', flush=True)
        if args.grippers:
            print('Dex1: after ramp, control starts with fresh ACTIVE input; '
                  'INDEX trigger 0..100% maps full open..near closed. '
                  'Trigger position control; excessive closing error stops that gripper. '
                  'Request a more-open position to release; other faults unload all. '
                  'Side grips remain the stop/clutch. '
                  'Grippers unload with arms; no object holding on exit.',flush=True)
        deadline = time.monotonic()+args.wait_seconds
        while not stop.is_set():
            now = time.monotonic()
            if now > deadline:
                raise TimeoutError('No fresh Quest engagement before wait deadline')
            io.check()
            if args.record:
                check_camera_status(source.camera_status,now)
            if args.shared_input:
                from continuous_input import check_service
                check_service(args.shared_input)
            if (input_proc is not None and input_proc.poll() is not None) or not ik_proc.is_alive():
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
                writer_open_attempted = True
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
        if args.shared_input:
            try:
                from continuous_input import set_route
                set_route(args.shared_input)
            except Exception as exc:
                result['input_route_cleanup_error'] = str(exc)
                result['error'] = result.get('error') or str(exc)
        if camera_proc:
            if camera_proc.poll() is None:
                camera_proc.terminate()
                try:camera_proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    camera_proc.kill();camera_proc.wait()
            result['camera_recorder_returncode']=camera_proc.returncode
            if camera_proc.returncode!=0:
                result['error']=result.get('error') or 'Camera recording did not finish cleanly'
        if camera_log:camera_log.close()
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
            if external_grippers:
                result['external_gripper_zero_submitted'] = list(io.external.zero_submitted)
                result['external_gripper_cleanup_error'] = io.external_cleanup_error
                result['physical_gripper_release_verified'] = False
                if io.external_cleanup_error:
                    result['error'] = result.get('error') or io.external_cleanup_error
        if args.control_mode == 'arm_sdk':
            result['arm_sdk_release_submitted'] = bool(args.execute and result.get('unload_submitted'))
            result['unload_submitted'] = False
            result['physical_handover_verified'] = False
            result['last_arm_sdk_request'] = io.last_sdk if io else None
        result['control_mode'] = args.control_mode
        if external_grippers:
            reconcile_external_preopen_cleanup(
                result, output, external_preopen_started, writer_open_attempted)
        result.update(hardware_motion_enabled=args.execute, physical_unload_verified=False)
        if not args.execute:
            result['unload_submitted'] = False
        write_json(output/'summary.json', result)
        print(json.dumps(result, indent=2), flush=True)
    export_failed = False
    if args.record and not args.defer_export:
        from export_recording import export_recording
        print('Recording closed. Exporting BiCAT/G1 LeRobot dataset offline...',flush=True)
        export_status = export_recording(output,outcome=args.outcome,allow_shadow=not args.execute)
        export_failed = not export_status['exported']
        print(json.dumps(export_status,indent=2),flush=True)
    return 1 if result['error'] or export_failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
