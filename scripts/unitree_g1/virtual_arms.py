#!/usr/bin/env python3
"""Quest relative-clutch control of a kinematic G1 model. Never publishes DDS."""
import argparse
import importlib.abc
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

from televuer_input import ROOT, valid_event


class NoHardwareImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split('.')[0] in {'unitree_sdk2py', 'cyclonedds'}
                or fullname.startswith(('teleop.robot_control.robot_arm.', 'teleop.robot_control.robot_hand'))
                or fullname in {'teleop.robot_control.robot_arm', 'teleop.teleop_hand_and_arm'}):
            raise ImportError(f'Hardware module forbidden in virtual arms: {fullname}')


# WebXR right/up/back -> robot forward/left/up. This is a fixed world frame,
# deliberately independent of head yaw. Relative rotations need no grip offset.
XR_TO_ROBOT = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])


def poses(raw):
    result = []
    for side in ('left', 'right'):
        source = np.asarray(raw[side]).reshape(4, 4, order='F')
        pose = np.eye(4)
        pose[:3, :3] = XR_TO_ROBOT @ source[:3, :3]
        pose[:3, 3] = XR_TO_ROBOT @ source[:3, 3]
        result.append(pose)
    return result


class RelativeClutch:
    """Both grips engage both arms; faults require a fresh release before rearming."""
    def __init__(self, timeout=0.25, scale=1.0, workspace_m=0.35):
        self.timeout, self.scale = timeout, scale
        self.workspace_m = workspace_m
        self.session = None
        self.last_time = None
        self.previous = None
        self.reference = None
        self.robot_reference = None
        self.released = False
        self.state = 'release_to_start'

    def stop(self, reason):
        self.reference = None
        self.released = False
        self.previous = None
        self.state = reason

    def tick(self, now):
        if self.last_time is not None and now - self.last_time > self.timeout:
            self.stop('stale')

    def accept(self, event, now, robot_poses):
        if event['session'] != self.session:
            self.stop('new_connection')
            self.session = event['session']
            self.last_time = None
        self.tick(now)
        if not event.get('valid') or not valid_event(event['kind'], event.get('raw')):
            self.stop('invalid_input')
            return None
        if event['kind'] != 'CONTROLLER_MOVE':
            return None
        received = event['host_receive_monotonic_s']
        if not 0 <= now - received <= self.timeout:
            self.stop('stale')
            return None
        if self.last_time is not None and received <= self.last_time:
            self.stop('nonmonotonic_input')
            return None
        self.last_time = received
        raw = event['raw']
        grip = [raw[s + 'State'].get('squeezeValue', 0) for s in ('left', 'right')]
        current = poses(raw)
        if max(grip) < 0.2:
            self.reference = None
            self.released = True
            self.previous = current
            self.state = 'ready'
            return None
        if min(grip) < 0.7:
            self.reference = None
            self.state = 'hold'
            return None
        if not self.released:
            return None
        if self.reference is None:
            self.reference = current
            self.robot_reference = [p.copy() for p in robot_poses]
            self.previous = current
            self.state = 'following'
            return None  # Engagement never changes the pose.
        for old, new in zip(self.previous, current):
            angle = math.acos(float(np.clip((np.trace(new[:3, :3] @ old[:3, :3].T)-1)/2, -1, 1)))
            if np.linalg.norm(new[:3, 3]-old[:3, 3]) > 0.12 or angle > 0.8:
                self.stop('pose_jump')
                return None
        self.previous = current
        targets = []
        for source, reference, robot in zip(current, self.reference, self.robot_reference):
            target = robot.copy()
            delta = self.scale * (source[:3, 3] - reference[:3, 3])
            if self.workspace_m is not None and np.linalg.norm(delta) > self.workspace_m:
                self.stop('relative_workspace')
                return None
            target[:3, 3] += delta
            target[:3, :3] = source[:3, :3] @ reference[:3, :3].T @ robot[:3, :3]
            targets.append(target)
        self.state = 'following'
        return targets


class VirtualRobot:
    def __init__(self, directory, visualize):
        sys.meta_path.insert(0, NoHardwareImports())
        xr = ROOT / '.local/unitree_g1/xr_teleoperate'
        sys.path.insert(0, str(xr))
        # Keep upstream's relative assets and pickle cache in a fresh private dir.
        (directory / 'assets').symlink_to(xr / 'assets', target_is_directory=True)
        work = directory / 'teleop'
        work.mkdir()
        before = Path.cwd()
        try:
            os.chdir(work)
            from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
            self.ik = G1_29_ArmIK(Visualization=False)
            # Upstream URDF uses relative mesh paths; the viewer runs after cwd is restored.
            for geometry in (self.ik.reduced_robot.visual_model, self.ik.reduced_robot.collision_model):
                for obj in geometry.geometryObjects:
                    if obj.meshPath:
                        obj.meshPath = str(Path(obj.meshPath).resolve())
        finally:
            os.chdir(before)
        import pinocchio as pin
        self.pin = pin
        self.model = self.ik.reduced_robot.model
        self.ik.reduced_robot.data = self.model.createData()
        self.data = self.model.createData()
        self.q = np.zeros(14)
        self.q[[3, 10]] = 0.6  # Virtual-only bent elbows; not a real startup pose.
        self.viewer = None
        self.vis = None
        if visualize:
            import meshcat
            from pinocchio.visualize import MeshcatVisualizer
            self.viewer = meshcat.Visualizer()
            self.vis = MeshcatVisualizer(self.model, self.ik.reduced_robot.collision_model,
                                        self.ik.reduced_robot.visual_model)
            try:
                self.vis.initViewer(viewer=self.viewer, open=False)
                self.vis.loadViewerModel('g1')
                self.vis.display(self.q)
                self.vis.setCameraZoom(2.0)
            except BaseException:
                self.close()
                raise
            print(f'Virtual G1: {self.viewer.url()}', flush=True)

    def forward(self):
        self.pin.framesForwardKinematics(self.model, self.data, self.q)
        return [self.data.oMf[i].homogeneous.copy() for i in (self.ik.L_hand_id, self.ik.R_hand_id)]

    def solve(self, targets, dt):
        # Reset upstream history: rejected solutions must never contaminate later steps.
        self.ik.smooth_filter._data_queue = []
        candidate, _ = self.ik.solve_ik(*targets, self.q.copy(), np.zeros(14))
        if not self.ik.opti.stats().get('success') or not np.isfinite(candidate).all():
            raise ValueError('ik_failed')
        if np.any(candidate < self.model.lowerPositionLimit-1e-6) or np.any(candidate > self.model.upperPositionLimit+1e-6):
            raise ValueError('joint_limit')
        self.pin.framesForwardKinematics(self.model, self.data, candidate)
        errors = [float(np.linalg.norm(self.data.oMf[i].translation-t[:3, 3]))
                  for i, t in zip((self.ik.L_hand_id, self.ik.R_hand_id), targets)]
        if max(errors) > 0.08:
            raise ValueError('unreachable_target')
        # Display-only velocity cap; not a validated hardware control limit.
        self.q += np.clip(candidate-self.q, -1.5*min(dt, 0.05), 1.5*min(dt, 0.05))
        if self.vis:
            self.vis.display(self.q)
        return errors

    def close(self):
        if self.viewer is not None:
            # Pinned meshcat's Visualizer.close calls a missing ViewerWindow.close.
            window = self.viewer.window
            window.zmq_socket.close(linger=0)
            if window.server_proc is not None:
                window.server_proc.terminate()
                try:
                    window.server_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    window.server_proc.kill()
                    window.server_proc.wait()
            self.viewer = None


def read_available(stream):
    """Consume complete JSONL records, retaining an in-progress last line."""
    records = []
    while True:
        position = stream.tell()
        line = stream.readline()
        if not line.endswith('\n'):
            stream.seek(position)
            return records
        records.append(json.loads(line))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New session directory')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--cert', type=Path, help='Launch live Quest input with this TLS certificate')
    source.add_argument('--replay', type=Path, help='Replay recorded TeleVuer events with their original timing')
    parser.add_argument('--key', type=Path)
    parser.add_argument('--host-address', default='127.0.0.1')
    parser.add_argument('--headless', action='store_true', help='IK/logging only, no model viewer')
    parser.add_argument('--duration', type=float, default=0)
    parser.add_argument('--timeout', type=float, default=0.25)
    parser.add_argument('--scale', type=float, default=1.0)
    args = parser.parse_args()
    for name in ('timeout', 'scale', 'duration'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name != 'duration' and value == 0):
            parser.error(f'invalid {name}')
    if args.cert and (not args.key or not args.key.is_file() or not args.cert.is_file()):
        parser.error('Live input requires existing --cert and --key')
    if args.replay and not args.replay.is_file():
        parser.error('Replay file missing')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = {'mode': 'kinematic_virtual_only', 'hardware_motion_enabled': False,
                'mapping': 'fixed WebXR world: -Z forward, -X left, +Y up; relative clutch',
                'timeout_s': args.timeout, 'scale': args.scale, 'replay': str(args.replay) if args.replay else None,
                'limitations': ['no dynamics or collision acceptance', 'Dex3 visual model, not confirmed real gripper',
                                'no reliable upstream tracking/reset flag; slow drift may not be detected']}
    (output / 'session.json').write_text(json.dumps(metadata, indent=2)+'\n')
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    robot, input_proc, stream, input_log = None, None, None, None
    clutch = RelativeClutch(args.timeout, args.scale)
    solved, failures, reason = 0, 0, 'error'
    try:
        with tempfile.TemporaryDirectory(prefix='g1-virtual-') as tmp:
            robot = VirtualRobot(Path(tmp), not args.headless)
            if robot.viewer is not None:
                (output / 'viewer.json').write_text(json.dumps({'url': robot.viewer.url()})+'\n')
            if args.cert:
                input_log = (output / 'input.log').open('w')
                input_proc = subprocess.Popen([sys.executable, str(ROOT / 'scripts/unitree_g1/televuer_input.py'),
                    '--cert', str(args.cert.resolve()), '--key', str(args.key.resolve()),
                    '--output', str(output / 'input'), '--host-address', args.host_address,
                    '--timeout', str(args.timeout)], stdout=input_log, stderr=subprocess.STDOUT)
                print(f'Quest: https://{args.host_address}:8012/?ws=wss://{args.host_address}:8012', flush=True)
            else:
                stream = args.replay.open()
            print('Release BOTH side grips, then hold BOTH to follow. Release either to hold. Ctrl-C exits.', flush=True)
            start = time.monotonic()
            last_solve, last_status = start, 0
            replay_origin, pending = None, None
            with (output / 'joints.jsonl').open('w', buffering=1) as log:
                while not args.duration or time.monotonic()-start < args.duration:
                    now = time.monotonic()
                    clutch.tick(now)
                    if input_proc:
                        if input_proc.poll() is not None:
                            raise RuntimeError('Input process exited; see input.log')
                        path = output / 'input/events.jsonl'
                        if stream is None and path.exists():
                            stream = path.open()
                        events = read_available(stream) if stream else []
                    else:
                        if pending is None:
                            line = stream.readline()
                            if not line:
                                reason = 'replay_complete'
                                break
                            pending = json.loads(line)
                            stamp = pending['host_receive_monotonic_s']
                            if replay_origin is None:
                                replay_origin = stamp
                            pending['host_receive_monotonic_s'] = start + stamp-replay_origin
                        events = []
                        if pending['host_receive_monotonic_s'] <= now:
                            events, pending = [pending], None
                    for event in events:
                        targets = clutch.accept(event, time.monotonic(), robot.forward())
                        if targets is not None:
                            current = time.monotonic()
                            try:
                                errors = robot.solve(targets, max(0, current-last_solve))
                                solved += 1
                                log.write(json.dumps({'time': current, 'q': robot.q.tolist(),
                                    'target': [p.tolist() for p in targets], 'ik_position_error_m': errors})+'\n')
                            except (ValueError, RuntimeError) as exc:
                                failures += 1
                                clutch.stop(str(exc))
                            last_solve = current
                    if now-last_status >= 0.5:
                        status = {'state': clutch.state, 'solved_frames': solved, 'ik_failures': failures,
                                  'q': robot.q.tolist(), 'hardware_motion_enabled': False}
                        temp = output / 'status.tmp'
                        temp.write_text(json.dumps(status, indent=2)+'\n')
                        temp.replace(output / 'status.json')
                        if status['state'] != getattr(clutch, 'printed_state', None):
                            print(json.dumps(status), flush=True)
                            clutch.printed_state = status['state']
                        last_status = now
                    time.sleep(0.002)
                else:
                    reason = 'duration_elapsed'
    except KeyboardInterrupt:
        reason = 'operator_stop'
    finally:
        if input_proc is not None and input_proc.poll() is None:
            input_proc.terminate()
            try:
                input_proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                input_proc.kill()
                input_proc.wait()
        if robot:
            robot.close()
        if stream:
            stream.close()
        if input_log:
            input_log.close()
        summary = {'end_reason': reason, 'solved_frames': solved, 'ik_failures': failures,
                   'hardware_motion_enabled': False, 'physical_axis_acceptance': 'pending_operator'}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
