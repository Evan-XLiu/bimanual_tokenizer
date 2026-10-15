#!/usr/bin/env python3
"""Input-only diagnostics for the pinned official TeleVuer; no robot or IK imports."""
import argparse
import asyncio
import dataclasses
import hashlib
import importlib.abc
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def gripper_status_lines(path,now):
    """Display-only reader. Missing/stale data never implies a maintained grasp."""
    try:
        if path is not None and path.is_dir():
            from continuous_input import routed_gripper_status
            path = routed_gripper_status(path)
            if path is None:
                return ['EPISODE FINISHED / PREPARING', 'Terminal: n = next, s = save, f = discard',
                        'Release both side grips before the next episode']
        status=json.loads(path.read_text())
        if not 0<=now-status['time']<=.5:
            return ['GRIPPER STATUS STALE - holding unknown']
        states=status['states']
        if len(states)!=2 or not all(isinstance(s,str) for s in states):raise ValueError('states')
        mode='LIVE' if status['hardware_motion_enabled'] else 'SHADOW'
        lines=[mode+' | L: '+states[0]+' | R: '+states[1]]
        remaining=status.get('remaining_s')
        if status['phase']=='ACTIVE' and remaining is not None:
            lines.append(f'Unload in {remaining:.1f}s'+(' - PLACE OBJECT DOWN' if remaining<=10 else ''))
        else:lines.append('Phase: '+status['phase']+' - no holding guarantee')
        lines.append('Release INDEX trigger: open | Side grip/timeout: unload all')
        lines.append('CLOSING_BLOCKED: ease INDEX trigger to open; grasp not verified')
        if status.get('fault'):lines.append('FAULT: '+str(status['fault']['reason']))
        return lines
    except (OSError,ValueError,KeyError,TypeError):
        return ['GRIPPER STATUS UNAVAILABLE - holding unknown']


def gripper_hud_image(path,now):
    from PIL import Image,ImageDraw,ImageFont
    canvas=Image.new('RGBA',(1024,768),(0,0,0,0))
    draw=ImageDraw.Draw(canvas)
    draw.rectangle((12,584,1012,756),fill=(10,15,20,220))
    try:font=ImageFont.truetype('DejaVuSans.ttf',20)
    except OSError:font=ImageFont.load_default()
    for i,line in enumerate(gripper_status_lines(path,now)):
        draw.text((24,596+i*29),line,font=font,fill=(255,225,150,255))
    return canvas


class NoRobotImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'unitree_sdk2py', 'cyclonedds', 'teleop', 'robot_control'}:
            raise ImportError(f'Robot module forbidden in input diagnostic: {fullname}')


def matrix_valid(value):
    try:
        import numpy as np
        m = np.asarray(value, dtype=float).reshape(4, 4, order='F')
        return bool(np.isfinite(m).all() and np.allclose(m[3], [0, 0, 0, 1], atol=1e-3)
                    and np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=0.05)
                    and abs(np.linalg.det(m[:3, :3]) - 1) < 0.05)
    except (ValueError, TypeError, OverflowError):
        return False


def valid_event(kind, value):
    if not isinstance(value, dict):
        return False
    if kind == 'CAMERA_MOVE':
        return isinstance(value.get('camera'), dict) and matrix_valid(value['camera'].get('matrix'))
    if kind != 'CONTROLLER_MOVE':
        return False
    for side in ('left', 'right'):
        if not matrix_valid(value.get(side)):
            return False
        state = value.get(side + 'State')
        if not isinstance(state, dict):
            return False
        for key in ('triggerValue', 'squeezeValue'):
            v = state.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1:
                return False
        stick = state.get('thumbstickValue', [0, 0])
        if not isinstance(stick, (list, tuple)) or len(stick) != 2:
            return False
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) > 1 for v in stick):
            return False
    return True


def serialize_event(record):
    """Unsupported payloads remain visible as invalid input, never button zeroes."""
    def numpy_value(value):
        import numpy as np
        if isinstance(value, (np.ndarray, np.generic)):
            return value.tolist()
        raise TypeError(f'Unsupported event value: {type(value).__name__}')

    try:
        return json.dumps(record, default=numpy_value, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        invalid = dict(record, valid=False, raw=repr(record.get('raw'))[:4096],
                       input_error=f'serialization: {type(exc).__name__}')
        invalid.pop('official_head_yaw', None)
        return json.dumps(invalid, allow_nan=False)


class Monitor:
    """Receive freshness only; never claim browser tracking validity or arm a robot."""
    def __init__(self, timeout):
        self.timeout = timeout
        self.last = {}
        self.session = None
        self.epoch = 0
        self.state = 'waiting'
        self.counts = {}
        self.invalid = 0
        self.gaps = []

    def accept(self, event):
        kind, now = event['kind'], event['host_receive_monotonic_s']
        if event['session'] != self.session:
            self.session = event['session']
            self.last.clear()
            self.epoch += 1
        self.counts[kind] = self.counts.get(kind, 0) + 1
        if not event['valid']:
            self.invalid += 1
            self.last.pop(kind, None)
        else:
            if kind == 'CONTROLLER_MOVE' and kind in self.last:
                self.gaps.append(now - self.last[kind])
            self.last[kind] = now

    def status(self, now):
        fresh = all(0 <= now - self.last.get(k, -1e20) <= self.timeout
                    for k in ('CAMERA_MOVE', 'CONTROLLER_MOVE'))
        self.state = 'receiving' if fresh else ('stale' if self.counts else 'waiting')
        return {'state': self.state, 'connection_epoch': self.epoch,
                'counts': self.counts, 'invalid_events': self.invalid,
                'receive_fresh': fresh, 'tracking_validity': 'not_exposed_by_upstream',
                'motion_enabled': False}


def create_tv(args, events, dropped):
    sys.meta_path.insert(0, NoRobotImports())
    from televuer import TeleVuer, TeleVuerWrapper

    class InputTeleVuer(TeleVuer):
        async def main_pass_through(self,session):
            path=getattr(args,'gripper_status',None)
            if path is None:
                return await super().main_pass_through(session)
            from concurrent.futures import ThreadPoolExecutor
            from vuer.schemas import ImageBackground
            renderer=ThreadPoolExecutor(max_workers=1,thread_name_prefix='gripper-hud')
            async def display():
                while True:
                    canvas=await asyncio.get_running_loop().run_in_executor(
                        renderer,gripper_hud_image,path,time.monotonic())
                    session.upsert(ImageBackground(canvas,aspect=4/3,height=1,distanceToCamera=1,
                        format='png',key='gripper-status',interpolate=True),to='bgChildren')
                    await asyncio.sleep(.2)
            # Rendering and file reads are independent of controller event callbacks.
            tasks=[asyncio.create_task(display()),
                   asyncio.create_task(super().main_pass_through(session))]
            try:await asyncio.gather(*tasks)
            finally:
                for task in tasks:task.cancel()
                await asyncio.gather(*tasks,return_exceptions=True)
                renderer.shutdown(wait=False,cancel_futures=True)

        def _vuer_run(self):
            # Vuer defaults to serving cwd through /static. Use an empty directory.
            self.vuer.static_root = str(args.output / 'empty-static')
            super()._vuer_run()

        async def capture(self, kind, event, session):
            now = time.monotonic()
            ok = valid_event(kind, event.value)
            record = {'kind': kind, 'host_receive_monotonic_s': now,
                      'session': str(session.CURRENT_WS_ID), 'valid': ok, 'raw': event.value}
            try:
                if ok:
                    if kind == 'CAMERA_MOVE':
                        await super().on_cam_move(event, session)
                    else:
                        await super().on_controller_move(event, session)
                    # Snapshot official transformation only with a fresh same-connection head.
                    sid = record['session']
                    if kind == 'CAMERA_MOVE':
                        self._head_received = (sid, now)
                    elif getattr(self, '_head_received', (None, 0))[0] == sid and now - self._head_received[1] <= args.timeout:
                        wrapper = TeleVuerWrapper.__new__(TeleVuerWrapper)
                        wrapper.tvuer = self
                        wrapper.use_hand_tracking = False
                        wrapper.arm_reference_mode = 'head_yaw'
                        record['official_head_yaw'] = dataclasses.asdict(wrapper.get_tele_data())
            except Exception as exc:
                # Upstream conversion is diagnostic; invalid input must still
                # reach the clutch so ACTIVE stops rather than using stale input.
                record['valid'] = False
                record['input_error'] = f'upstream: {type(exc).__name__}: {exc}'
                record.pop('official_head_yaw', None)
            if kind == 'CAMERA_MOVE' and not record['valid']:
                self._head_received = (None, 0)
            # Serialize before enqueueing; binary/malformed input is an invalid
            # event for the clutch, not an exception escaping the callback.
            payload = serialize_event(record)
            try:
                events.put_nowait(payload)
            except queue.Full:
                with dropped.get_lock():
                    dropped.value += 1

        async def on_cam_move(self, event, session, fps=60):
            await self.capture('CAMERA_MOVE', event, session)

        async def on_controller_move(self, event, session, fps=60):
            await self.capture('CONTROLLER_MOVE', event, session)

    return InputTeleVuer(use_hand_tracking=False, binocular=False, img_shape=(480, 640),
                        display_mode='pass-through', zmq=False, webrtc=False,
                        cert_file=str(args.cert), key_file=str(args.key))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cert', type=Path, required=True)
    parser.add_argument('--key', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New session directory outside tracked data')
    parser.add_argument('--host-address', default='127.0.0.1', help='Address printed for headset; upstream binds 0.0.0.0:8012')
    parser.add_argument('--timeout', type=float, default=1.0)
    parser.add_argument('--duration', type=float, default=0, help='Wall seconds from server startup; 0 until Ctrl-C')
    parser.add_argument('--device-note', default='Quest 3S; OS/browser versions pending operator')
    parser.add_argument('--gripper-status',type=Path,help='Read-only status file for the Quest HUD')
    parser.add_argument('--routed',action='store_true',help='Persistent input service; route events to pending attempts')
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0 or not math.isfinite(args.duration) or args.duration < 0:
        parser.error('timeout must be positive and duration nonnegative, both finite')
    args.cert, args.key, args.output = args.cert.resolve(), args.key.resolve(), args.output.resolve()
    if not args.cert.is_file() or not args.key.is_file():
        parser.error('TLS certificate/key missing')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'empty-static').mkdir()
    metadata = {'schema': 'televuer-input-v1', 'device_note': args.device_note,
                'baseline': json.loads((ROOT / 'artifacts/unitree_g1/baseline.lock.json').read_text()),
                'entry_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'mode': 'pass-through', 'motion_enabled': False, 'timeout_s': args.timeout,
                'clock': 'host monotonic receipt; not device sample time or end-to-end latency',
                'limitations': ['upstream does not expose reliable tracking-loss/reset/session visibility flags',
                                'same websocket XR re-entry requires operator annotation',
                                'fresh matrices do not prove physical tracking',
                                'official_head_yaw triggerValue is upstream 10-10*raw, not raw 0..1']}
    (args.output / 'session.json').write_text(json.dumps(metadata, indent=2) + '\n')
    routed = None
    if args.routed:
        from continuous_input import RoutedEvents, write_json
        write_json(args.output/'route.json',{})
        routed = RoutedEvents(args.output)
    events, dropped = mp.Queue(maxsize=4096), mp.Value('L', 0)
    monitor = Monitor(args.timeout)
    tv = None
    reason = 'error'
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    start = time.monotonic()
    try:
        tv = create_tv(args, events, dropped)
        print(f'Quest: https://{args.host_address}:8012/?ws=wss://{args.host_address}:8012', flush=True)
        print(f'Input only. Select pass-through. Logs: {args.output}', flush=True)
        last_print = 0
        from contextlib import nullcontext
        with (nullcontext(routed) if routed else (args.output / 'events.jsonl').open('x', buffering=1)) as log:
            while not args.duration or time.monotonic() - start < args.duration:
                if routed:
                    routed.sync()
                if not tv.process.is_alive():
                    raise RuntimeError('TeleVuer server exited; check TLS and port 8012')
                try:
                    payload = events.get(timeout=0.1)
                    event = json.loads(payload)
                    monitor.accept(event)
                    log.write(payload if routed else payload + '\n')
                except queue.Empty:
                    pass
                now = time.monotonic()
                if now - last_print >= 1:
                    status = dict(monitor.status(now), time=now, queue_dropped=dropped.value)
                    temp = args.output / 'status.tmp'
                    temp.write_text(json.dumps(status, indent=2) + '\n')
                    temp.replace(args.output / 'status.json')
                    print(json.dumps(status), flush=True)
                    last_print = now
            reason = 'duration_elapsed'
    except KeyboardInterrupt:
        reason = 'operator_stop'
    finally:
        if routed:
            routed.close()
        if tv is not None:
            tv.close()
        summary = dict(monitor.status(time.monotonic()), end_reason=reason,
                       duration_s=time.monotonic()-start, queue_dropped=dropped.value,
                       field_acceptance='pending_operator_review')
        if monitor.gaps:
            gaps = sorted(monitor.gaps)
            summary['controller_receive_gap_s'] = {'max': max(gaps), 'p95': gaps[int((len(gaps)-1)*0.95)]}
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        events.close()
        print(json.dumps(summary), flush=True)
    return 0 if reason != 'error' else 1


if __name__ == '__main__':
    sys.exit(main())
