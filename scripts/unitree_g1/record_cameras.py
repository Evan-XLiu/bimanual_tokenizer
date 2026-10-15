#!/usr/bin/env python3
"""Read-only three-camera SSH recorder; stdout on --serve is a framed protocol."""
import argparse
import json
import math
from pathlib import Path
import shlex
import signal
import struct
import subprocess
import sys
import threading
import time

CAMERAS = {
    'head': ('*USB2.0_Camera_RGB*video-index0', 'MJPG'),
    'left_wrist': ('*261523071841-video-index4', 'YUYV'),
    'right_wrist': ('*261523072307-video-index4', 'YUYV'),
}
DEFAULT_CONFIG = (Path(__file__).resolve().parents[2] / 'artifacts/unitree_g1/camera_capture_profile.json'
                  if '__file__' in globals() else None)


def validate_config(config):
    if config.get('schema_version') != 1 or set(config.get('cameras', {})) != set(CAMERAS):
        raise ValueError('Camera config must define head, left_wrist and right_wrist')
    for role, spec in config['cameras'].items():
        allowed = ([480, 640, 3], [480, 1280, 3]) if role == 'head' else ([480, 640, 3],)
        if spec.get('shape') not in allowed or spec.get('rotation_degrees') != 0:
            raise ValueError(f'{role}: unsupported RGB dimensions or rotation')
        if not isinstance(spec.get('fps'), (int, float)) or not 0 < spec['fps'] <= 30:
            raise ValueError(f'{role}: invalid capture FPS')
        if spec.get('backend') == 'v4l2':
            if not spec.get('device', '').startswith('/dev/v4l/') or spec.get('fourcc') not in ('MJPG', 'YUYV'):
                raise ValueError(f'{role}: use stable V4L device path and explicit format')
        elif spec.get('backend') == 'ros_compressed':
            if not spec.get('topic', '').startswith('/camera/'):
                raise ValueError(f'{role}: expected camera image topic')
        else:
            raise ValueError(f'{role}: unsupported camera backend')
    if any(s['backend'] == 'ros_compressed' for s in config['cameras'].values()):
        ros = config.get('ros', {})
        if not ros.get('setup') or not all(isinstance(p, str) and p.startswith('/') for p in ros['setup']):
            raise ValueError('ROS capture requires explicit absolute setup paths')
        if not ros.get('cyclonedds_uri', '').startswith('/'):
            raise ValueError('ROS capture requires current host DDS configuration')
    return config


def load_config(path):
    return validate_config(json.loads(Path(path).read_text()))


def remote_command(source, config):
    command = 'python3 -u -c ' + shlex.quote(source) + ' --serve --config-json ' + shlex.quote(json.dumps(config))
    if any(s['backend'] == 'ros_compressed' for s in config['cameras'].values()):
        ros = config['ros']
        setup = ''.join('source ' + shlex.quote(p) + ' && ' for p in ros['setup'])
        command = setup + 'env RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=' + shlex.quote(ros['cyclonedds_uri']) + ' ' + command
    return command


def check_camera_status(path, now):
    status = json.loads(path.read_text())
    if (status.get('error') or not status.get('ready') or not 0 <= now-status['time'] <= .6
            or set(status['latest']) != set(CAMERAS)
            or any(not 0 <= now-t <= .6 for t in status['latest'].values())):
        raise ValueError('Three-camera recording unavailable or stale')
    return status


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, allow_nan=False) + '\n')
    temp.replace(path)


def read_exact(stream, length):
    data = bytearray()
    while len(data) < length:
        part = stream.read(length - len(data))
        if not part:
            raise EOFError('Camera SSH stream ended')
        data.extend(part)
    return bytes(data)


def read_packet(stream):
    size = struct.unpack('!I', read_exact(stream, 4))[0]
    if not 0 < size <= 65536:
        raise ValueError('Invalid camera header length')
    header = json.loads(read_exact(stream, size))
    length = header.get('bytes', 0)
    if not isinstance(length, int) or not 0 <= length <= 8_000_000:
        raise ValueError('Invalid camera payload length')
    return header, read_exact(stream, length)


def send_packet(header, payload=b''):
    header = dict(header, bytes=len(payload))
    data = json.dumps(header, allow_nan=False).encode()
    sys.stdout.buffer.write(struct.pack('!I', len(data)) + data + payload)
    sys.stdout.buffer.flush()


def serve(config):
    import cv2
    stop = threading.Event()
    lock = threading.Lock()
    latest, errors, devices = {}, {}, {}

    def capture(role, spec):
        camera = None
        try:
            device = Path(spec['device'])
            if not device.exists():
                raise RuntimeError(f'{role}: device missing: {device}')
            camera = cv2.VideoCapture(str(device), cv2.CAP_V4L2)
            if not camera.isOpened():
                raise RuntimeError(f'{role}: open failed')
            for prop, value in ((cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*spec['fourcc'])),
                                (cv2.CAP_PROP_FRAME_WIDTH, spec['shape'][1]), (cv2.CAP_PROP_FRAME_HEIGHT, spec['shape'][0]),
                                (cv2.CAP_PROP_FPS, spec['fps']), (cv2.CAP_PROP_BUFFERSIZE, 1)):
                camera.set(prop, value)
            with lock:
                devices[role] = dict(device=str(device), backend='v4l2', requested_fps=spec['fps'],
                                     reported_fps=camera.get(cv2.CAP_PROP_FPS), rotation_degrees=0)
            seq = 0
            while not stop.is_set():
                begin = time.monotonic()
                ok, frame = camera.read()
                end = time.monotonic()
                if not ok or frame.shape != tuple(spec['shape']):
                    raise RuntimeError(f'{role}: missing frame or unexpected shape')
                ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise RuntimeError(f'{role}: JPEG encode failed')
                seq += 1
                with lock:
                    latest[role] = (dict(type='frame', role=role, sequence=seq,
                        remote_read_start=begin, remote_read_end=end, shape=spec['shape'],
                        timing='v4l2_read_completion_not_exposure'), jpeg.tobytes())
        except Exception as exc:
            with lock:
                errors[role] = f'{type(exc).__name__}: {exc}'
            stop.set()
        finally:
            if camera is not None:
                camera.release()

    def capture_ros(specs):
        node = None
        context = None
        executor = None
        try:
            import numpy as np
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import CompressedImage
            context = Context()
            rclpy.init(context=context)
            node = rclpy.create_node('g1_camera_readonly_capture', context=context,
                enable_rosout=False, start_parameter_services=False)
            executor = SingleThreadedExecutor(context=context)
            executor.add_node(node)
            sequences = dict.fromkeys(specs, 0)
            stamps = {}

            def receive_image(message, role):
                begin = time.monotonic()
                stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
                if stamp <= stamps.get(role, -1):
                    return  # Replayed/stalled source timestamps never refresh liveness.
                payload = bytes(message.data)
                frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                if frame is None or frame.shape != tuple(specs[role]['shape']) or not payload.startswith(b'\xff\xd8'):
                    raise RuntimeError(f'{role}: unexpected JPEG dimensions from existing publisher')
                stamps[role] = stamp
                sequences[role] += 1
                with lock:
                    latest[role] = (dict(type='frame', role=role, sequence=sequences[role],
                        remote_read_start=begin, remote_read_end=time.monotonic(), shape=specs[role]['shape'],
                        source_timestamp_ns=stamp, timing='ros_callback_receipt_not_exposure'), payload)

            subscriptions = []
            for role, spec in specs.items():
                subscriptions.append(node.create_subscription(CompressedImage, spec['topic'],
                    lambda msg, role=role: receive_image(msg, role), qos_profile_sensor_data))
                with lock:
                    devices[role] = dict(backend='ros_compressed', topic=spec['topic'],
                        requested_fps=spec['fps'], reported_fps=None, rotation_degrees=0,
                        encoding='existing publisher JPEG; not re-encoded',
                        expected_serial=spec.get('expected_serial'))
            while not stop.is_set():
                executor.spin_once(timeout_sec=.1)
        except Exception as exc:
            with lock:
                errors['ros'] = f'{type(exc).__name__}: {exc}'
            stop.set()
        finally:
            if executor is not None:
                executor.shutdown()
            if node is not None:
                node.destroy_node()
            if context is not None and context.ok():
                context.shutdown()

    workers = [threading.Thread(target=capture, args=(role, spec), daemon=True)
               for role, spec in config['cameras'].items() if spec['backend'] == 'v4l2']
    ros_specs = {role: spec for role, spec in config['cameras'].items() if spec['backend'] == 'ros_compressed'}
    if ros_specs:
        workers.append(threading.Thread(target=capture_ros, args=(ros_specs,), daemon=True))
    try:
        for thread in workers:
            thread.start()
        deadline = time.monotonic() + 15
        while len(latest) < 3 and not stop.is_set() and time.monotonic() < deadline:
            time.sleep(.01)
        if len(latest) != 3 or errors:
            with lock:
                missing=sorted(set(config['cameras'])-set(latest))
                details=dict(errors)
            raise RuntimeError(f'Camera startup failed: missing streams={missing}; errors={details}')
        send_packet(dict(type='hello', devices=devices))
        while True:
            command = sys.stdin.readline().strip()
            if command == 'PING':
                send_packet(dict(type='pong', remote_monotonic=time.monotonic()))
            elif command == 'START':
                break
            else:
                return
        def watch_stdin():
            sys.stdin.readline()  # STOP or EOF both end this bounded camera session.
            stop.set()
        threading.Thread(target=watch_stdin, daemon=True).start()
        last = dict.fromkeys(CAMERAS, 0)
        deadline = time.monotonic() + 1200
        while not stop.is_set() and time.monotonic() < deadline:
            with lock:
                snapshots = dict(latest)
            for role, (header, jpeg) in snapshots.items():
                if time.monotonic() - header['remote_read_end'] > 1:
                    raise RuntimeError(f'{role}: camera stalled')
                if header['sequence'] > last[role]:
                    send_packet(header, jpeg)
                    last[role] = header['sequence']
            time.sleep(.002)
        if errors:
            raise RuntimeError(str(errors))
    finally:
        stop.set()
        for thread in workers:
            thread.join(timeout=.5)


def record(args):
    config = load_config(getattr(args, 'config', DEFAULT_CONFIG))
    args.output.mkdir(parents=True, exist_ok=False)
    for role in CAMERAS:
        (args.output / role).mkdir()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    # Execute this exact source on the camera host; no installation or remote writes.
    source = Path(__file__).read_text()
    command = remote_command(source, config)
    ssh = subprocess.Popen(['ssh', '-T', '-i', str(args.ssh_key), '-o', 'BatchMode=yes',
        '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=5',
        '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2', args.host, command],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    latest, error, metadata = {}, [], {}
    lock = threading.Lock()

    def receive():
        try:
            hello, _ = read_packet(ssh.stdout)
            if hello.get('type') != 'hello' or set(hello['devices']) != set(CAMERAS):
                raise ValueError('Unexpected camera handshake')
            probes = []
            for _ in range(8):
                begin = time.monotonic()
                ssh.stdin.write(b'PING\n'); ssh.stdin.flush()
                pong, _ = read_packet(ssh.stdout)
                end = time.monotonic()
                if pong.get('type') != 'pong':
                    raise ValueError('Clock handshake failed')
                probes.append(dict(rtt=end-begin, offset=(begin+end)/2-pong['remote_monotonic']))
            best = min(probes, key=lambda p: p['rtt'])
            metadata.update(hello=hello, clock_probes=probes, clock_offset=best['offset'],
                startup_clock_uncertainty_s=best['rtt']/2,
                capture_config=config,
                timing='V4L2 completion or ROS callback receipt, not exposure; export causality uses local receipt',
                clock_drift_corrected=False, source_sha256=__import__('hashlib').sha256(source.encode()).hexdigest())
            atomic_json(args.output/'metadata.json', metadata)
            ssh.stdin.write(b'START\n'); ssh.stdin.flush()
            previous = dict.fromkeys(CAMERAS, 0)
            with (args.output/'frames.jsonl').open('x', buffering=1) as log:
                while not stop.is_set():
                    header, jpeg = read_packet(ssh.stdout)
                    received = time.monotonic()
                    role = header['role']
                    seq = header['sequence']
                    if (header.get('type') != 'frame' or role not in CAMERAS or not jpeg
                            or not isinstance(seq, int) or seq <= previous[role]
                            or header['shape'] != config['cameras'][role]['shape']
                            or not all(math.isfinite(header[k]) for k in ('remote_read_start','remote_read_end'))
                            or header['remote_read_end'] < header['remote_read_start']):
                        raise ValueError('Invalid camera frame')
                    name = f'{role}/{seq:08d}.jpg'
                    (args.output/name).write_bytes(jpeg)
                    row = dict(header, file=name, host_received=received,
                               estimated_read_end=header['remote_read_end']+best['offset'],
                               skipped_source_frames=max(0, seq-previous[role]-1) if previous[role] else 0)
                    log.write(json.dumps(row, allow_nan=False)+'\n')
                    previous[role] = seq
                    with lock:
                        latest[role] = received
        except Exception as exc:
            if not stop.is_set():
                error.append(f'{type(exc).__name__}: {exc}')
            stop.set()

    reader = threading.Thread(target=receive, daemon=True)
    reader.start()
    start = time.monotonic()
    recording_start = None
    failure = None
    try:
        while not stop.is_set():
            now = time.monotonic()
            with lock:
                times = dict(latest)
            if error:
                raise RuntimeError(error[0])
            if now-start > 25 and len(times) != 3:
                raise RuntimeError('Three-camera startup deadline')
            if times and any(now-t > 1 for t in times.values()):
                raise RuntimeError('Camera stream stalled')
            ready = len(times) == 3 and all(0 <= now-t <= .6 for t in times.values())
            if ready and recording_start is None:
                recording_start = now
            atomic_json(args.output/'status.json', dict(time=now, ready=ready, latest=times, error=None))
            if args.duration and recording_start is not None and now-recording_start >= args.duration:
                break
            stop.wait(.05)
        if error:
            raise RuntimeError(error[0])
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
    finally:
        stop.set()
        try:
            ssh.stdin.write(b'STOP\n'); ssh.stdin.flush(); ssh.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            ssh.wait(timeout=3)
        except subprocess.TimeoutExpired:
            ssh.terminate()
            try: ssh.wait(timeout=2)
            except subprocess.TimeoutExpired: ssh.kill(); ssh.wait()
        reader.join(timeout=2)
        if reader.is_alive():failure = failure or 'Camera reader did not exit'
        if len(latest) != len(CAMERAS):
            failure = failure or 'Camera startup incomplete: missing frames from ' + ', '.join(sorted(set(CAMERAS)-set(latest)))
        atomic_json(args.output/'status.json',dict(time=time.monotonic(),ready=False,latest=latest,error=failure))
        atomic_json(args.output/'summary.json',dict(error=failure,elapsed_s=time.monotonic()-start,latest=latest))
    if failure:
        raise RuntimeError(failure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--host', default='unitree@192.168.123.164')
    parser.add_argument('--ssh-key', type=Path, default=Path.home()/'.ssh/g1_camera_ed25519')
    parser.add_argument('--duration', type=float, default=0.)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--config-json', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.serve:
        serve(validate_config(json.loads(args.config_json)) if args.config_json else load_config(args.config)); return
    if args.output is None or not math.isfinite(args.duration) or args.duration < 0:
        parser.error('--output required; duration must be finite and nonnegative')
    record(args)


if __name__ == '__main__':
    main()
