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


def check_camera_status(path, now):
    status = json.loads(path.read_text())
    if (status.get('error') or not status.get('ready') or not 0 <= now-status['time'] <= .3
            or set(status['latest']) != set(CAMERAS)
            or any(not 0 <= now-t <= .3 for t in status['latest'].values())):
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


def serve():
    import cv2
    stop = threading.Event()
    lock = threading.Lock()
    latest, errors, devices = {}, {}, {}

    def capture(role, pattern, fourcc):
        camera = None
        try:
            nodes = list(Path('/dev/v4l/by-id').glob(pattern))
            if len(nodes) != 1:
                raise RuntimeError(f'{role}: expected unique device matching {pattern}: {nodes}')
            camera = cv2.VideoCapture(str(nodes[0]), cv2.CAP_V4L2)
            if not camera.isOpened():
                raise RuntimeError(f'{role}: open failed')
            for prop, value in ((cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc)),
                                (cv2.CAP_PROP_FRAME_WIDTH, 640), (cv2.CAP_PROP_FRAME_HEIGHT, 480),
                                (cv2.CAP_PROP_FPS, 15), (cv2.CAP_PROP_BUFFERSIZE, 1)):
                camera.set(prop, value)
            with lock:
                devices[role] = dict(device=str(nodes[0]), requested_fps=15,
                                     reported_fps=camera.get(cv2.CAP_PROP_FPS), rotation_degrees=0)
            seq = 0
            while not stop.is_set():
                begin = time.monotonic()
                ok, frame = camera.read()
                end = time.monotonic()
                if not ok or frame.shape != (480, 640, 3):
                    raise RuntimeError(f'{role}: missing frame or unexpected shape')
                ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise RuntimeError(f'{role}: JPEG encode failed')
                seq += 1
                with lock:
                    latest[role] = (dict(type='frame', role=role, sequence=seq,
                        remote_read_start=begin, remote_read_end=end, shape=[480, 640, 3]), jpeg.tobytes())
        except Exception as exc:
            with lock:
                errors[role] = f'{type(exc).__name__}: {exc}'
            stop.set()
        finally:
            if camera is not None:
                camera.release()

    workers = [threading.Thread(target=capture, args=(role, *spec), daemon=True)
               for role, spec in CAMERAS.items()]
    try:
        for thread in workers:
            thread.start()
        deadline = time.monotonic() + 15
        while len(latest) < 3 and not stop.is_set() and time.monotonic() < deadline:
            time.sleep(.01)
        if len(latest) != 3 or errors:
            raise RuntimeError(f'Camera startup failed: {errors}')
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
    args.output.mkdir(parents=True, exist_ok=False)
    for role in CAMERAS:
        (args.output / role).mkdir()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    # Execute this exact source on the camera host; no installation or remote writes.
    source = Path(__file__).read_text()
    command = 'python3 -u -c ' + shlex.quote(source) + ' --serve'
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
                timing='OpenCV read-completion proxy, not exposure; export causality uses local receipt',
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
                            or header['shape'] != [480, 640, 3]
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
            ready = len(times) == 3 and all(0 <= now-t <= .3 for t in times.values())
            atomic_json(args.output/'status.json', dict(time=now, ready=ready, latest=times, error=None))
            if args.duration and now-start >= args.duration:
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
    args = parser.parse_args()
    if args.serve:
        serve(); return
    if args.output is None or not math.isfinite(args.duration) or args.duration < 0:
        parser.error('--output required; duration must be finite and nonnegative')
    record(args)


if __name__ == '__main__':
    main()
