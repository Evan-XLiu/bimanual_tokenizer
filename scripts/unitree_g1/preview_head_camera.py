#!/usr/bin/env python3
"""Show high-resolution G1 stereo head video locally over authenticated SSH.

Close the ffplay window or press Q to stop. No robot motion commands.
"""
import shlex
import struct
import subprocess
from pathlib import Path

REMOTE = r'''
import cv2, sys, time, threading, struct
from pathlib import Path
nodes=list(Path('/dev/v4l/by-id').glob('*USB2.0_Camera_RGB*video-index0'))
if len(nodes)!=1: raise RuntimeError('Head camera node not uniquely identified')
c=cv2.VideoCapture(str(nodes[0]),cv2.CAP_V4L2)
latest=[None]; lock=threading.Lock(); stop=threading.Event()
def reader():
    try:
        while not stop.is_set():
            ok,frame=c.read()
            if not ok: break
            with lock: latest[0]=(time.monotonic(),frame)
    finally: stop.set()
try:
    if not c.isOpened(): raise RuntimeError('Head camera could not open')
    c.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
    c.set(cv2.CAP_PROP_FRAME_WIDTH,3840)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT,1200)
    c.set(cv2.CAP_PROP_FPS,5)
    print('Head camera: requested 3840x1200, preview up to 5 FPS',file=sys.stderr,flush=True)
    threading.Thread(target=reader,daemon=True).start()
    last=0.;start=time.monotonic();verified=False
    while not stop.is_set():
        with lock: item=latest[0]
        if item is None:
            if time.monotonic()-start>10: raise RuntimeError('No first frame within 10s')
            time.sleep(.01);continue
        stamp,frame=item
        if time.monotonic()-stamp>3: raise RuntimeError('Camera frame stream stalled')
        if stamp<=last:
            time.sleep(.01);continue
        if not verified:
            if frame.shape[:2]!=(1200,3840): raise RuntimeError('Unexpected resolution '+str(frame.shape))
            print('Verified frame: 3840x1200; stereo side-by-side; no rotation',file=sys.stderr,flush=True)
            verified=True
        last=stamp
        for half in (frame[:,:1920],frame[:,1920:]):
            ok,jpeg=cv2.imencode('.jpg',half,[cv2.IMWRITE_JPEG_QUALITY,90])
            if not ok: raise RuntimeError('JPEG encoding failed')
            data=jpeg.tobytes()
            sys.stdout.buffer.write(struct.pack('!I',len(data)))
            sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
        time.sleep(.2)
except BrokenPipeError:
    pass
finally:
    stop.set()
    c.release()
'''


def main():
    ssh = subprocess.Popen([
        'ssh', '-T', '-i', str(Path.home()/'.ssh/g1_camera_ed25519'),
        '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
        '-o', 'ConnectTimeout=5', '-o', 'ServerAliveInterval=5',
        '-o', 'ServerAliveCountMax=2', 'unitree@192.168.123.164',
        'python3 -u -c '+shlex.quote(REMOTE)], stdout=subprocess.PIPE)
    viewers = []
    def read_exact(size):
        data = bytearray()
        while len(data) < size:
            part = ssh.stdout.read(size-len(data))
            if not part:
                raise EOFError('Head camera stream ended')
            data.extend(part)
        return data
    try:
        for index, label in enumerate(('Left half', 'Right half')):
            viewers.append(subprocess.Popen([
                'ffplay', '-hide_banner', '-loglevel', 'warning',
                '-f', 'mjpeg', '-framerate', '5', '-probesize', '32',
                '-analyzeduration', '0', '-framedrop', '-autoexit',
                '-window_title', f'G1 Head {label} | 1920x1200 | Q to stop',
                '-x', '960', '-y', '600', '-left', str(index*960), '-top', '40',
                '-i', 'pipe:0'], stdin=subprocess.PIPE))
        frames = 0
        while all(v.poll() is None for v in viewers):
            for viewer in viewers:
                size = struct.unpack('!I', read_exact(4))[0]
                if not 0 < size < 10000000:
                    raise ValueError('Invalid image packet length')
                viewer.stdin.write(read_exact(size))
                viewer.stdin.flush()
            frames += 1
            if frames == 1 or frames % 150 == 0:
                print(f'Delivered {frames} stereo frame pairs to two preview windows', flush=True)
        return 0
    except (BrokenPipeError, EOFError) as exc:
        print(str(exc), flush=True)
        return 0
    finally:
        for viewer in viewers:
            if viewer.poll() is None:
                viewer.terminate()
                try: viewer.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    viewer.kill();viewer.wait()
        if ssh.poll() is None:
            ssh.terminate()
            try: ssh.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ssh.kill();ssh.wait()


if __name__ == '__main__':
    raise SystemExit(main())
