#!/usr/bin/env python3
"""Capture the right D405 without rotation after hardware orientation correction."""
import argparse
import json
from pathlib import Path
import time

import cv2

SERIAL = '261523072307'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    devices = list(Path('/dev/v4l/by-id').glob(f'*{SERIAL}-video-index4'))
    if len(devices) != 1:
        raise RuntimeError(f'Expected one RGB node for right D405 {SERIAL}: {devices}')
    args.output.mkdir(parents=True, exist_ok=False)
    camera = cv2.VideoCapture(str(devices[0]), cv2.CAP_V4L2)
    frames = []
    try:
        if not camera.isOpened():
            raise RuntimeError('Cannot open right camera')
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 15)
        for index in range(1, 16):
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError(f'Frame {index} read failed')
            name = f'frame_{index:02d}.jpg'
            if not cv2.imwrite(str(args.output / name), frame):
                raise RuntimeError(f'Failed to save {name}')
            frames.append(dict(file=name, shape=list(frame.shape),
                               host_monotonic_s=time.monotonic()))
    finally:
        camera.release()
    report = dict(role='right_wrist', serial=SERIAL, device=str(devices[0]),
                  rotation_degrees=0, frames=frames)
    (args.output / 'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(serial=SERIAL, rotation_degrees=0, saved_frames=len(frames))))


if __name__ == '__main__':
    main()
