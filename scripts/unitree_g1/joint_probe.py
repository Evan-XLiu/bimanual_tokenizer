#!/usr/bin/env python3
"""Finite left-elbow +0.02 rad/return candidate probe. Default is shadow only."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import signal
import threading

from hold_arms import DDSHoldIO, load_profile, run_hold

ROOT = Path(__file__).resolve().parents[2]


class ElbowProbe:
    def initialize(self, executor, now):
        self.initial = executor.observation.q
        self.started = None
        # All other targets remain exactly at initial feedback. Check both the
        # requested target and the interpolator's stopping distance envelope.
        c = executor.c
        if self.initial[3]+.025 > c.upper[3]:
            raise ValueError('Insufficient elbow limit margin')
        radii = [.001]*14
        radii[3] = .025
        executor.c = replace(c,
            lower=tuple(max(lo,q-r) for lo,q,r in zip(c.lower,self.initial,radii)),
            upper=tuple(min(hi,q+r) for hi,q,r in zip(c.upper,self.initial,radii)),
            velocity=(.01,)*14, acceleration=(.02,)*14)

    def update(self, executor, now):
        if executor.phase != 'ACTIVE':
            return
        if self.started is None:
            self.started = now
        target = list(self.initial)
        if now-self.started < 3.:
            target[3] += .02
        executor.send_targets(target, now, now)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--scene-confirmation')
    args = parser.parse_args()
    if args.execute and not (args.scene_confirmation or '').strip():
        parser.error('Current --scene-confirmation required')
    profile, config = load_profile(ROOT/'artifacts/unitree_g1/hold_shadow_profile.json', False)
    profile['hold_s'] = 6.
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(profile=profile, hardware_motion_enabled=args.execute,
        scene_confirmation=args.scene_confirmation, joint='left_elbow_joint', index=18,
        displacement_rad=.02, velocity_rad_s=.01, acceleration_rad_s2=.02,
        phases_s={'engage':2., 'outward':3., 'return':3., 'unload':2., 'tail':.3},
        source_sha256={name:hashlib.sha256((ROOT/'scripts/unitree_g1'/name).read_bytes()).hexdigest()
                       for name in ('joint_probe.py','hold_arms.py','arm_execution.py')})
    (args.output/'session.json').write_text(json.dumps(metadata, indent=2)+'\n')
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    io = None
    result = dict(error='initialization_failed', command_frames=0, unload_submitted=False)
    try:
        io = DDSHoldIO(args.interface, args.execute, profile)
        io.preflight()
        if stop.is_set():
            raise RuntimeError('Stopped before engagement')
        io.open_writer()
        with (args.output/'commands.jsonl').open('x', buffering=1) as log:
            result = run_hold(io, config, profile, stop.is_set,
                lambda row: log.write(json.dumps(row, allow_nan=False)+'\n'), target_source=ElbowProbe())
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if io:
            io.close()
        result.update(hardware_motion_enabled=args.execute, physical_unload_verified=False)
        if not args.execute:
            result['unload_submitted'] = False
        (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2), flush=True)
    return 1 if result['error'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
