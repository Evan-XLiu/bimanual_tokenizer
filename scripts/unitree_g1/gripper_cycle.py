#!/usr/bin/env python3
"""Finite Dex1 negative-open / positive-close cycle; default read-only.

Closing targets come from the near-closed start of the successful direction
probe, not a newly inferred mechanical stop or a full-stroke calibration.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import signal
import threading
import time

from arm_execution import vector
from gripper_probe import GripperIO, gripper_log, gripper_fault, ROOT
from hold_arms import load_profile, run_hold

OPEN_DELTA = -.5
OPEN_S, CLOSE_S, DWELL_S = 10., 16., 1.
ACTIVE_S = OPEN_S+CLOSE_S+2*DWELL_S
ABSOLUTE_OPEN_S = 110.
REFERENCE = ROOT/'artifacts/unitree_g1/gripper_negative_open_live_20261009_001.json'


def closed_reference():
    report=json.loads(REFERENCE.read_text())
    motors={m['index']:m for m in report['motors']}
    return vector([motors[i]['initial_q'] for i in (31,33)]+[0.]*12)


def cycle_config(base, initial, closed, open_target=None):
    initial,closed=vector(initial),vector(closed)
    if any(x != 0 for x in initial[2:]+closed[2:]):
        raise ValueError('Nonzero padding')
    if any(not -.05<=c<=.05 or not c-.35<=q<=c+.03
           for q,c in zip(initial[:2],closed[:2])):
        raise ValueError('Outside reviewed initial/closed-reference envelope')
    if open_target is not None and open_target != -5.4:
        raise ValueError('Only the explicitly requested -5.4 diagnostic target is supported')
    return replace(base,
        lower=tuple((q+OPEN_DELTA if open_target is None else open_target)-.01 for q in initial[:2])+(-.1,)*12,
        upper=tuple(max(q,c)+.01 for q,c in zip(initial[:2],closed[:2]))+(.1,)*12,
        kp=(5.,)*14,kd=(.05,)*14,velocity=(.1,)*14,
        acceleration=(.2,)*14,tracking_error=(.15,)*14)


def smooth(u):
    u=max(0.,min(1.,u))
    return u*u*u*(10+u*(-15+6*u))


class CycleSource:
    def __init__(self,initial,closed,open_target=None):
        self.expected=vector(initial)
        self.closed=vector(closed)
        self.started=None
        self.open_target=open_target

    def initialize(self,executor,now):
        self.initial=executor.observation.q
        if max(abs(a-b) for a,b in zip(self.initial,self.expected))>.005:
            raise ValueError('Gripper pose changed after preflight')
        executor.c=cycle_config(executor.c,self.initial,self.closed,self.open_target)
        self.opened=tuple(x+OPEN_DELTA if self.open_target is None else self.open_target for x in self.initial[:2])+(0.,)*12

    def update(self,executor,now):
        if executor.phase!='ACTIVE':return
        if self.started is None:self.started=now
        elapsed=now-self.started
        if self.open_target is not None:
            start,end,u=self.initial,self.opened,elapsed/ABSOLUTE_OPEN_S
        elif elapsed<OPEN_S:
            start,end,u=self.initial,self.opened,elapsed/OPEN_S
        elif elapsed<OPEN_S+DWELL_S:
            start,end,u=self.opened,self.opened,1.
        else:
            start,end,u=self.opened,self.closed,(elapsed-OPEN_S-DWELL_S)/CLOSE_S
        f=smooth(u)
        executor.send_targets(tuple(a+(b-a)*f for a,b in zip(start,end)),now,now)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--open-target',type=float,choices=[-5.4],
                        help='Diagnostic absolute opening target; 110s opening then unload, no closing return')
    parser.add_argument('--scene-confirmation')
    args=parser.parse_args()
    if args.execute and not (args.scene_confirmation or '').strip():
        parser.error('--execute requires --scene-confirmation')
    closed=closed_reference()
    profile,base=load_profile(ROOT/'artifacts/unitree_g1/hold_shadow_profile.json',False)
    profile.update(hold_s=ACTIVE_S if args.open_target is None else ABSOLUTE_OPEN_S+DWELL_S,period_s=.004)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'source').mkdir()
    hashes={}
    for name in ('gripper_cycle.py','gripper_probe.py','hold_arms.py','arm_execution.py'):
        data=(ROOT/'scripts/unitree_g1'/name).read_bytes()
        (args.output/'source'/name).write_bytes(data)
        hashes[name]=hashlib.sha256(data).hexdigest()
    (args.output/'closed_reference.json').write_bytes(REFERENCE.read_bytes())
    metadata=dict(hardware_approved=False,hardware_motion_enabled=args.execute,
        scene_confirmation=args.scene_confirmation,interface=args.interface,
        motor_indices=[31,33],closed_reference=list(closed[:2]),
        reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        displacement_motor_units=OPEN_DELTA,kp=5.,kd=.05,
        velocity_motor_units_s=.1,acceleration_motor_units_s2=.2,
        tracking_error_motor_units=.15,period_s=.004,
        phases_s=dict(engage=2.,open=OPEN_S,open_dwell=DWELL_S,
                      close=CLOSE_S,closed_dwell=DWELL_S,unload=2.,tail=.3),
        source_sha256=hashes,
        scope='Local open and close; closed reference is previously observed, not calibrated contact. Other motors passive.')
    if args.open_target is not None:
        metadata.update(absolute_open_target=args.open_target,displacement_motor_units=None,
            phases_s=dict(engage=2.,open=ABSOLUTE_OPEN_S,open_dwell=DWELL_S,unload=2.,tail=.3),
            scope='Uncalibrated -5.4 opening diagnostic, no automatic close; other motors passive; stop on tracking fault.')
    (args.output/'session.json').write_text(json.dumps(metadata,indent=2)+'\n')
    stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    io=None
    result=dict(error='initialization_failed',command_frames=0,unload_submitted=False)
    try:
        io=GripperIO(args.interface,args.execute,profile)
        io.preflight()
        state=io.observation()
        config=cycle_config(base,state.q,closed,args.open_target)
        (args.output/'preflight.json').write_text(json.dumps(dict(
            observation=asdict(state),telemetry=io.telemetry(),controller=asdict(config)),indent=2)+'\n')
        if stop.is_set():raise RuntimeError('Stopped before engagement')
        if args.execute:
            print('Dex1: absolute -5.4 opening over 110s, then unload without closing; Kp5/Kd.05.' if args.open_target is not None else
                  'Dex1: -0.5 from current, then close to recorded reference; 32.3s; Kp5/Kd.05.',flush=True)
            io.open_writer()
            with (args.output/'commands.jsonl').open('x',buffering=1) as log:
                result=run_hold(io,config,profile,stop.is_set,
                    lambda row:log.write(json.dumps(gripper_log(row),allow_nan=False)+'\n'),
                    target_source=CycleSource(state.q,closed,args.open_target))
        else:
            result=dict(error=None,preflight_passed=True,command_frames=0,unload_submitted=False)
    except Exception as exc:result['error']=f'{type(exc).__name__}: {exc}'
    finally:
        if io:io.close()
        if 'first_fault' in result:result['first_fault']=gripper_fault(result['first_fault'])
        result.update(hardware_motion_enabled=args.execute,physical_unload_verified=False,
            execution_vector_layout='first two entries = left/right Dex1, remaining twelve inert padding')
        (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
    return 1 if result['error'] else 0


if __name__=='__main__':raise SystemExit(main())
