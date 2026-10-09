#!/usr/bin/env python3
"""Finite internal Dex1 small open/return probe. Default: read-only preflight.

Only lowcmd slots 31/33 may receive gains. Arms/body remain passive; no Quest
input, mode switching, full-stroke calibration, or automatic retry.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import signal
import threading
import time

from arm_execution import Observation, vector
from hold_arms import DDSHoldIO, load_profile, run_hold

ROOT = Path(__file__).resolve().parents[2]
INDICES = (31, 33)
# Pinned upstream internal Dex1 candidate values; not hardware calibration.
KP, KD = 5., .05
DISPLACEMENT = .5
VELOCITY, ACCELERATION = .2, .4
TRACKING_ERROR = .15
TRAVEL, DWELL = 5., 1.
ACTIVE = 2*(TRAVEL+DWELL)
NEGATIVE_OPEN_DISPLACEMENT = -.2


def gripper_fault(fault):
    if fault is None:
        return None
    mapped=dict(fault)
    slots=mapped.pop('tracking_exceeded_arm_indices',[])
    mapped['tracking_exceeded_gripper_motor_indices']=[INDICES[i-15] for i in slots if i in (15,16)]
    mapped['tracking_error_motor_units']=mapped.pop('tracking_error_rad')[:2]
    return mapped


def gripper_log(row):
    """Label the reused executor's padded vector; report physical motor indices."""
    row=dict(row)
    row['execution_vector_layout']='first two entries left/right Dex1; remaining twelve inert padding'
    row['gripper_motor_indices']=list(INDICES)
    row['first_fault']=gripper_fault(row.get('first_fault'))
    return row


def encode_gripper_command(command):
    """Encode only internal Dex1 slots; never copy arm gains into LowCmd."""
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    q, kp, kd = vector(command.q), vector(command.kp), vector(command.kd)
    if min(kp+kd) < 0:
        raise ValueError('Negative gains')
    if any(abs(x)>1e-12 for x in q[2:]):
        raise ValueError('Nonzero unused execution slots')
    msg=unitree_hg_msg_dds__LowCmd_()
    msg.mode_pr, msg.mode_machine = 0, command.mode_machine
    # Keep the same passive body convention as the reviewed arm executor.
    for index,motor in enumerate(msg.motor_cmd):
        motor.mode=1 if index < 29 or index in INDICES else 0
    for j,index in enumerate(INDICES):
        m=msg.motor_cmd[index]
        m.q,m.kp,m.kd=q[j],kp[j],kd[j]
    msg.crc=CRC().Crc(msg)
    return msg


def observation_from_telemetry(telemetry):
    if telemetry is None:
        raise ValueError('No gripper telemetry')
    motors={m['index']:m for m in telemetry['motors']}
    q=[]
    for index in INDICES:
        if index not in motors:
            raise ValueError(f'Missing internal Dex1 slot {index}')
        m=motors[index]
        if m['motorstate']!=0 or not all(math.isfinite(m[k]) for k in ('q','dq')):
            raise ValueError(f'Invalid internal Dex1 feedback at {index}')
        q.append(m['q'])
    return Observation(vector(q+[0.]*12),telemetry['received'],telemetry['tick'],5)


class GripperIO(DDSHoldIO):
    def observation(self):
        # Superclass callback validates machine variant and tick progression.
        with self.lock:
            if self.feedback_error:
                raise ValueError(self.feedback_error)
            return observation_from_telemetry(self.latest_telemetry)

    def preflight(self):
        super().preflight()
        end=time.monotonic()+.5
        while time.monotonic()<end:
            self.check()
            state=self.observation()
            if not 0<=time.monotonic()-state.received<=.1:
                raise ValueError('Stale gripper feedback')
            telemetry=self.telemetry()
            if any(abs(m['dq'])>.1 for m in telemetry['motors'] if m['index'] in INDICES):
                raise ValueError('Grippers not stationary')
            time.sleep(.02)

    def write(self,command):
        if not self.live:
            raise RuntimeError('Read-only gripper mode cannot publish commands')
        if self.writer is None or self.matched==0:
            return False
        self.writer.write(encode_gripper_command(command))
        return True


def probe_config(profile_config, initial, kp=KP, negative_open=False):
    if kp not in (5.,6.,8.):
        raise ValueError('Only reviewed diagnostic Kp 5, 6 or 8 is supported')
    q=vector(initial)
    if any(abs(x)>1e-12 for x in q[2:]):
        raise ValueError('Invalid padded feedback')
    if negative_open:
        # Local direction probe from the observed near-closed pose. This is
        # not a calibrated full negative travel range.
        if kp != KP or any(not -.05 <= x <= .05 for x in q[:2]):
            raise ValueError('Negative opening probe requires Kp=5 and near-zero initial pose')
        return replace(profile_config,
            lower=tuple(x+NEGATIVE_OPEN_DISPLACEMENT-.01 for x in q[:2])+(-.1,)*12,
            upper=tuple(x+.01 for x in q[:2])+(.1,)*12,
            kp=(KP,)*14,kd=(KD,)*14,
            velocity=(.1,)*14,acceleration=(.2,)*14,
            tracking_error=(TRACKING_ERROR,)*14)
    # The small negative tolerance only admits the observed near-zero offset;
    # it is not an instruction to search for or drive through the closed stop.
    if any(not -.05<=x<=5.4-DISPLACEMENT-.01 for x in q[:2]):
        raise ValueError('Insufficient margin for small open/return probe')
    return replace(profile_config,
        lower=tuple(min(0.,x)-.01 for x in q[:2])+(-.1,)*12,
        upper=tuple(x+DISPLACEMENT+.01 for x in q[:2])+(.1,)*12,
        kp=(kp,)*14,kd=(KD,)*14,
        velocity=(VELOCITY,)*14,acceleration=(ACCELERATION,)*14,
        tracking_error=(TRACKING_ERROR,)*14)


class GripperSource:
    def __init__(self,expected_initial,kp=KP,negative_open=False):
        self.expected=vector(expected_initial)
        self.kp=kp
        self.negative_open=negative_open
        self.started=None
        self.initial=None

    def initialize(self,executor,now):
        self.initial=executor.observation.q
        if max(abs(a-b) for a,b in zip(self.initial,self.expected))>.005:
            raise ValueError('Gripper pose changed after preflight')
        # Use actual engagement feedback as first target and return position.
        executor.c=probe_config(executor.c,self.initial,self.kp,self.negative_open)

    def update(self,executor,now):
        if executor.phase!='ACTIVE':return
        if self.started is None:self.started=now
        elapsed=now-self.started
        if elapsed<TRAVEL:u=max(0.,elapsed/TRAVEL)
        elif self.negative_open or elapsed<TRAVEL+DWELL:u=1.
        elif elapsed<2*TRAVEL+DWELL:u=1-(elapsed-TRAVEL-DWELL)/TRAVEL
        else:u=0.
        fraction=u*u*u*(10+u*(-15+6*u))
        target=list(self.initial)
        displacement=NEGATIVE_OPEN_DISPLACEMENT if self.negative_open else DISPLACEMENT
        for j in range(2):target[j]+=displacement*fraction
        executor.send_targets(target,now,now)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--negative-open',action='store_true',
                        help='Near-zero start: -0.2 local opening probe, hold 1s then unload without returning')
    parser.add_argument('--kp',type=float,choices=(5.,6.,8.),default=KP,
                        help='Reviewed diagnostic gain; tracking protection remains 0.15')
    parser.add_argument('--scene-confirmation',help='Current Dex1, empty jaws, fingers clear, body/arm support and stop readiness')
    args=parser.parse_args()
    if args.negative_open and args.kp != KP:
        parser.error('--negative-open requires --kp 5')
    if args.execute and not (args.scene_confirmation or '').strip():
        parser.error('--execute requires current --scene-confirmation')
    profile,config=load_profile(ROOT/'artifacts/unitree_g1/hold_shadow_profile.json',False)
    profile['hold_s']=TRAVEL+DWELL if args.negative_open else ACTIVE
    profile['period_s']=.004  # Match the pinned internal Dex1 publisher (250 Hz).
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'source').mkdir()
    hashes={}
    for name in ('gripper_probe.py','hold_arms.py','arm_execution.py'):
        data=(ROOT/'scripts/unitree_g1'/name).read_bytes()
        (args.output/'source'/name).write_bytes(data);hashes[name]=hashlib.sha256(data).hexdigest()
    metadata=dict(hardware_approved=False,hardware_motion_enabled=args.execute,
        scene_confirmation=args.scene_confirmation,interface=args.interface,
        motor_indices=list(INDICES),kp=args.kp,kd=KD,
        negative_open=args.negative_open,
        displacement_motor_units=NEGATIVE_OPEN_DISPLACEMENT if args.negative_open else DISPLACEMENT,
        period_s=profile['period_s'],
        velocity_motor_units_s=.1 if args.negative_open else VELOCITY,
        acceleration_motor_units_s2=.2 if args.negative_open else ACCELERATION,
        tracking_error_motor_units=TRACKING_ERROR,
        phases_s=dict(engage=2.,open=TRAVEL,dwell=DWELL,
                      return_to_start=0. if args.negative_open else TRAVEL,
                      settle=0. if args.negative_open else DWELL,unload=2.,tail=.3),
        source_sha256=hashes,scope='Candidate local direction probe; motor units are not jaw gap or calibrated force. All other motors passive.')
    (args.output/'session.json').write_text(json.dumps(metadata,indent=2)+'\n')
    stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    io=None;result=dict(error='initialization_failed',command_frames=0,unload_submitted=False)
    try:
        io=GripperIO(args.interface,args.execute,profile);io.preflight()
        state=io.observation();config=probe_config(config,state.q,args.kp,args.negative_open)
        (args.output/'preflight.json').write_text(json.dumps(dict(observation=asdict(state),telemetry=io.telemetry(),controller=asdict(config)),indent=2)+'\n')
        if stop.is_set():raise RuntimeError('Stopped before engagement')
        if not args.execute:
            result=dict(error=None,preflight_passed=True,command_frames=0,unload_submitted=False)
        else:
            motion='-0.2 motor units, no return, 10.3s' if args.negative_open else '+0.5 motor units then return, 16.3s'
            print(f'Dex1 slots 31/33: {motion}; Kp={args.kp}, Kd={KD}, '
                  'tracking limit=0.15; all arm/body gains zero.',flush=True)
            io.open_writer()
            with (args.output/'commands.jsonl').open('x',buffering=1) as log:
                result=run_hold(io,config,profile,stop.is_set,
                    lambda row:log.write(json.dumps(gripper_log(row),allow_nan=False)+'\n'),
                    target_source=GripperSource(state.q,args.kp,args.negative_open))
    except Exception as exc:result['error']=f'{type(exc).__name__}: {exc}'
    finally:
        if io:io.close()
        if 'first_fault' in result:
            result['first_fault']=gripper_fault(result['first_fault'])
        result.update(hardware_motion_enabled=args.execute,physical_unload_verified=False,
            execution_vector_layout='first two entries = left/right Dex1, remaining twelve inert padding')
        (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
    return 1 if result['error'] else 0


if __name__=='__main__':raise SystemExit(main())
