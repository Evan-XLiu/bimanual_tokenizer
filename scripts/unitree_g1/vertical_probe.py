#!/usr/bin/env python3
"""One finite bilateral +5cm/return diagnostic, free orientation. Default read-only planning."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
TRAVEL = 12.
DWELL = 2.
ACTIVE = 2*(TRAVEL+DWELL)
RADIUS = .195  # Reserve room for the <=.003 rad pre-engagement pose drift.


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


class VerticalSource:
    def __init__(self, plan):
        self.plan = plan
        self.started = None
        self.target(0.)  # Load NumPy before the timed servo loop starts.

    def target(self, elapsed):
        import numpy as np
        if elapsed < TRAVEL:
            u = max(0., elapsed/TRAVEL)
        elif elapsed < TRAVEL+DWELL:
            u = 1.
        elif elapsed < 2*TRAVEL+DWELL:
            u = 1.-(elapsed-TRAVEL-DWELL)/TRAVEL
        else:
            u = 0.
        s = u*u*u*(10+u*(-15+6*u))
        points = self.plan['q_path']
        x = s*(len(points)-1)
        i = min(int(x),len(points)-2)
        return tuple((np.asarray(points[i])*(1-(x-i))+np.asarray(points[i+1])*(x-i)).tolist())

    def initialize(self, executor, now):
        q = executor.observation.q
        if max(abs(a-b) for a,b in zip(q,self.plan['initial_q'])) > .003:
            raise ValueError('Pose changed after planning; no engagement')
        c = executor.c
        executor.c = replace(c,
            lower=tuple(max(lo,x-.2) for lo,x in zip(c.lower,q)),
            upper=tuple(min(hi,x+.2) for hi,x in zip(c.upper,q)),
            velocity=(.05,)*14, acceleration=(.1,)*14)

    def update(self, executor, now):
        if executor.phase == 'ACTIVE':
            if self.started is None:
                self.started = now
            executor.send_targets(self.target(now-self.started),now,now)


def make_plan(snapshot, output):
    import tempfile
    import numpy as np
    from virtual_arms import VirtualRobot
    from quest_bridge import configure_measured_pose_ik
    from arm_execution import Observation
    from hold_arms import run_hold
    from quest_bridge import load_quest_profile
    initial = np.asarray(snapshot['q'])
    with tempfile.TemporaryDirectory() as tmp:
        robot = VirtualRobot(Path(tmp), False)
        configure_measured_pose_ik(robot)
        robot.ik.opti.minimize(500*robot.ik.translational_cost+.1*robot.ik.smooth_cost)
        # This independent diagnostic retains its small initial-pose envelope.
        radius = RADIUS - 1e-6
        robot.ik.opti.subject_to(robot.ik.opti.bounded(
            initial-radius, robot.ik.var_q, initial+radius))
        robot.q = initial.copy()
        reference = robot.forward()
        seed = initial.copy()
        plan = dict(initial_q=initial.tolist(), q_path=[initial.tolist()], samples=[],
                    reference=[x.tolist() for x in reference], orientation='free',
                    height_m=.05, travel_s=TRAVEL, dwell_s=DWELL, active_s=ACTIVE)
        for h in np.linspace(.0025,.05,20):
            targets = [x.copy() for x in reference]
            for x in targets:
                x[2,3] += h
            robot.ik.smooth_filter._data_queue = []
            q,_ = robot.ik.solve_ik(*targets,seed.copy(),np.zeros(14))
            q = np.asarray(q)
            if (not robot.ik.opti.stats().get('success') or not np.isfinite(q).all()
                    or np.max(np.abs(q-initial)) > RADIUS
                    or np.any(q < robot.model.lowerPositionLimit)
                    or np.any(q > robot.model.upperPositionLimit)):
                raise ValueError('IK failure or joint envelope violation')
            robot.q = q
            actual = robot.forward()
            errors = [float(np.linalg.norm(a[:3,3]-t[:3,3])) for a,t in zip(actual,targets)]
            if max(errors) > .003:
                raise ValueError(f'5cm diagnostic path inaccurate at {h}: {errors}')
            angles = [float(np.arccos(np.clip((np.trace(a[:3,:3]@r[:3,:3].T)-1)/2,-1,1)))
                      for a,r in zip(actual,reference)]
            plan['samples'].append(dict(height_m=float(h),error_m=errors,rotation_rad=angles))
            plan['q_path'].append(q.tolist())
            seed=q.copy()
        # Simulate the real executor, including ramp, rate limiting and return,
        # using perfect feedback. This verifies commands, not physical tracking.
        profile, config = load_quest_profile()
        profile['hold_s'] = ACTIVE
        class SimIO:
            def __init__(self):
                self.t=0.; self.q=tuple(initial); self.commands=[]
            def check(self): pass
            def observation(self): return Observation(self.q,self.t,round(self.t*100000)+1,5)
            def sleep(self,dt): self.t+=dt
            def write(self,c):
                self.q=c.q; self.commands.append(c); return True
        io=SimIO()
        result=run_hold(io,config,profile,lambda:False,lambda row:None,
                        lambda:io.t,io.sleep,VerticalSource(plan))
        if result['error']:
            raise ValueError(f'Simulated executor failed: {result}')
        deltas=[]
        for c in io.commands:
            robot.q=np.asarray(c.q)
            deltas.append([a[:3,3]-r[:3,3] for a,r in zip(robot.forward(),reference)])
        deltas=np.asarray(deltas)
        if np.max(np.abs(deltas[:,:,:2]))>.003 or deltas[:,:,2].min()<-.003 or deltas[:,:,2].max()>.053:
            raise ValueError('Interpolated command leaves vertical corridor')
        if min(deltas[:,:,2].max(axis=0))<.047:
            raise ValueError('Interpolated command did not reach 5cm within 3mm')
        if max(abs(a-b) for a,b in zip(io.commands[-1].q,initial))>1e-4:
            raise ValueError('Simulated command did not return to initial joints')
        plan['simulation']=dict(result=result, max_lift_m=deltas[:,:,2].max(axis=0).tolist(),
                                 final_delta_m=deltas[-1].tolist(),command_frames=len(io.commands))
        save(output,plan)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface',default='enp130s0')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan-snapshot',type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--scene-confirmation')
    args=parser.parse_args()
    if args.plan_snapshot:
        if args.execute: parser.error('Planning subprocess cannot execute')
        make_plan(json.loads(args.plan_snapshot.read_text()),args.output)
        return 0
    if args.execute and not (args.scene_confirmation or '').strip():
        parser.error('Current scene confirmation required')
    from dataclasses import asdict
    from hold_arms import DDSHoldIO, run_hold
    from quest_bridge import load_quest_profile
    profile,config=load_quest_profile(); profile['hold_s']=ACTIVE
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'source').mkdir()
    for name in ('vertical_probe.py','quest_bridge.py','hold_arms.py','arm_execution.py','virtual_arms.py'):
        (args.output/'source'/name).write_bytes((ROOT/'scripts/unitree_g1'/name).read_bytes())
    save(args.output/'session.json',dict(scene_confirmation=args.scene_confirmation,
        hardware_motion_enabled=args.execute,profile=profile,height_m=.05,orientation='free',
        joint_radius_rad=.2,velocity_rad_s=.05,acceleration_rad_s2=.1,
        phases_s=dict(engage=2,up=TRAVEL,top=DWELL,down=TRAVEL,settle=DWELL,unload=2,tail=.3),
        scope='Single diagnostic; no Quest input, no gravity feedforward, no automatic retry'))
    stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:stop.set())
    io=None; result=dict(error='initialization_failed',command_frames=0,unload_submitted=False)
    try:
        io=DDSHoldIO(args.interface,args.execute,profile);io.preflight()
        save(args.output/'snapshot.json',asdict(io.observation()))
        with (args.output/'planning.log').open('x') as log:
            proc=subprocess.Popen([sys.executable,__file__,'--plan-snapshot',str(args.output/'snapshot.json'),
                                    '--output',str(args.output/'plan.json')],stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+45
            try:
                while proc.poll() is None:
                    io.check()
                    if stop.is_set() or time.monotonic()>deadline: raise RuntimeError('Planning stopped or timed out')
                    time.sleep(.02)
                if proc.returncode: raise ValueError('Offline plan rejected; see planning.log')
            finally:
                if proc.poll() is None: proc.terminate();proc.wait(timeout=5)
        plan=json.loads((args.output/'plan.json').read_text())
        if args.execute:
            io.preflight()
            if stop.is_set(): raise RuntimeError('Stopped before engagement')
            if max(abs(a-b) for a,b in zip(io.observation().q,plan['initial_q']))>.003:
                raise ValueError('Pose changed after planning; no motor commands')
            print('Plan passed. 2s engagement, 12s up 5cm, 2s dwell, 12s return, 2s settle, 2s unload.',flush=True)
            io.open_writer()
            with (args.output/'commands.jsonl').open('x',buffering=1) as log:
                result=run_hold(io,config,profile,stop.is_set,
                    lambda row:log.write(json.dumps(row,allow_nan=False)+'\n'),target_source=VerticalSource(plan))
        else:
            result=dict(error=None,command_frames=0,unload_submitted=False,planning_passed=True)
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
    finally:
        if io:io.close()
        result.update(hardware_motion_enabled=args.execute,physical_unload_verified=False)
        save(args.output/'summary.json',result);print(json.dumps(result,indent=2),flush=True)
    return 1 if result['error'] else 0


if __name__=='__main__':
    raise SystemExit(main())
