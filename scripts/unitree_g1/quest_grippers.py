"""Optional Dex1 control merged into the Quest arm publisher (no second writer)."""
import json
import math
from pathlib import Path
import time

from arm_execution import bounded_joint_step, encode_debug_command
from hold_arms import DDSHoldIO

ROOT=Path(__file__).resolve().parents[2]


def gripper_settings():
    path=ROOT/'artifacts/unitree_g1/gripper_negative_open_live_20261009_001.json'
    report=json.loads(path.read_text())
    motors={m['index']:m for m in report['motors']}
    closed=[motors[i]['initial_q'] for i in (31,33)]
    if any(not math.isfinite(q) or not -.05<=q<=.05 for q in closed):
        raise ValueError('Invalid recorded closed reference')
    open_path=ROOT/'artifacts/unitree_g1/gripper_absolute_open_live_20261009_001.json'
    open_report=json.loads(open_path.read_text())
    if (open_report.get('error') is not None or
            open_report.get('absolute_target') != [-5.4,-5.4] or
            open_report.get('full_open_visually_confirmed') is not True):
        raise ValueError('Missing successful visually confirmed full-open trial')
    return dict(indices=[31,33],closed=closed,opened=[-5.4,-5.4],
                kp=5.,kd=.05,velocity=.6,acceleration=.2,tracking_error=.15,
                timeout=.1,pickup_trigger=.95,
                mapping='Squeeze trigger fully once after ACTIVE; then release=open, squeeze=close',
                reference=str(path.relative_to(ROOT)),open_reference=str(open_path.relative_to(ROOT)))


def trigger_pair(values):
    if not isinstance(values,(list,tuple)) or len(values)!=2:
        raise ValueError('Missing left/right trigger values')
    if any(isinstance(x,bool) or not isinstance(x,(float,int)) or
           not math.isfinite(x) or not 0<=x<=1 for x in values):
        raise ValueError('Invalid trigger values')
    return tuple(values)


def raw_triggers(event):
    return trigger_pair([event['raw'][side+'State']['triggerValue'] for side in ('left','right')])


class GripperFollower:
    def __init__(self,settings):
        self.c=settings
        self.initialized=False
        self.fault=None
        self.last_sent=None

    def feedback(self,telemetry,now):
        if telemetry is None or not 0<=now-telemetry['received']<=self.c['timeout']:
            raise ValueError('Stale gripper feedback')
        motors={m['index']:m for m in telemetry['motors']}
        selected=[motors[i] for i in self.c['indices']]
        if any(m['motorstate']!=0 or not all(math.isfinite(m[k]) for k in ('q','dq')) for m in selected):
            raise ValueError('Invalid gripper feedback')
        return tuple(m['q'] for m in selected),tuple(m['dq'] for m in selected)

    def initialize(self,telemetry,now):
        q,dq=self.feedback(telemetry,now)
        if max(map(abs,dq))>.1:
            raise ValueError('Grippers not stationary at engagement')
        if any(not lo-.01<=x<=hi+.01 for x,lo,hi in zip(q,self.c['opened'],self.c['closed'])):
            raise ValueError('Gripper initial pose outside tested range')
        self.q=self.target=q
        self.lower=tuple(min(lo,x) for lo,x in zip(self.c['opened'],q))
        self.upper=tuple(max(hi,x) for hi,x in zip(self.c['closed'],q))
        self.v=(0.,0.)
        self.picked_up=[False,False]
        self.input_time=None
        self.active_since=None
        self.last_time=now
        self.initialized=True

    def update(self,triggers,stamp,now):
        values=trigger_pair(triggers)
        if not math.isfinite(stamp) or not 0<=now-stamp<=self.c['timeout']:
            raise ValueError('Stale gripper input')
        if self.input_time is not None and stamp<=self.input_time:
            raise ValueError('Nonmonotonic gripper input')
        self.input_time=stamp
        target=list(self.target)
        for i,t in enumerate(values):
            if t>=self.c['pickup_trigger']:self.picked_up[i]=True
            if self.picked_up[i]:
                target[i]=self.c['opened'][i]+t*(self.c['closed'][i]-self.c['opened'][i])
        self.target=tuple(target)

    def step(self,command,telemetry,now,full_arm_kp):
        if not self.initialized:raise ValueError('Grippers not initialized')
        if self.fault is None:
            try:
                feedback,_=self.feedback(telemetry,now)
                errors=[abs(q-f) for q,f in zip(self.q,feedback)]
                if max(errors)>self.c['tracking_error']:
                    raise ValueError('Gripper tracking_error')
                dt=now-self.last_time
                if not 0<=dt<=.05:raise ValueError('Gripper servo deadline')
                if command.phase=='ACTIVE':
                    if self.active_since is None:self.active_since=now
                    stamp=self.input_time if self.input_time is not None else self.active_since
                    if now-stamp>self.c['timeout']:raise ValueError('Stale gripper target')
                    if dt>0:
                        steps=[bounded_joint_step(q,t,v,self.c['velocity'],self.c['acceleration'],lo,hi,dt)
                               for q,t,v,lo,hi in zip(self.q,self.target,self.v,self.lower,self.upper)]
                        self.q=tuple(s[0] for s in steps)
                        self.v=tuple(s[1] for s in steps)
                else:
                    self.v=(0.,0.)  # Exit freezes the last setpoint; never auto-close.
            except (ValueError,KeyError,TypeError) as exc:
                # Raise once so run_hold enters its bounded finally-unload.
                # Subsequent release writes must still be possible with bad feedback.
                self.fault=dict(reason=str(exc),phase=command.phase,time=now,
                                command_q=self.q,motor_indices=self.c['indices'])
                raise ValueError('gripper: '+str(exc)) from exc
        self.last_time=now
        scale=command.kp[0]/full_arm_kp
        if not 0<=scale<=1.+1e-9:raise ValueError('Invalid arm/gripper ramp scale')
        self.last_sent=dict(q=self.q,kp=[self.c['kp']*scale]*2,kd=[self.c['kd']*scale]*2,
                            target=self.target,picked_up=list(self.picked_up),
                            phase=command.phase,input_time=self.input_time)
        return self.last_sent


def encode_combined(command,gripper):
    msg=encode_debug_command(command)
    for i in (29,30,32,34):msg.motor_cmd[i].mode=0
    for j,i in enumerate((31,33)):
        m=msg.motor_cmd[i]
        m.mode=1
        m.q=gripper['q'][j]
        m.kp=gripper['kp'][j]
        m.kd=gripper['kd'][j]
    from unitree_sdk2py.utils.crc import CRC
    msg.crc=CRC().Crc(msg)
    return msg


class QuestGripperIO(DDSHoldIO):
    def __init__(self,interface,live,profile,settings):
        self.grippers=GripperFollower(settings)
        self.full_arm_kp=profile['controller']['kp'][0]
        super().__init__(interface,live,profile)

    def preflight(self):
        super().preflight()
        self.grippers.initialize(self.telemetry(),time.monotonic())

    def initialize_grippers(self,now):
        self.grippers.initialize(self.telemetry(),now)

    def set_gripper_targets(self,values,stamp,now):
        self.grippers.update(values,stamp,now)

    def write(self,command):
        packet=self.grippers.step(command,self.telemetry(),time.monotonic(),self.full_arm_kp)
        if not self.live:return True  # Shadow must never create/write a DDS publisher.
        if self.writer is None or self.matched==0:return False
        self.writer.write(encode_combined(command,packet))
        return True

    def decorate_log(self,row):
        return dict(row,gripper_command=self.grippers.last_sent,gripper_fault=self.grippers.fault)
