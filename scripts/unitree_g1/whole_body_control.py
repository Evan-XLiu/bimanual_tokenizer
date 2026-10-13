"""Offline whole-body execution core. No DDS publisher or live entry point.

Inference and upper-body producers feed immutable snapshots. One servo owner
composes all 35 slots. Upper task termination holds its last command while legs
continue; body faults latch and require an external, validated stop procedure.
"""
from dataclasses import dataclass
import math
import threading

from arm_execution import Command, bounded_joint_step, vector
from lower_body_policy import gravity
from quest_grippers import GripperFollower


class BodyFault(RuntimeError):
    """No fallback is claimed to balance a robot with invalid body control."""


@dataclass(frozen=True)
class BodyState:
    stamp: float
    tick: int
    mode_machine: int
    q: tuple
    dq: tuple
    quaternion: tuple
    gyro: tuple
    motorstate: tuple = (0,)*35


@dataclass(frozen=True)
class PolicyResult:
    stamp: float
    source_stamp: float
    sequence: int
    target: tuple
    observation: tuple
    action: tuple
    inference_s: float
    waist_target: tuple | None = None


@dataclass(frozen=True)
class WholeBodyFrame:
    time: float
    mode_machine: int
    q: tuple
    kp: tuple
    kd: tuple
    upper_phase: str
    upper_reason: str
    policy_sequence: int
    clipped_leg_indices: tuple

    def action16(self):
        return self.q[15:22]+(self.q[31],)+self.q[22:29]+(self.q[33],)


class WholeBodyControl:
    """Single servo-thread owner; update_policy is safe from a worker thread.

stop_upper is an immediate protective setpoint freeze, not a smooth physical
stop guarantee. Body fault handling deliberately does not send zero-gain tails.
"""
    def __init__(self, arm_config, leg_config, body_limits, gripper_config):
        self.arm = arm_config
        self.leg = dict(leg_config)
        for key, size in (('kps', 12), ('kds', 12), ('arm_waist_kps', 17), ('arm_waist_kds', 17)):
            self.leg[key] = vector(self.leg[key], size)
            if min(self.leg[key]) <= 0:
                raise ValueError('Body gains must be positive')
        self.lower = vector(body_limits[0], 29)
        self.upper = vector(body_limits[1], 29)
        if any(lo >= hi for lo, hi in zip(self.lower, self.upper)):
            raise ValueError('Invalid body bounds')
        self.gripper = GripperFollower(gripper_config)
        self.phase = 'DISARMED'
        self.reason = ''
        self.body_fault = None
        self.policy = None
        self._lock = threading.Lock()
        self.owner = None
        self.state = None
        self.last_step = None
        self.target_stamp = None
        self.preopen_started = None

    def _owner(self):
        if self.owner != threading.get_ident():
            raise RuntimeError('Only the servo owner may mutate upper control or compose frames')

    def _fault(self, reason):
        if self.body_fault is None:
            self.body_fault = reason
        raise BodyFault(self.body_fault)

    def _state(self, state, now):
        if self.body_fault:
            raise BodyFault(self.body_fault)
        try:
            if not math.isfinite(now) or not math.isfinite(state.stamp) or not 0 <= now-state.stamp <= .04:
                raise ValueError('feedback_deadline')
            if state.mode_machine != 5 or type(state.tick) is not int or not 0 <= state.tick < 2**32:
                raise ValueError('body_mode_or_tick')
            q, dq = vector(state.q, 35), vector(state.dq, 35)
            vector(state.gyro, 3)
            if len(state.motorstate) != 35 or any(type(x) is not int for x in state.motorstate):
                raise ValueError('invalid_motor_status')
            if any(state.motorstate[:15]):
                raise ValueError('lower_body_motor_fault')
            g = gravity(state.quaternion)
            if -float(g[2]) < math.cos(.35):
                raise ValueError('body_tilt')
            if any(not lo-.02 <= x <= hi+.02 for x, lo, hi in zip(q[:29], self.lower, self.upper)):
                raise ValueError('body_joint_limit')
            if self.state is not None:
                delta = (state.tick-self.state.tick) % 2**32
                if delta >= 2**31 or state.stamp < self.state.stamp:
                    raise ValueError('feedback_clock_reset')
                if delta == 0 and state.stamp != self.state.stamp:
                    raise ValueError('frozen_tick_refreshed')
                if delta > 0 and state.stamp <= self.state.stamp:
                    raise ValueError('feedback_timestamp_not_advancing')
            self.state = BodyState(state.stamp, state.tick, state.mode_machine, q, dq,
                                   vector(state.quaternion, 4), vector(state.gyro, 3), tuple(state.motorstate))
        except (ValueError, TypeError) as exc:
            self._fault(str(exc))

    def initialize(self, state, now):
        if self.phase != 'DISARMED':
            raise RuntimeError('No implicit rearming')
        self.owner = threading.get_ident()
        self._state(state, now)
        self.arm_q = self.arm_target = self.state.q[15:29]
        self.arm_v = (0.,)*14
        self.waist_q = self.state.q[12:15]
        self.waist_v = (0.,)*3
        self.gripper.initialize(self._telemetry(), now)
        self.phase = 'STANDING'
        self.last_step = now

    def _telemetry(self):
        return dict(received=self.state.stamp, motors=[dict(index=i, q=q, dq=dq, motorstate=self.state.motorstate[i])
                    for i, (q, dq) in enumerate(zip(self.state.q, self.state.dq))])

    def update_policy(self, result):
        try:
            if (not all(math.isfinite(t) for t in (result.stamp, result.source_stamp, result.inference_s))
                    or not 0 <= result.inference_s <= .02 or result.stamp < result.source_stamp
                    or type(result.sequence) is not int or result.sequence < 1):
                raise ValueError('invalid_inference_timing')
            validated = PolicyResult(result.stamp, result.source_stamp, result.sequence,
                                     vector(result.target, 12), vector(result.observation, 47),
                                     vector(result.action, 12), result.inference_s,
                                     None if result.waist_target is None else vector(result.waist_target, 3))
            if validated.waist_target is not None and any(abs(x) > .2 for x in validated.waist_target):
                raise ValueError('waist_compensation_bound')
            with self._lock:
                if self.body_fault:
                    raise BodyFault(self.body_fault)
                if self.policy is not None and (result.sequence <= self.policy.sequence
                        or result.source_stamp <= self.policy.source_stamp or result.stamp <= self.policy.stamp):
                    raise ValueError('nonmonotonic_policy_result')
                self.policy = validated
        except (ValueError, TypeError) as exc:
            self._fault(str(exc))

    def preopen(self, now):
        self._owner()
        if self.phase != 'STANDING' or self.body_fault:
            raise RuntimeError('Preopen requires initialized standing state')
        self.phase = 'PREOPEN'
        self.preopen_started = now

    def begin_task(self):
        self._owner()
        if self.phase != 'READY' or self.body_fault:
            raise RuntimeError('Task requires completed integrated preopen')
        self.phase = 'ACTIVE'
        self.target_stamp = None
        self.active_started = self.last_step

    def upper_target(self, q, triggers, stamp, now):
        self._owner()
        if self.phase != 'ACTIVE':
            raise RuntimeError('Upper task is not active')
        try:
            q = vector(q)
            if not math.isfinite(stamp) or not 0 <= now-stamp <= .2:
                raise ValueError('upper_input_deadline')
            if self.target_stamp is not None and stamp <= self.target_stamp:
                raise ValueError('nonmonotonic_upper_target')
            if any(not lo <= x <= hi for x, lo, hi in zip(q, self.arm.lower, self.arm.upper)):
                raise ValueError('upper_target_limit')
            self.gripper.update(triggers, stamp, now)
            self.arm_target, self.target_stamp = q, stamp
        except (ValueError, TypeError) as exc:
            self.stop_upper(str(exc))

    def stop_upper(self, reason):
        self._owner()
        if self.phase == 'DISARMED':
            raise RuntimeError('No initialized hold pose')
        self.phase, self.reason = 'UPPER_HOLD', reason
        self.arm_target = self.arm_q
        self.arm_v = (0.,)*14
        self.gripper.target = self.gripper.q
        self.gripper.v = (0., 0.)

    def step(self, state, now):
        self._owner()
        self._state(state, now)
        if any(self.state.motorstate[15:29]):
            self.stop_upper('upper_motor_fault')
        dt = now-self.last_step
        if not 0 < dt <= .02:
            self._fault('servo_deadline')
        with self._lock:
            policy = self.policy
        if policy is None or not 0 <= now-policy.stamp <= .04 or not 0 <= now-policy.source_stamp <= .04:
            self._fault('policy_deadline')
        if policy.waist_target is not None:
            steps = [bounded_joint_step(q, target, v, .3, .6, lo, hi, dt)
                     for q, target, v, lo, hi in zip(self.waist_q, policy.waist_target, self.waist_v,
                                                    self.lower[12:15], self.upper[12:15])]
            self.waist_q, self.waist_v = tuple(x[0] for x in steps), tuple(x[1] for x in steps)
        if self.phase == 'ACTIVE':
            stamp = self.target_stamp if self.target_stamp is not None else self.active_started
            if now-stamp > .2:
                self.stop_upper('upper_input_deadline')
        if self.phase in ('PREOPEN', 'ACTIVE'):
            if any(abs(q-measured) > bound for q, measured, bound in
                   zip(self.arm_q, self.state.q[15:29], self.arm.tracking_error)):
                self.stop_upper('arm_tracking_error')
        if self.phase == 'ACTIVE':
            steps = [bounded_joint_step(q, target, v, vmax, amax, lo, hi, dt)
                     for q, target, v, vmax, amax, lo, hi in zip(self.arm_q, self.arm_target, self.arm_v,
                         self.arm.velocity, self.arm.acceleration, self.arm.lower, self.arm.upper)]
            self.arm_q, self.arm_v = tuple(x[0] for x in steps), tuple(x[1] for x in steps)
        if self.phase == 'PREOPEN':
            if now-self.preopen_started > 10.:
                self.stop_upper('preopen_deadline')
            else:
                self.gripper.update([0., 0.], now, now)
        command = Command(self.arm_q, self.arm.kp, self.arm.kd, 5,
                          'ACTIVE' if self.phase in ('PREOPEN', 'ACTIVE') else 'STANDING', self.reason)
        try:
            packet = self.gripper.step(command, self._telemetry(), now, self.arm.kp[0])
        except ValueError as exc:
            self.stop_upper(str(exc))
            packet = dict(q=self.gripper.q, kp=[self.gripper.c['kp']]*2, kd=[self.gripper.c['kd']]*2)
        if self.phase == 'PREOPEN' and all(abs(q-lo) <= .01 and abs(v) < 1e-6 and abs(f-lo) <= .15
                for q, v, f, lo in zip(self.gripper.q, self.gripper.v,
                    (self.state.q[31], self.state.q[33]), self.gripper.c['opened'])):
            self.phase = 'READY'
        q, kp, kd = [0.]*35, [0.]*35, [0.]*35
        clipped = []
        for i, raw in enumerate(policy.target):
            q[i] = max(self.lower[i], min(self.upper[i], raw))
            if q[i] != raw:
                clipped.append(i)
            kp[i], kd[i] = self.leg['kps'][i], self.leg['kds'][i]
        q[12:15], kp[12:15], kd[12:15] = self.waist_q, self.leg['arm_waist_kps'][:3], self.leg['arm_waist_kds'][:3]
        q[15:29], kp[15:29], kd[15:29] = self.arm_q, self.arm.kp, self.arm.kd
        for j, i in enumerate((31, 33)):
            q[i], kp[i], kd[i] = packet['q'][j], packet['kp'][j], packet['kd'][j]
        self.last_step = now
        self.last_policy_used = policy
        return WholeBodyFrame(now, 5, tuple(q), tuple(kp), tuple(kd), self.phase,
                              self.reason, policy.sequence, tuple(clipped))


def encode_whole_body(frame):
    """Offline packet construction only; no writer, channel or implicit live mode."""
    q, kp, kd = (vector(getattr(frame, name), 35) for name in ('q', 'kp', 'kd'))
    if frame.mode_machine != 5 or min(kp+kd) < 0:
        raise ValueError('Invalid whole-body command')
    if any(kp[i] <= 0 or kd[i] <= 0 for i in range(29)):
        raise ValueError('Whole-body frame must retain body gains')
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    msg = unitree_hg_msg_dds__LowCmd_()
    msg.mode_pr, msg.mode_machine = 0, frame.mode_machine
    for i in list(range(29))+[31, 33]:
        motor = msg.motor_cmd[i]
        motor.mode, motor.q, motor.kp, motor.kd = 1, q[i], kp[i], kd[i]
    msg.crc = CRC().Crc(msg)
    return msg
