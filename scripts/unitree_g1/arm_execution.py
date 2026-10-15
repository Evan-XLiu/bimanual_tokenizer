"""Deterministic G1 arm executor, with an injectable transport and no DDS startup.

Candidate protocol: debug rt/lowcmd, non-arm joints passive. NOT arm_sdk.
This is not a validated hardware profile. See docs/workflows/g1_arm_execution.md.
All times must use the same monotonic clock. The owner calls step at servo rate.
"""
from dataclasses import dataclass
import math

ARM_INDICES = tuple(range(15, 29))  # Official arm7 reference; must be verified.


def vector(value, size=14):
    result = tuple(float(x) for x in value)
    if len(result) != size or not all(math.isfinite(x) for x in result):
        raise ValueError(f'Expected {size} finite values')
    return result


def braking_speed(distance, acceleration, dt):
    """Speed whose next step AND continuous stopping distance fit in distance.

    u*dt + u**2/(2*a) <= distance. The rationalized root avoids
    cancellation near a boundary. Semi-implicit subsequent braking travels
    no farther than the continuous stopping distance, even with varying dt.
    """
    if distance <= 0.:
        return 0.
    adt = acceleration*dt
    return 2*acceleration*distance / (math.sqrt(adt*adt+2*acceleration*distance)+adt)


def bounded_joint_step(q, target, v, vmax, amax, lower, upper, dt):
    error = target-q
    desired = math.copysign(min(vmax, braking_speed(abs(error), amax, dt)), error) if error else 0.
    # Intersect acceleration bounds with velocities that retain braking room
    # at BOTH hard boundaries, independently of a changing operator target.
    low = max(-vmax, v-amax*dt, -braking_speed(q-lower, amax, dt))
    high = min(vmax, v+amax*dt, braking_speed(upper-q, amax, dt))
    if low > high:
        raise ValueError('No feasible bounded braking step')
    vn = max(low, min(high, desired))
    return q+vn*dt, vn


@dataclass(frozen=True)
class Config:
    # No default hardware gains or limits: these must come from a verified profile.
    lower: tuple
    upper: tuple
    kp: tuple
    kd: tuple
    velocity: tuple
    acceleration: tuple
    tracking_error: tuple
    ramp_seconds: float
    state_timeout: float
    target_timeout: float
    max_step_seconds: float

    def __post_init__(self):
        for key in ('lower', 'upper', 'kp', 'kd', 'velocity', 'acceleration', 'tracking_error'):
            object.__setattr__(self, key, vector(getattr(self, key)))
        if any(a >= b for a, b in zip(self.lower, self.upper)):
            raise ValueError('Invalid joint limits')
        for key in ('kp', 'kd', 'velocity', 'acceleration', 'tracking_error'):
            if min(getattr(self, key)) <= 0:
                raise ValueError(f'{key} must be positive')
        for key in ('ramp_seconds', 'state_timeout', 'target_timeout', 'max_step_seconds'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive and finite')


@dataclass(frozen=True)
class Observation:
    q: tuple
    received: float  # Timestamp of advancing device tick, not a polled cache.
    tick: int
    mode_machine: int


@dataclass(frozen=True)
class Command:
    q: tuple
    kp: tuple
    kd: tuple
    mode_machine: int
    phase: str
    reason: str


@dataclass(frozen=True)
class FaultEvidence:
    reason: str
    phase: str  # Phase BEFORE fault handling, not the resulting RELEASING.
    last_step_time: float | None
    gain_scale: float  # Last submitted/requested scale before this step's fade.
    feedback_received: float | None
    tracking_error_rad: tuple
    tracking_exceeded_arm_indices: tuple


class ArmExecution:
    """Pure state machine. emit(Command) returns True only on successful submission.

    No threads, controller construction, mode switching or implicit auto-rearming.
    stop/release freeze the last commanded pose and fade gains to zero. Faults
    take the same candidate unload path and latch; actual stop behavior requires
    hardware validation. Lost transport cannot guarantee delivery of that path.
    """
    def __init__(self, config, emit):
        self.c, self.emit = config, emit
        self.phase, self.reason = 'DISARMED', ''
        self.observation = None
        self.last_tick = None
        self.last_step = None
        self.scale = 0.0
        self.q = self.target = None
        self.v = (0.,) * 14
        self.target_time = None
        self.fault = False
        self.first_fault = None
        self.release_reason = None

    def read_observation(self):
        return self.observation

    def observe(self, observation):
        try:
            q = vector(observation.q)
            if not math.isfinite(observation.received):
                raise ValueError('Invalid state timestamp')
            if not isinstance(observation.tick, int) or not 0 <= observation.tick < 2**32:
                raise ValueError('Invalid device tick')
            if not isinstance(observation.mode_machine, int) or not 0 <= observation.mode_machine <= 255:
                raise ValueError('Invalid machine variant')
            if self.observation is not None:
                delta = (observation.tick - self.last_tick) % 2**32
                if delta == 0:
                    return  # Frozen tick never refreshes freshness.
                if delta >= 2**31 or observation.received <= self.observation.received:
                    raise ValueError('State clock reset or out of order')
            self.observation = Observation(q, observation.received, observation.tick, observation.mode_machine)
            self.last_tick = observation.tick
        except (ValueError, TypeError):
            self._stop('invalid_state', True)
            raise

    def _fresh(self, now):
        return self.observation is not None and 0 <= now - self.observation.received <= self.c.state_timeout

    def request_control(self, now):
        if self.phase != 'DISARMED' or self.fault:
            raise RuntimeError('Executor must be disarmed without a latched fault')
        if not math.isfinite(now) or not self._fresh(now):
            raise RuntimeError('Fresh advancing feedback required')
        q = self.observation.q
        self._check_limits(q)
        self.q = self.target = q
        self.mode_machine = self.observation.mode_machine
        self.last_step = self.target_time = now
        self.v = (0.,) * 14
        self.scale = 0.
        self.phase, self.reason = 'ENGAGING', ''
        self.release_reason = None
        self._emit()  # Exact measured pose, zero gains, no hidden zero target.

    def _check_limits(self, q):
        if any(not lo <= x <= hi for lo, x, hi in zip(self.c.lower, q, self.c.upper)):
            raise ValueError('Joint target outside profile limits')

    def send_targets(self, q, received, now):
        if self.phase != 'ACTIVE':
            raise RuntimeError('Targets accepted only after engagement')
        try:
            q = vector(q)
            self._check_limits(q)
            if not all(math.isfinite(t) for t in (received, now)) or not 0 <= now-received <= self.c.target_timeout:
                raise ValueError('Stale or future target')
            if received <= self.target_time:
                raise ValueError('Nonmonotonic target')
        except (ValueError, TypeError):
            self._stop('invalid_target', True)
            raise
        self.target, self.target_time = q, received

    def _stop(self, reason, fault=False):
        if fault and self.first_fault is None:
            errors = (() if self.q is None or self.observation is None else
                      tuple(abs(a-b) for a, b in zip(self.q, self.observation.q)))
            self.first_fault = FaultEvidence(
                reason, self.phase, self.last_step, self.scale,
                None if self.observation is None else self.observation.received,
                errors, tuple(ARM_INDICES[i] for i, (error, limit) in
                              enumerate(zip(errors, self.c.tracking_error)) if error > limit))
        if self.phase in ('ENGAGING', 'ACTIVE'):
            self.release_reason = reason
        self.fault |= fault
        # Preserve the first fault even if feedback later expires or an operator
        # requests release. Existing faults remain latched throughout unloading.
        self.reason = self.first_fault.reason if self.first_fault else reason
        if self.phase in ('ENGAGING', 'ACTIVE', 'RELEASING'):
            self.phase = 'RELEASING'
            self.target = self.q
            self.v = (0.,) * 14
        elif self.fault:
            self.phase = 'FAULT'

    def stop(self):
        self._stop('operator_stop')

    def release_control(self):
        self._stop('operator_release')

    def _emit(self):
        command = Command(self.q, tuple(self.scale*x for x in self.c.kp),
                          tuple(self.scale*x for x in self.c.kd),
                          self.mode_machine, self.phase, self.reason)
        try:
            if self.emit(command) is not True:
                raise OSError('Transport did not confirm submission')
        except Exception:
            self._stop('transport_failure', True)
            self.phase = 'FAULT'
            raise
        return command

    def step(self, now):
        if self.phase in ('DISARMED', 'FAULT'):
            return None
        if not math.isfinite(now) or now <= self.last_step:
            self._stop('invalid_clock', True)
            raise ValueError('Servo clock must advance monotonically')
        dt = now - self.last_step
        self.last_step = now
        if dt > self.c.max_step_seconds:
            self._stop('servo_deadline', True)
        dt = min(dt, self.c.max_step_seconds)
        if not self._fresh(now):
            self._stop('stale_state', True)
        elif self.observation.mode_machine != self.mode_machine:
            self._stop('mode_machine_changed', True)
        elif self.phase in ('ENGAGING', 'ACTIVE') and any(
                abs(a-b) > limit for a, b, limit in zip(self.q, self.observation.q, self.c.tracking_error)):
            self._stop('tracking_error', True)
        if self.phase == 'ENGAGING':
            self.scale = min(1., self.scale + dt/self.c.ramp_seconds)
            if self.scale >= 1.-1e-12:
                self.scale = 1.
                self.phase, self.target_time = 'ACTIVE', now
        elif self.phase == 'ACTIVE':
            if now-self.target_time > self.c.target_timeout:
                self._stop('stale_target', True)
            else:
                q_next, v_next = [], []
                try:
                    for values in zip(self.q, self.target, self.v, self.c.velocity,
                                      self.c.acceleration, self.c.lower, self.c.upper):
                        qn, vn = bounded_joint_step(*values, dt)
                        q_next.append(qn)
                        v_next.append(vn)
                    self._check_limits(q_next)
                except ValueError:
                    self._stop('trajectory_limit', True)
                else:
                    self.q, self.v = tuple(q_next), tuple(v_next)
        if self.phase == 'RELEASING':
            self.scale = max(0., self.scale-dt/self.c.ramp_seconds)
            if self.scale < 1e-12:
                self.scale = 0.
        command = self._emit()
        if self.phase == 'RELEASING' and self.scale == 0:
            self.phase = 'FAULT' if self.fault else 'DISARMED'
        return command


def encode_debug_command(command):
    """Construct an SDK message without initializing DDS or sending anything.

    Full-body passive command outside arms; NOT preservation of an existing
    waist/leg controller. Use only after the debug/passive contract is verified.
    """
    q, kp, kd = vector(command.q), vector(command.kp), vector(command.kd)
    if min(kp+kd) < 0:
        raise ValueError('Negative gains')
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    msg = unitree_hg_msg_dds__LowCmd_()
    msg.mode_pr, msg.mode_machine = 0, command.mode_machine
    for motor in msg.motor_cmd:
        motor.mode = 1
    for i, index in enumerate(ARM_INDICES):
        motor = msg.motor_cmd[index]
        motor.q, motor.kp, motor.kd = q[i], kp[i], kd[i]
    msg.crc = CRC().Crc(msg)
    return msg
