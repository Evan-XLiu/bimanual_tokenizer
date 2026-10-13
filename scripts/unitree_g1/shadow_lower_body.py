"""Read real LowState and evaluate the pinned policy; never publish commands.

This diagnostic does not validate closed-loop balance, contact, localization,
authority transfer, or physical stopping. No execute option is provided.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import threading
import time

from lower_body_policy import LowerBodyPolicy, finite, gravity
from whole_body_control import BodyState


def decode_state(msg, stamp):
    if len(msg.motor_state) != 35 or msg.mode_machine != 5 or msg.mode_pr != 0:
        raise ValueError('unsupported_feedback_layout_or_mode')
    if not math.isfinite(stamp):
        raise ValueError('invalid_receive_time')
    q = tuple(float(m.q) for m in msg.motor_state)
    dq = tuple(float(m.dq) for m in msg.motor_state)
    finite(q, 35)
    finite(dq, 35)
    quat = tuple(float(x) for x in msg.imu_state.quaternion)
    gyro = tuple(float(x) for x in msg.imu_state.gyroscope)
    gravity(quat)
    finite(gyro, 3)
    tick = int(msg.tick)
    if not 0 <= tick < 2**32:
        raise ValueError('invalid_tick')
    return BodyState(stamp, tick, int(msg.mode_machine), q, dq, quat, gyro,
                     tuple(int(m.motorstate) for m in msg.motor_state))


class FeedbackMailbox:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = None
        self.error = None
        self.received = 0
        self.duplicates = 0

    def receive(self, msg):
        self.accept(msg, time.monotonic())

    def accept(self, msg, stamp):
        with self.lock:
            if self.error:
                return
            try:
                state = decode_state(msg, stamp)
                self.received += 1
                if self.state is not None:
                    delta = (state.tick-self.state.tick) % 2**32
                    if delta >= 2**31 or stamp <= self.state.stamp:
                        raise ValueError('feedback_clock_reset')
                    if delta == 0:
                        self.duplicates += 1
                        return  # Never refresh freshness on a frozen device tick.
                self.state = state
            except Exception as exc:
                self.error = str(exc)

    def latest(self, now):
        with self.lock:
            if self.error:
                raise ValueError(self.error)
            if self.state is None or not 0 <= now-self.state.stamp <= .04:
                raise ValueError('feedback_deadline')
            return self.state


def check_posture(state):
    if any(state.motorstate[:29]):
        raise ValueError('body_motor_status_nonzero')
    tilt = math.acos(max(-1., min(1., -float(gravity(state.quaternion)[2]))))
    if tilt > .35:
        raise ValueError(f'body_tilt: {tilt:.6f} rad > 0.35')
    return tilt


def run(policy, mailbox, duration, emit):
    started = time.monotonic()
    deadline = started
    count = 0
    max_inference = max_age = max_tilt = max_target_delta = 0.
    reason = None
    try:
        while time.monotonic()-started < duration:
            time.sleep(max(0., deadline-time.monotonic()))
            now = time.monotonic()
            if now-deadline > .02:
                raise ValueError('shadow_schedule_deadline')
            state = mailbox.latest(now)
            tilt = check_posture(state)
            before = time.monotonic()
            target = policy.step(state.q[:12], state.dq[:12], state.quaternion, state.gyro)
            after = time.monotonic()
            elapsed = after-before
            delta = max(abs(float(t)-q) for t, q in zip(target, state.q))
            max_inference = max(max_inference, elapsed)
            max_age = max(max_age, after-state.stamp)
            max_tilt = max(max_tilt, tilt)
            max_target_delta = max(max_target_delta, delta)
            emit(dict(type='shadow_policy', state=asdict(state), observation=policy.last_observation.tolist(),
                      action=policy.previous.tolist(), unsent_target=target.tolist(), inference_s=elapsed,
                      tilt_rad=tilt, max_target_delta_rad=delta))
            count += 1
            if elapsed > .02 or after-state.stamp > .04:
                raise ValueError('policy_deadline')
            deadline += .02
    except (Exception, KeyboardInterrupt) as exc:
        reason = str(exc) or type(exc).__name__
    summary = dict(type='summary', result='failed' if reason else 'shadow_completed', reason=reason,
                   samples=count, elapsed_s=time.monotonic()-started, max_inference_s=max_inference,
                   max_source_age_s=max_age, max_tilt_rad=max_tilt, max_target_delta_rad=max_target_delta,
                   motion_commands_sent=0, standing_validated=False,
                   received=mailbox.received, duplicate_ticks=mailbox.duplicates)
    emit(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface', required=True)
    parser.add_argument('--seconds', type=float, default=20.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 60:
        parser.error('seconds must be in (0, 60]')
    policy = LowerBodyPolicy()
    policy.warmup()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    with args.output.open('x') as output:
        def emit(row):
            output.write(json.dumps(row, allow_nan=False)+'\n')
            output.flush()
        emit(dict(type='metadata', control_mode='lower_body_shadow', hardware_motion_enabled=False,
                  interface=args.interface, duration_s=args.seconds, velocity_command=[0, 0, 0],
                  policy_commit=policy.lock['commit'], policy_sha256=policy.lock['sha256']['deploy/pre_train/g1/motion.pt'],
                  note='Raw policy outputs only; no position hold, command encoding, publisher or actuator control.'))
        ChannelFactoryInitialize(0, args.interface)
        mailbox = FeedbackMailbox()
        subscriber = ChannelSubscriber('rt/lowstate', LowState_)
        try:
            subscriber.Init(mailbox.receive, 10)
            time.sleep(.1)
            summary = run(policy, mailbox, args.seconds, emit)
        finally:
            subscriber.Close()
    print(json.dumps(summary, indent=2))
    return 0 if summary['result'] == 'shadow_completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
