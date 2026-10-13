"""Injectable offline runtime: inference mailbox, bounded recorder, one servo sink.

There is no DDS factory here. Hardware startup/authority transfer is intentionally
unavailable until the pose source and physical fault procedure are validated.
"""
from dataclasses import asdict, replace
import queue
import threading
import time

from arm_execution import vector
from whole_body_control import PolicyResult, BodyFault


class PolicyWorker:
    """Inference never runs on the servo thread. Mailbox holds latest input only."""
    def __init__(self, policy, controller, clock=time.monotonic):
        self.policy, self.controller, self.clock = policy, controller, clock
        self.inputs = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, name='g1-policy-offline', daemon=True)

    def start(self):
        self.thread.start()

    def submit(self, state, velocity_command=(0., 0., 0.), waist_target=None):
        if self.stop_event.is_set() or self.error:
            raise BodyFault(self.error or 'inference worker stopped')
        state = replace(state, q=vector(state.q, 35), dq=vector(state.dq, 35),
                        quaternion=vector(state.quaternion, 4), gyro=vector(state.gyro, 3))
        item = (state, vector(velocity_command, 3), None if waist_target is None else vector(waist_target, 3))
        try:
            self.inputs.put_nowait(item)
        except queue.Full:
            try:
                self.inputs.get_nowait()
            except queue.Empty:
                pass
            self.inputs.put_nowait(item)

    def _run(self):
        sequence = 0
        previous_stamp = None
        try:
            # Startup warmup is outside any motion loop; controller has no result
            # until warmup and the first fresh inference have actually completed.
            self.policy.warmup()
            while not self.stop_event.is_set():
                try:
                    state, command, waist = self.inputs.get(timeout=.02)
                except queue.Empty:
                    continue
                now = self.clock()
                if not 0 <= now-state.stamp <= .04:
                    raise BodyFault('inference_input_deadline')
                if previous_stamp is not None and not .019 <= state.stamp-previous_stamp <= .04:
                    raise BodyFault('inference_cadence')
                before = self.clock()
                target = self.policy.step(state.q[:12], state.dq[:12], state.quaternion, state.gyro, command)
                finished = self.clock()
                sequence += 1
                self.controller.update_policy(PolicyResult(finished, state.stamp, sequence, tuple(target),
                    tuple(self.policy.last_observation), tuple(self.policy.previous), finished-before, waist))
                previous_stamp = state.stamp
        except Exception as exc:
            self.error = 'inference_worker: '+str(exc)
            if self.controller.body_fault is None:
                self.controller.body_fault = self.error

    def close(self):
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=.5)
        return not self.thread.is_alive()


class RecordingQueue:
    """Servo-side enqueue only; consume/write in a separate recorder thread."""
    def __init__(self, capacity=1000):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError('Recording capacity must be positive')
        self.queue = queue.Queue(maxsize=capacity)
        self.failure = None

    def enqueue(self, row):
        if self.failure is not None:
            return False
        try:
            self.queue.put_nowait(row)
            return True
        except queue.Full:
            self.failure = 'recording_queue_full'
            return False

    def write_one(self, writer):
        """Called by an independent consumer, never by servo tick."""
        if self.failure:
            return False
        try:
            row = self.queue.get_nowait()
        except queue.Empty:
            return False
        try:
            writer(row)
        except Exception as exc:
            self.failure = 'recording_write_failed: '+str(exc)
            return False
        return True


class OfflineServoSession:
    """Exactly one synchronous simulation/shadow sink and a nonblocking recorder."""
    def __init__(self, controller, emit, recorder=None, clock=time.monotonic):
        self.controller, self.emit, self.recorder, self.clock = controller, emit, recorder, clock
        self.frames_submitted = 0

    def tick(self, state, now):
        c = self.controller
        if self.recorder is not None and self.recorder.failure:
            c.stop_upper(self.recorder.failure)
        frame = c.step(state, now)
        before = asdict(c.state)
        inference = asdict(c.last_policy_used)
        started = self.clock()
        try:
            if self.emit(frame) is not True:
                raise OSError('simulation sink rejected frame')
        except Exception as exc:
            c._fault('submission_failed: '+str(exc))
        finished = self.clock()
        self.frames_submitted += 1
        if self.recorder is not None:
            row = dict(type='whole_body_simulation_command', control_mode='whole_body_simulation',
                       observation_before_command=before, policy_result=inference,
                       command=asdict(frame), action16=frame.action16(),
                       command_send_started=started, command_send_finished=finished,
                       hardware_motion_enabled=False, simulator_submission=True)
            if not self.recorder.enqueue(row):
                c.stop_upper(self.recorder.failure)
        return frame
