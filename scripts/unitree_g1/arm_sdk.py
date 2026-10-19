"""Candidate upper-body transport while the official G1 controller owns the legs.

No service activation, standing transition or walking RPC is issued here.
The operator first establishes Regular mode with the official controls.
Internal Dex1 is deliberately unsupported. The CLI is a read-only preflight.
"""
import argparse
import json
import time

from arm_execution import ARM_INDICES, vector
from hold_arms import DDSHoldIO, OwnershipLost


# 500 is the pinned LocoClient.Start FSM. 501 is the operator-confirmed mode
# on this robot and is used by the Holosoma arm_sdk+loco split-body backend.
# This remains a candidate, not hardware acceptance. No arbitrary FSM override;
# lock the exact mode during takeover. Never infer FSM from mode_machine.
STANDING = dict(service='ai', fsm_id=500, allowed_fsm_ids=(500, 501), preflight_tilt_rad=.30,
                preflight_gyro_rad_s=.4, preflight_leg_speed_rad_s=.30,
                active_tilt_rad=.30, active_gyro_rad_s=.5,
                waist_tracking_error_rad=.50, waist_kp=60., waist_kd=1.5)
# Waist gains follow the official arm7 example, not the debug profile.


def sdk_weight(command, config):
    """Reuse the executor's common ramp as authority weight, not gain scaling."""
    ratios = [a/b for values, bases in ((command.kp, config.kp), (command.kd, config.kd))
              for a, b in zip(vector(values), bases)]
    weight = ratios[0]
    if (not 0 <= weight <= 1 or any(abs(x-weight) > 1e-8 for x in ratios)):
        raise ValueError('arm_sdk requires one finite common authority ramp')
    return weight


def encode_arm_sdk_command(command, config, waist_q):
    """Populate arms + captured waist pose + slot 29 authority, never legs/Dex1."""
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    weight = sdk_weight(command, config)
    q, waist_q = vector(command.q), vector(waist_q, 3)
    msg = unitree_hg_msg_dds__LowCmd_()
    msg.mode_pr, msg.mode_machine = 0, command.mode_machine
    for j, i in enumerate(ARM_INDICES):
        motor = msg.motor_cmd[i]
        motor.mode, motor.q = 1, q[j]
        motor.kp, motor.kd = config.kp[j], config.kd[j]
    for j, i in enumerate(range(12, 15)):
        motor = msg.motor_cmd[i]
        motor.mode, motor.q = 1, waist_q[j]
        motor.kp, motor.kd = STANDING['waist_kp'], STANDING['waist_kd']
    msg.motor_cmd[29].q = weight
    msg.crc = CRC().Crc(msg)
    return msg


def check_body(telemetry, now, timeout, *, preflight=False, waist_q=None):
    if telemetry is None or not 0 <= now-telemetry['received'] <= timeout:
        raise ValueError('Stale body feedback')
    rpy, gyro = vector(telemetry['imu_rpy'], 3), vector(telemetry['imu_gyroscope'], 3)
    prefix = 'preflight' if preflight else 'active'
    if max(abs(v) for v in rpy[:2]) > STANDING[prefix+'_tilt_rad']:
        raise ValueError('Body tilt exceeds standing gate')
    if max(abs(v) for v in gyro) > STANDING[prefix+'_gyro_rad_s']:
        raise ValueError('Body angular speed exceeds standing gate')
    motors = {m['index']: m for m in telemetry['motors']}
    for i in range(15):
        if motors[i]['motorstate'] != 0:
            raise ValueError('Leg/waist motor fault')
        vector((motors[i]['q'], motors[i]['dq']), 2)
    if preflight and max(abs(motors[i]['dq']) for i in range(15)) > STANDING['preflight_leg_speed_rad_s']:
        raise ValueError('Legs/waist not stationary')
    if waist_q is not None and any(abs(motors[i]['q']-q) > STANDING['waist_tracking_error_rad']
                                  for i, q in zip(range(12, 15), waist_q)):
        raise ValueError('Waist tracking error')


class ArmSDKIO(DDSHoldIO):
    command_topic_name = 'rt/arm_sdk'
    # rt/lowcmd is EXPECTED from official locomotion. It cannot be treated as
    # a competitor. This does not authenticate the publisher of that topic.
    monitored_topics = ('rt/arm_sdk', 'rt/user_lowcmd')

    def __init__(self, interface, live, profile, config):
        self.config, self.waist_q = config, None
        self.last_sdk = None
        self.mode_evidence = None
        super().__init__(interface, live, profile)

    def _init_mode_clients(self, client_type):
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        super()._init_mode_clients(client_type)
        self.client.SetTimeout(.4)
        self.loco = LocoClient()
        self.loco.SetTimeout(.4)
        self.loco.Init()

    def _poll_mode(self):
        while not self.done.is_set():
            started = time.monotonic()
            evidence = {}
            try:
                code, value = self.client.CheckMode()
                fsm_code, fsm = self.loco.GetFsmId()
                evidence = dict(service_code=code, service=value, fsm_code=fsm_code, fsm_id=fsm)
                ok = (code == 0 and isinstance(value, dict) and value.get('name') == STANDING['service']
                      and fsm_code == 0 and type(fsm) is int and fsm in STANDING['allowed_fsm_ids'])
            except Exception as exc:
                ok = False
                evidence['error'] = str(exc)
            with self.lock:
                self.mode_ok, self.mode_time = ok, started
                self.mode_evidence = evidence
            self.done.wait(.2)

    def _check_mode(self):
        with self.lock:
            if not self.mode_ok or not 0 <= time.monotonic()-self.mode_time <= 1.6:
                raise OwnershipLost(f'Official ai/FSM={STANDING["allowed_fsm_ids"]} not confirmed or query stale: {self.mode_evidence}')
            if (getattr(self, 'engaged_fsm', None) is not None
                    and self.mode_evidence.get('fsm_id') != self.engaged_fsm):
                raise OwnershipLost('Official FSM changed during upper-body ownership')

    def check_readiness(self):
        self.observation()  # Surface rejected feedback before a generic freshness error.
        check_body(self.telemetry(), time.monotonic(), self.config.state_timeout, preflight=True)

    def check_active_body(self):
        self.observation()
        check_body(self.telemetry(), time.monotonic(), self.config.state_timeout, waist_q=self.waist_q)

    def open_writer(self):
        self.check()
        with self.lock:
            self.engaged_fsm = self.mode_evidence['fsm_id']
        self.check_readiness()
        motors = {m['index']: m for m in self.telemetry()['motors']}
        self.waist_q = tuple(motors[i]['q'] for i in range(12, 15))
        if self.writer is not None:
            # Preopening may have released authority without tearing down DDS.
            # All ownership/readiness checks above still apply at takeover.
            if self.matched == 0:
                raise RuntimeError('Arm SDK command subscriber lost before reuse')
            return
        super().open_writer()

    def write(self, command):
        self.check()
        msg = encode_arm_sdk_command(command, self.config, self.waist_q)
        if self.live:
            if self.writer is None or self.matched == 0:
                return False
            self.writer.write(msg)
        self.last_sdk = dict(topic=self.command_topic_name, weight=float(msg.motor_cmd[29].q),
                             joint_indices=list(range(12, 29)),
                             q=[float(msg.motor_cmd[i].q) for i in range(12, 29)],
                             kp=[float(msg.motor_cmd[i].kp) for i in range(12, 29)],
                             kd=[float(msg.motor_cmd[i].kd) for i in range(12, 29)])
        return True

    def command_metadata(self):
        return dict(arm_sdk_request=self.last_sdk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface', required=True)
    args = parser.parse_args()
    from quest_bridge import load_quest_profile
    profile, config = load_quest_profile()
    io = None
    result = dict(read_only=True, command_frames=0, preflight_passed=False,
                  standing_gate=STANDING, physical_balance_verified=False)
    try:
        io = ArmSDKIO(args.interface, False, profile, config)
        io.preflight()
        result['preflight_passed'] = True
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if io is not None:
            result['mode'] = io.mode_evidence
            io.close()
    print(json.dumps(result, indent=2))
    return 0 if result['preflight_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
