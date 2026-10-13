"""Headless candidate screening. Never imports SDK or opens robot transport."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from lower_body_policy import BODY_NAMES, ROOT, LowerBodyPolicy, gravity, load_candidate
from g1_sim_model import build_model
from standing_reference import StandingReference, quaternion_yaw
from stance_odometry import StanceOdometry

# Isolated optional simulation packages; the teleoperation environment is unchanged.
sys.path.insert(0, str(ROOT/'.local/unitree_g1/sim_deps'))
import mujoco


def run(model_name, seconds, push=0., arm_motion=False, delay_steps=0, simulation_dt=.001,
        servo='implicit', position_hold=False, dex1=False, extra_mass=0., payload_mass=0.,
        gripper_cycle=False, elbow=0., pose_source='simulation_truth', unified=False,
        upper_stop='operator_release', load_compensation=False):
    if (not math.isfinite(seconds) or not 1 <= seconds <= 180
            or simulation_dt not in (.002, .001, .0005)
            or type(delay_steps) is not int or not 0 <= delay_steps <= 10
            or not math.isfinite(push) or abs(push) > 100
            or pose_source not in ('simulation_truth', 'kinematic')
            or elbow not in (0., .5, 1.)
            or upper_stop not in ('operator_release', 'camera_failure', 'input_timeout')):
        raise ValueError('Invalid simulation scenario')
    root, c, lock = load_candidate()
    m, c, model_metadata = build_model(model_name, simulation_dt, servo, dex1, extra_mass, payload_mass)
    decimation = round(.02/simulation_dt)
    d = mujoco.MjData(m)
    kinematic_data = mujoco.MjData(m)
    odometry = StanceOdometry()
    body_count = 12 if model_name == '12dof' else 29
    names = BODY_NAMES[:body_count]
    if m.nu != body_count+(4 if dex1 else 0):
        raise ValueError('Unexpected actuator count')
    joints = [m.joint(name).id for name in names]
    qa, va = m.jnt_qposadr[joints], m.jnt_dofadr[joints]
    actuators = [m.actuator(name).id for name in names]
    if any(m.actuator_trnid[a, 0] != j for a, j in zip(actuators, joints)):
        raise ValueError('Actuator mapping mismatch')
    policy = LowerBodyPolicy()
    policy.warmup()
    target = np.array(c['default_angles'] + (c['arm_waist_target'] if body_count == 29 else []))
    if body_count == 29:
        target[[18, 25]] = elbow
    kp = np.array(c['kps'] + (c['arm_waist_kps'] if body_count == 29 else []))
    kd = np.array(c['kds'] + (c['arm_waist_kds'] if body_count == 29 else []))
    # Explicit ideal start at default joint posture, with policy active immediately.
    # This is NOT a model of real robot startup/authority transfer.
    d.qpos[qa] = target
    mujoco.mj_forward(m, d)
    standing = StandingReference(d.qpos[:2], quaternion_yaw(d.qpos[3:7]), -.02)
    velocity_command = np.zeros(3)
    compensator = None
    if load_compensation:
        if not unified or not dex1:
            raise ValueError('Load compensation requires unified Dex1 simulation')
        from upper_body_balance import UpperBodyBalance
        compensator = UpperBodyBalance(m, c, elbow)
    controller = None
    final_frame = None
    upper_started = False
    if unified:
        if not dex1 or model_name != '29dof' or servo != 'implicit':
            raise ValueError('Unified simulation requires 29dof + Dex1 + implicit servo')
        from whole_body_control import WholeBodyControl, BodyState, PolicyResult, BodyFault
        from quest_bridge import load_quest_profile
        from quest_grippers import gripper_settings
        _, arm_config = load_quest_profile()
        controller = WholeBodyControl(arm_config, c, (m.jnt_range[joints, 0], m.jnt_range[joints, 1]), gripper_settings())
        def body_state(stamp, tick):
            q, dq = np.zeros(35), np.zeros(35)
            q[:29], dq[:29] = d.qpos[qa], d.qvel[va]
            for slot, side in ((31, 'left'), (33, 'right')):
                j = m.joint(f'{side}_dex1_finger_joint_1').id
                # Uncalibrated affine simulation surrogate, not a motor model.
                q[slot] = -5.4*(d.qpos[m.jnt_qposadr[j]]+.02)/.0445
                dq[slot] = -5.4*d.qvel[m.jnt_dofadr[j]]/.0445
            return BodyState(stamp, tick, 5, tuple(q), tuple(dq), tuple(d.qpos[3:7]), tuple(d.qvel[3:6]))
        controller.initialize(body_state(-simulation_dt, 0), -simulation_dt)
        controller.preopen(0.)
    pelvis = m.body('pelvis').id
    feet = [m.body(f'{side}_ankle_roll_link').id for side in ('left', 'right')]
    sole_geoms = [i for i in range(m.ngeom) if m.geom_bodyid[i] in feet and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE]
    if len(sole_geoms) != 8:
        raise ValueError('Expected eight sole contact spheres')
    start_xy = None
    initial_feet = None
    metrics = dict(max_tilt_rad=0., max_drift_m=0., max_foot_lift_m=0.,
                   max_joint_limit_excess_rad=0., max_abs_requested_torque_nm=0.,
                   max_abs_applied_torque_nm=0., min_pelvis_height_m=10., max_odometry_error_m=0.)
    latencies, trace, pending = [], [], []
    phase_history = []
    clipped_frames = 0
    completed_steps = 0
    failure = None
    for step in range(round(seconds/m.opt.timestep)):
        t = step*m.opt.timestep
        if t >= 0 and start_xy is None:
            start_xy = d.qpos[:2].copy()
            initial_feet = d.xpos[feet, 2].copy()
        if t >= 0 and step % decimation == 0:
            if position_hold:
                if pose_source == 'kinematic':
                    kinematic_data.qpos[:] = d.qpos
                    kinematic_data.qpos[:3] = 0.  # no world position/velocity supplied to estimator
                    mujoco.mj_kinematics(m, kinematic_data)
                    position, velocity = odometry.update(kinematic_data.geom_xpos[sole_geoms], t)
                    metrics['max_odometry_error_m'] = max(metrics['max_odometry_error_m'], float(np.linalg.norm(position-(d.qpos[:2]-start_xy))))
                else:
                    position, velocity = d.qpos[:2], d.qvel[:2]
                velocity_command = standing.update(position, velocity, quaternion_yaw(d.qpos[3:7]), t, t)
            before = time.perf_counter()
            waist_target = compensator.target(d.qpos[qa], d.qpos[3:7]) if compensator is not None else None
            leg = policy.step(d.qpos[qa[:12]], d.qvel[va[:12]], d.qpos[3:7], d.qvel[3:6], velocity_command)
            latencies.append(time.perf_counter()-before)
            # Inject latency after 5 s, preserving the original observation age.
            delay = delay_steps if t >= 5 else 0
            pending.append((step + delay*decimation, leg, t, policy.counter,
                            tuple(policy.last_observation), tuple(policy.previous), latencies[-1], waist_target))
            while pending and pending[0][0] <= step:
                result = pending.pop(0)
                target[:12] = result[1]
                if controller is not None:
                    try:
                        controller.update_policy(PolicyResult(t, result[2], result[3], tuple(result[1]),
                            result[4], result[5], result[6], result[7]))
                    except BodyFault as exc:
                        failure = str(exc)
                        break
            if failure is not None:
                break
        if arm_motion and controller is None and body_count == 29 and 2 <= t <= 10:
            # Reproducible asymmetric smooth left-arm surrogate, not Quest acceptance.
            target[15] = -.6*(1-math.cos(2*math.pi*(t-2)/8))/2
        if controller is not None:
            if controller.phase == 'READY' and not upper_started:
                controller.begin_task()
                upper_started = True
            if controller.phase == 'ACTIVE':
                if t >= 10 and upper_stop != 'input_timeout':
                    controller.stop_upper(upper_stop)
                elif t < 10:
                    arm_target = np.array(c['arm_waist_target'][3:])
                    arm_target[[3, 10]] = elbow
                    if arm_motion and 4 <= t <= 10:
                        arm_target[0] = -.6*(1-math.cos(2*math.pi*(t-4)/6))/2
                    triggers = [.5*(1-math.cos(2*math.pi*(t-4)/6))]*2 if gripper_cycle and 4 <= t <= 10 else [0., 0.]
                    controller.upper_target(arm_target, triggers, t, t)
            try:
                final_frame = controller.step(body_state(t, step+1), t)
            except BodyFault as exc:
                failure = str(exc)
                break
            if not phase_history or phase_history[-1]['phase'] != final_frame.upper_phase:
                phase_history.append(dict(time=t, phase=final_frame.upper_phase, reason=final_frame.upper_reason))
            clipped_frames += bool(final_frame.clipped_leg_indices)
            target[:] = final_frame.q[:29]
            kp[:], kd[:] = final_frame.kp[:29], final_frame.kd[:29]
            m.actuator_gainprm[actuators, 0] = kp
            m.actuator_biasprm[actuators, 1] = -kp
            m.actuator_biasprm[actuators, 2] = -kd
        d.xfrc_applied[:] = 0
        if 5 <= t < 5.2:
            d.xfrc_applied[pelvis, 1] = push
        torque = kp*(target-d.qpos[qa])-kd*d.qvel[va]
        d.ctrl[actuators] = target if servo == 'implicit' else torque
        if dex1:
            # Metres of finger travel, deliberately separate from uncalibrated
            # internal Dex1 motor units. Smooth open-close mass-motion screening.
            travel = -.02 + .0445*(1-math.cos(2*math.pi*min(t, 8)/8))/2 if gripper_cycle else 0.
            for side in ('left', 'right'):
                if final_frame is not None:
                    travel = -.02-.0445*final_frame.q[31 if side == 'left' else 33]/5.4
                for finger in (1, 2):
                    d.ctrl[m.actuator(f'{side}_dex1_finger_joint_{finger}').id] = travel
        before_state = dict(q=d.qpos[qa].tolist(), dq=d.qvel[va].tolist(),
                            quaternion_wxyz=d.qpos[3:7].tolist(), gyro_pelvis=d.qvel[3:6].tolist(),
                            time=float(d.time)) if step % decimation == 0 else None
        if controller is not None and before_state is not None:
            before_state['body_state'] = asdict(body_state(t, step+1))
        mujoco.mj_step(m, d)
        completed_steps += 1
        applied_torque = d.qfrc_actuator[va].copy()
        mujoco.mj_forward(m, d)  # derived poses/contact data must match post-step qpos
        if any(w.number for w in d.warning) or abs(d.time-completed_steps*simulation_dt) > 1e-7:
            failure = 'simulator_warning_or_reset'
            break
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            failure = 'nonfinite_state'
            break
        if t < 0:
            continue
        g = gravity(d.qpos[3:7])
        tilt = math.acos(float(np.clip(-g[2], -1, 1)))
        drift = float(np.linalg.norm(d.qpos[:2]-start_xy))
        excess = float(np.max(np.maximum(m.jnt_range[joints, 0]-d.qpos[qa], d.qpos[qa]-m.jnt_range[joints, 1])))
        metrics['max_tilt_rad'] = max(metrics['max_tilt_rad'], tilt)
        metrics['max_drift_m'] = max(metrics['max_drift_m'], drift)
        metrics['max_foot_lift_m'] = max(metrics['max_foot_lift_m'], float(np.max(d.xpos[feet, 2]-initial_feet)))
        metrics['max_joint_limit_excess_rad'] = max(metrics['max_joint_limit_excess_rad'], excess)
        metrics['max_abs_requested_torque_nm'] = max(metrics['max_abs_requested_torque_nm'], float(np.max(np.abs(torque))))
        metrics['max_abs_applied_torque_nm'] = max(metrics['max_abs_applied_torque_nm'], float(np.max(np.abs(applied_torque))))
        metrics['min_pelvis_height_m'] = min(metrics['min_pelvis_height_m'], float(d.qpos[2]))
        if step % decimation == 0:
            trace.append(dict(t=round(t, 3), before_state=before_state, q=d.qpos[qa].tolist(), dq=d.qvel[va].tolist(),
                              quaternion_wxyz=d.qpos[3:7].tolist(), observation=policy.last_observation.tolist(),
                              velocity_command=velocity_command.tolist(),
                              upper_phase=final_frame.upper_phase if final_frame else None,
                              whole_body_frame=asdict(final_frame) if final_frame else None,
                              policy_result=asdict(controller.policy) if controller else None,
                              action16=list(final_frame.action16()) if final_frame else None,
                              action=policy.previous.tolist(), target=target.tolist(), tilt=tilt, drift=drift))
        if tilt > .6 or d.qpos[2] < .45:
            failure = 'fall_screen'
            break
    # Screening thresholds selected BEFORE trials, not hardware acceptance thresholds.
    limits = {'max_tilt_rad': .2, 'max_drift_m': .15, 'max_foot_lift_m': .03,
              'max_joint_limit_excess_rad': .01}
    violations = [key for key, bound in limits.items() if metrics[key] > bound]
    return dict(model=model_name, model_metadata=model_metadata, position_hold=position_hold,
                unified=unified, load_compensation=load_compensation,
                upper_phase_history=phase_history, clipped_leg_frames=clipped_frames,
                final_body_gains_positive=all(x > 0 for x in final_frame.kp[:29]) if final_frame else None,
                pose_source=pose_source if position_hold else None, gripper_cycle=gripper_cycle, elbow_target=elbow,
                source_commit=lock['commit'], simulation_dt=simulation_dt, seconds_requested=seconds,
                seconds_completed=completed_steps*simulation_dt, push_newtons=push,
                arm_motion=arm_motion, inference_delay_steps=delay_steps, metrics=metrics,
                inference_ms_p50=float(np.median(latencies)*1000) if latencies else None,
                inference_ms_max=float(max(latencies)*1000) if latencies else None,
                screening_limits=limits, violations=violations, failure=failure,
                passed=failure is None and not violations, hardware_approved=False,
                limitations=['ideal initialized posture', 'no measured Dex1/mount calibration',
                             'world pose source: '+pose_source if position_hold else 'no world pose feedback',
                             'Dex1 motor-to-slider mapping is uncalibrated affine surrogate in unified simulation',
                             'no camera/Quest concurrency', 'no hardware authority transition'], trace=trace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['12dof', '29dof'], default='29dof')
    parser.add_argument('--seconds', type=float, default=20.)
    parser.add_argument('--push', type=float, default=0.)
    parser.add_argument('--arm-motion', action='store_true')
    parser.add_argument('--delay-steps', type=int, default=0)
    parser.add_argument('--simulation-dt', type=float, choices=[.002, .001, .0005], default=.001)
    parser.add_argument('--servo', choices=['implicit', 'explicit'], default='implicit')
    parser.add_argument('--position-hold', action='store_true')
    parser.add_argument('--dex1', action='store_true')
    parser.add_argument('--extra-mass', type=float, default=0.)
    parser.add_argument('--payload-mass', type=float, default=0.)
    parser.add_argument('--gripper-cycle', action='store_true')
    parser.add_argument('--elbow', type=float, choices=[0., .5, 1.], default=0.)
    parser.add_argument('--pose-source', choices=['simulation_truth', 'kinematic'], default='simulation_truth')
    parser.add_argument('--unified', action='store_true')
    parser.add_argument('--load-compensation', action='store_true')
    parser.add_argument('--upper-stop', choices=['operator_release', 'camera_failure', 'input_timeout'], default='operator_release')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 180:
        parser.error('seconds must be 1..180')
    if not math.isfinite(args.push) or abs(args.push) > 100 or not 0 <= args.delay_steps <= 10:
        parser.error('invalid disturbance or delay')
    if args.arm_motion and args.model != '29dof':
        parser.error('arm motion requires 29dof')
    if args.gripper_cycle and not args.dex1:
        parser.error('gripper cycle requires Dex1')
    if args.output.exists():
        parser.error('output already exists')
    result = run(args.model, args.seconds, args.push, args.arm_motion, args.delay_steps, args.simulation_dt,
                 args.servo, args.position_hold, args.dex1, args.extra_mass, args.payload_mass, args.gripper_cycle,
                 args.elbow, args.pose_source, args.unified, args.upper_stop, args.load_compensation)
    trace = result.pop('trace')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace_path = args.output.with_suffix('.trace.jsonl')
    with trace_path.open('x') as stream:
        for row in trace:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    result['trace_path'] = str(trace_path)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
