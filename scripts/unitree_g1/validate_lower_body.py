"""Reproducible headless development matrix. A failing case is retained, not retried."""
import argparse
import hashlib
import json
from pathlib import Path

from sim_lower_body import run
from lower_body_policy import ROOT


def cases():
    yield 'explicit_29dof', dict(servo='explicit', simulation_dt=.002)
    for dt in (.002, .001, .0005):
        yield f'implicit_zero_{dt}', dict(simulation_dt=dt)
    for dt in (.002, .001, .0005):
        yield f'dex1_hold_{dt}', dict(dex1=True, elbow=.5, position_hold=True, simulation_dt=dt)
    for source in ('simulation_truth', 'kinematic'):
        yield f'unified_180s_{source}', dict(dex1=True, elbow=.5, position_hold=True, pose_source=source,
            unified=True, arm_motion=True, gripper_cycle=True, seconds=180.)
    for name, changes in [
        ('mount_200g', dict(extra_mass=.2)),
        ('left_payload_500g', dict(payload_mass=.5)),
        ('mount_and_payload', dict(extra_mass=.2, payload_mass=.5)),
        ('push_10N', dict(push=10.)),
        ('push_20N', dict(push=20.)),
        ('camera_failure', dict(upper_stop='camera_failure')),
        ('input_timeout', dict(upper_stop='input_timeout')),
        ('inference_delay_20ms', dict(delay_steps=1)),
        ('inference_delay_60ms', dict(delay_steps=3))]:
        yield name, dict(dex1=True, elbow=.5, position_hold=True, unified=True,
                         arm_motion=True, gripper_cycle=True, **changes)
    for dt in (.001, .0005):
        yield f'compensated_load_{dt}', dict(dex1=True, elbow=.5, position_hold=True, unified=True,
            arm_motion=True, gripper_cycle=True, extra_mass=.2, payload_mass=.5,
            load_compensation=True, seconds=60., simulation_dt=dt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--case', action='append', help='Run only named cases (repeatable)')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = {}
    if args.case and set(args.case)-{name for name, _ in cases()}:
        p.error('Unknown case name')
    for name, kwargs in cases():
        if args.case and name not in args.case:
            continue
        options = dict(model_name='29dof', seconds=20.)
        options.update(kwargs)
        result = run(**options)
        trace = result.pop('trace')
        path = args.output_dir/(name+'.trace.jsonl')
        with path.open('x') as stream:
            for row in trace:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
        result['trace_path'] = str(path)
        result['trace_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        (args.output_dir/(name+'.json')).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        results[name] = result
        print(name, 'PASS' if result['passed'] else 'FAIL', result['failure'], result['violations'], flush=True)
    sources = ['sim_lower_body.py', 'g1_sim_model.py', 'lower_body_policy.py', 'standing_reference.py',
               'stance_odometry.py', 'whole_body_control.py', 'upper_body_balance.py', 'validate_lower_body.py']
    report = dict(hardware_approved=False, hardware_commands_sent=0,
                  purpose='Offline development matrix, including negative and stress cases',
                  code_sha256={name: hashlib.sha256((ROOT/'scripts/unitree_g1'/name).read_bytes()).hexdigest() for name in sources},
                  cases=results)
    (args.output_dir/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
