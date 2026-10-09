#!/usr/bin/env python3
"""Verify the pinned XR software offline; never instantiate a robot controller."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / 'artifacts/unitree_g1/baseline.lock.json'


def run(args, **kwargs):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, **kwargs).strip()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ik', action='store_true', help='Also build the upstream model and solve one offline IK target')
    parser.add_argument('--output', type=Path, help='Write the verification report')
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text())
    require(not os.environ.get('PYTHONPATH'), 'Unset PYTHONPATH to exclude ROS/system packages.')
    require(f'{sys.version_info.major}.{sys.version_info.minor}' == lock['python_minor'], 'Use the locked Python 3.10 environment.')
    report = {'status': 'passed', 'scope': 'offline software only', 'hardware_motion_tested': False, 'checks': [], 'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'baseline_sha256': hashlib.sha256(LOCK.read_bytes()).hexdigest()}
    for key in ['lock_sha256', 'g1_asset_sha256']:
        for path, digest in lock[key].items():
            require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, f'Hash mismatch: {path}')
    report['checks'].append(f"lock files and {len(lock['g1_asset_sha256'])} G1 asset hashes")
    for repo in lock['repositories']:
        path = ROOT / repo['path']
        require(run(['git', 'rev-parse', 'HEAD'], cwd=path) == repo['commit'], f"Commit mismatch: {repo['name']}")
        expected = (ROOT / repo['patch']).read_text().strip() if 'patch' in repo else ''
        require(run(['git', 'diff', 'HEAD', '--ignore-submodules=all'], cwd=path) == expected, f"Unexpected tracked changes: {repo['name']}")
        require(not run(['git', 'ls-files', '--others', '--exclude-standard', '--exclude=build/', '--exclude=*.egg-info/', '--exclude=__pycache__/'], cwd=path), f"Untracked source files: {repo['name']}")
        report['checks'].append(f"source {repo['name']} {repo['commit'][:12]}")
    for name, version in lock['python_packages'].items():
        require(importlib.metadata.version(name) == version, f'Package version mismatch: {name}')
    for item in lock['conda_packages']:
        path = Path(sys.prefix) / 'conda-meta' / f"{item['name']}-{item['version']}-{item['build_string']}.json"
        require(path.exists(), f"Conda build missing: {path.name}")
    report['checks'].append(run([sys.executable, '-m', 'pip', '--no-cache-dir', 'check']))
    sdk_dir = Path(importlib.util.find_spec('unitree_sdk2py').origin).parent
    crc_relative = Path('utils/lib/crc_amd64.so')
    crc_source = ROOT / '.local/unitree_g1/unitree_sdk2_python/unitree_sdk2py' / crc_relative
    require((sdk_dir / crc_relative).is_file(), 'SDK CRC library missing; see g1_arm_execution.md')
    require(hashlib.sha256((sdk_dir / crc_relative).read_bytes()).digest() ==
            hashlib.sha256(crc_source.read_bytes()).digest(), 'SDK CRC library differs from pinned source')
    report['checks'].append('SDK native CRC library matches pinned source')
    xr = ROOT / '.local/unitree_g1/xr_teleoperate'
    # Import each component in a fresh process: upstream entrypoints configure
    # logging at import time. Reject socket bind/connect even if upstream changes.
    guard = '''
import sys

def block_network(event, args):
    if event in ('socket.bind', 'socket.connect', 'socket.getaddrinfo'):
        raise RuntimeError('Network access forbidden in offline baseline verification')
sys.addaudithook(block_network)
'''
    with tempfile.TemporaryDirectory(prefix='g1-xr-verify-') as tmp:
        env = dict(os.environ, MPLCONFIGDIR=tmp, PYTHONNOUSERSITE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        for name in ['televuer', 'teleimager.image_client', 'teleimager.image_server', 'dex_retargeting',
                     'teleop.robot_control.robot_arm', 'teleop.robot_control.robot_hand_unitree',
                     'teleop.robot_control.robot_arm_ik', 'teleop.utils.episode_writer', 'teleop.teleop_hand_and_arm']:
            code = guard + f'\nsys.path.insert(0, {str(xr)!r})\nimport {name}\n'
            run([sys.executable, '-c', code], env=env, timeout=45)
            report['checks'].append(f'import {name}')
        if args.ik:
            tmp = Path(tmp)
            (tmp / 'assets').symlink_to(xr / 'assets', target_is_directory=True)
            (tmp / 'teleop').mkdir()
            code = guard + f'\nsys.path.insert(0, {str(xr)!r})\n' + '''
import json
import numpy as np
import pinocchio as pin
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
ik = G1_29_ArmIK(Visualization=False)
model = ik.reduced_robot.model
# Upstream adds end-effector frames after creating reduced_robot.data.
# Make fresh Data for this FK check to match the final frame count.
data = model.createData()
q = np.zeros(model.nq)
pin.framesForwardKinematics(model, data, q)
left = data.oMf[ik.L_hand_id].homogeneous.copy()
right = data.oMf[ik.R_hand_id].homogeneous.copy()
result, torque = ik.solve_ik(left, right, q, np.zeros(model.nv))
if not ik.opti.stats()['success']:
    raise RuntimeError('IK solver failed (fallback results are not acceptance)')
if result.shape != (14,) or not np.isfinite(result).all() or not np.isfinite(torque).all():
    raise RuntimeError('Invalid IK output')
pin.framesForwardKinematics(model, data, result)
error = max(np.linalg.norm(data.oMf[ik.L_hand_id].translation - left[:3, 3]),
            np.linalg.norm(data.oMf[ik.R_hand_id].translation - right[:3, 3]))
if error >= 0.001:
    raise RuntimeError(f'FK/IK position error: {error}')
print(json.dumps({'model_nq': model.nq, 'solver_success': True, 'position_error_m': float(error)}))
'''
            output = run([sys.executable, '-c', code], cwd=tmp / 'teleop', env=env, timeout=90)
            report['ik'] = json.loads(output.splitlines()[-1])
    payload = json.dumps(report, indent=2, ensure_ascii=False) + '\n'
    if args.output:
        args.output.write_text(payload)
    print(payload, end='')


if __name__ == '__main__':
    main()
