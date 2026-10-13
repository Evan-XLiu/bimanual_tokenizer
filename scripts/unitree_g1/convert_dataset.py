#!/usr/bin/env python3
"""Export recorded Quest sessions to LeRobot v3 (0.4.4), without any robot imports."""
import argparse
from bisect import bisect_left, bisect_right
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import uuid

import numpy as np
from PIL import Image

from export_kinematics import MODEL_PATH, conventions, eef_features

# Raw recording roles stay stable so historical sessions remain readable.
ROLES = ('head', 'left_wrist', 'right_wrist')
EXPORT_FPS = 30
IMAGE_KEYS = dict(head='top', left_wrist='left', right_wrist='right')
IMAGE_SHAPES = dict(head=(360,640,3), left_wrist=(480,640,3), right_wrist=(480,640,3))
INDICES = (*range(15, 22), 31, *range(22, 29), 33)
ROOT = Path(__file__).resolve().parents[2]
ARM_NAMES = json.loads((ROOT/'artifacts/unitree_g1/hold_shadow_profile.json').read_text())['joint_names']
NAMES = ARM_NAMES[:7] + ['left_gripper_motor'] + ARM_NAMES[7:] + ['right_gripper_motor']


def read_jsonl(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def finite(values, size):
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f'Expected {size} finite values')
    return array


def features():
    result = {f'observation.images.{IMAGE_KEYS[role]}': dict(dtype='video', shape=IMAGE_SHAPES[role],
               names=['height','width','channels']) for role in ROLES}
    for key in ('observation.state_joint_angles','observation.velocity','observation.effort','action_joint_angles'):
        result[key] = dict(dtype='float32',shape=(16,),names=NAMES)
    for key in ('observation.state_eef_absolute','action_eef_absolute','action_eef_delta'):
        suffix = 'eef_delta' if key == 'action_eef_delta' else 'eef'
        names = [f'{side}_{suffix}.{component}' for side in ('left','right')
                 for component in ('x','y','z','qw','qx','qy','qz','gripper')]
        result[key] = dict(dtype='float32',shape=(16,),names=names)
    for key in ('q','dq','tau_est'):
        result[f'observation.full_body_{key}'] = dict(dtype='float32',shape=(35,),
                                                      names=[f'motor_{i}' for i in range(35)])
    result['observation.source_timestamps'] = dict(dtype='float64',shape=(6,),
        names=['command_send_started','command_send_finished','state_received',*ROLES])
    result['observation.camera_estimated_read_end'] = dict(dtype='float64',shape=(3,),names=list(ROLES))
    result['observation.camera_sequence'] = dict(dtype='int64',shape=(3,),names=list(ROLES))
    result['observation.robot_tick'] = dict(dtype='int64',shape=(1,),names=['tick'])
    result['next.done'] = dict(dtype='bool',shape=(1,),names=None)
    result['next.success'] = dict(dtype='bool',shape=(1,),names=None)
    return result


def plan_episode(session, *, max_age_s=.2, allow_shadow=False, include_failed=False):
    """Select a contiguous 30Hz grid; never fill gaps or select future observations."""
    metadata = json.loads((session/'session.json').read_text())
    if metadata.get('control_mode') == 'arm_sdk':
        raise ValueError('arm_sdk recordings are raw-only: no verified internal Dex1 action or motor-command semantics')
    if metadata.get('control_mode') not in (None, 'debug'):
        raise ValueError('Unvalidated control_mode: whole-body/simulation recordings require independent export validation')
    summary = json.loads((session/'summary.json').read_text())
    camera_summary = json.loads((session/'cameras/summary.json').read_text())
    recording = metadata['recording']
    if (not recording['enabled'] or recording['fps'] not in (15, 30) or recording['shape'] != [480,640,3]
            or recording['cameras'] != list(ROLES) or not (recording.get('task') or '').strip()):
        raise ValueError('Session is not a supported three-camera training recording')
    if not metadata['hardware_motion_enabled'] and not allow_shadow:
        raise ValueError('Shadow recording cannot be exported as live training data')
    failed = bool(summary.get('error') or summary.get('first_fault') or summary.get('gripper_fault') or camera_summary.get('error')
                  or summary.get('camera_recorder_returncode') != 0)
    if failed and not include_failed:
        raise ValueError('Faulted/incomplete session: raw data retained; use --include-failed intentionally')
    if not math.isfinite(max_age_s) or not 0 < max_age_s <= .2:
        raise ValueError('max_age_s must be in (0, 0.2]')
    commands = [row for row in read_jsonl(session/'commands.jsonl') if row.get('phase') == 'ACTIVE']
    pickup_index = next((i for i,row in enumerate(commands)
                         if all(row.get('gripper_command',{}).get('picked_up',[False,False]))),None)
    if pickup_index is None:
        raise ValueError('No complete dual-gripper activation or training interval exists')
    commands = commands[pickup_index:]
    if len(commands) < 2:
        raise ValueError('No usable ACTIVE interval')
    command_times = [row['command_send_started'] for row in commands]
    if (not all(math.isfinite(t) for t in command_times)
            or any(b <= a for a,b in zip(command_times,command_times[1:]))):
        raise ValueError('Nonmonotonic command timestamps')
    if command_times[-1]-command_times[0] > 180.1:
        raise ValueError('ACTIVE recording exceeds 180 seconds')
    camera_rows = {role: [] for role in ROLES}
    for row in read_jsonl(session/'cameras/frames.jsonl'):
        if row['role'] not in camera_rows:
            raise ValueError('Unknown camera role')
        camera_rows[row['role']].append(row)
    camera_times = {}
    for role, rows in camera_rows.items():
        stamps = [row['host_received'] for row in rows]
        if (not stamps or not all(math.isfinite(t) for t in stamps)
                or any(b <= a for a,b in zip(stamps,stamps[1:]))
                or any(b['sequence'] <= a['sequence'] for a,b in zip(rows,rows[1:]))):
            raise ValueError(f'{role}: absent or nonmonotonic frames')
        camera_times[role] = stamps
    fps = EXPORT_FPS
    count = math.floor((command_times[-1]-command_times[0])*fps+1e-8)+1
    if count < 2:
        raise ValueError('Episode shorter than two dataset frames')
    selected, ages, skews, jitters = [], [], [], []
    camera_root = (session/'cameras').resolve()
    last_command_index = -1
    for index in range(count):
        grid = command_times[0]+index/fps
        ci = bisect_left(command_times,grid-1e-9)
        if ci >= len(commands) or ci <= last_command_index:
            raise ValueError('Command gap/repeated action row on sample grid')
        last_command_index = ci
        row = commands[ci]
        stamp = row['command_send_started']
        if stamp-grid > 1/fps:
            raise ValueError('Command stream gap exceeds a dataset period')
        state = row['observation_before_command']
        if not state or not 0 <= stamp-state['received'] <= max_age_s:
            raise ValueError('Missing, future or stale pre-command state')
        if not 0 <= row['command_send_finished']-stamp <= .05:
            raise ValueError('Invalid/slow action submission')
        motors = {m['index']:m for m in state['motors']}
        if len(state['motors']) != 35 or set(motors) != set(range(35)):
            raise ValueError('Expected all 35 LowState motor slots')
        for key in ('q','dq','tau_est'):
            finite([motors[i][key] for i in range(35)],35)
        if any(m['motorstate'] != 0 for m in motors.values()):
            raise ValueError('Motor error in training observation')
        finite(row['q'],14)
        grip = row['gripper_command']
        finite(grip['q'],2)
        if row.get('gripper_fault') or not all(grip['picked_up']):
            raise ValueError('Gripper fault or missing gripper activation in ACTIVE interval')
        images = []
        for role in ROLES:
            fi = bisect_right(camera_times[role],stamp)-1
            if fi < 0:
                raise ValueError(f'{role}: no causal frame')
            frame = camera_rows[role][fi]
            age = stamp-frame['host_received']
            if not 0 <= age <= max_age_s:
                raise ValueError(f'{role}: stale frame ({age:.4f}s)')
            path = (camera_root/frame['file']).resolve()
            if not path.is_relative_to(camera_root) or not path.is_file():
                raise ValueError('Missing/unsafe camera image path')
            if frame['shape'] != [480,640,3]:
                raise ValueError('Camera shape changed')
            finite([frame['estimated_read_end'],frame['remote_read_start'],frame['remote_read_end']],3)
            if frame['remote_read_end'] < frame['remote_read_start']:
                raise ValueError('Invalid camera read interval')
            images.append((frame,path))
        frame_ages = [stamp-f['host_received'] for f,_ in images]
        ages.append(frame_ages)
        skews.append(max(frame_ages)-min(frame_ages))
        jitters.append(stamp-grid)
        selected.append((row,images))
    age_array = np.asarray(ages)
    quality = dict(frames=count, fps=fps, task=recording['task'], failed_session=failed,
        excluded_active_commands_before_dual_pickup=pickup_index,
        configured_max_age_s=max_age_s,
        hardware_motion_enabled=metadata['hardware_motion_enabled'],
        active_duration_s=command_times[-1]-command_times[0], max_camera_age_s=float(age_array.max()),
        camera_age_p95_s=np.percentile(age_array,95,axis=0).tolist(),
        max_intercamera_receipt_skew_s=max(skews), max_command_grid_jitter_s=max(jitters),
        over_one_period_camera_samples=int((age_array>1/fps).sum()),
        duplicate_camera_samples={role:sum(selected[i][1][j][0]['sequence']==selected[i-1][1][j][0]['sequence']
            for i in range(1,count)) for j,role in enumerate(ROLES)},
        alignment='causal host receipt; remote OpenCV read-completion estimate is diagnostic only',
        exposure_synchronization_verified=False,
        source_session=str(session.resolve()), source_nonce=metadata['nonce'],
        source_hashes={name:hashlib.sha256((session/name).read_bytes()).hexdigest()
            for name in ('session.json','summary.json','commands.jsonl','cameras/frames.jsonl','cameras/metadata.json')})
    return selected, quality


def make_frame(row, images, *, task, last, success):
    motors = {m['index']:m for m in row['observation_before_command']['motors']}
    action = [*row['q'][:7],row['gripper_command']['q'][0],*row['q'][7:],row['gripper_command']['q'][1]]
    frame = {'action_joint_angles':np.asarray(action,dtype=np.float32),'task':task,
        'next.done':np.asarray([last],dtype=bool),'next.success':np.asarray([last and success],dtype=bool),
        'observation.source_timestamps':np.asarray([row['command_send_started'],row['command_send_finished'],
            row['observation_before_command']['received'],*[f['host_received'] for f,_ in images]],dtype=np.float64),
        'observation.camera_estimated_read_end':np.asarray([f['estimated_read_end'] for f,_ in images],dtype=np.float64),
        'observation.camera_sequence':np.asarray([f['sequence'] for f,_ in images],dtype=np.int64),
        'observation.robot_tick':np.asarray([row['observation_before_command']['tick']],dtype=np.int64)}
    for name,key in (('state_joint_angles','q'),('velocity','dq'),('effort','tau_est')):
        frame[f'observation.{name}'] = np.asarray([motors[i][key] for i in INDICES],dtype=np.float32)
        frame[f'observation.full_body_{key}'] = np.asarray([motors[i][key] for i in range(35)],dtype=np.float32)
    frame.update(eef_features([motors[i]['q'] for i in range(35)],row['q'],row['gripper_command']['q']))
    for role,(_,path) in zip(ROLES,images):
        with Image.open(path) as image:
            if image.size != (640,480):
                raise ValueError('Decoded camera image has wrong dimensions')
            image = image.convert('RGB')
            height,width,_ = IMAGE_SHAPES[role]
            if image.size != (width,height):
                image = image.resize((width,height), Image.Resampling.LANCZOS)
            frame[f'observation.images.{IMAGE_KEYS[role]}'] = np.array(image)
    return frame


def convert(sessions, root, repo_id, *, outcome, allow_shadow=False, include_failed=False, max_age_s=.2):
    if importlib.metadata.version('lerobot') != '0.4.4':
        raise RuntimeError('Use the isolated LeRobot 0.4.4 data environment')
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if root.exists():
        raise FileExistsError('Output exists; never overwrite or silently append a dataset')
    plans = [plan_episode(path,max_age_s=max_age_s,allow_shadow=allow_shadow,include_failed=include_failed)
             for path in sessions]
    if outcome not in ('success','failure'):
        raise ValueError('Explicit outcome required')
    if outcome == 'success' and any(q['failed_session'] for _,q in plans):
        raise ValueError('Cannot label a faulted session successful')
    stage = root.with_name(root.name+'.partial-'+uuid.uuid4().hex[:8])
    stage.parent.mkdir(parents=True,exist_ok=True)
    dataset = LeRobotDataset.create(repo_id=repo_id,root=stage,fps=EXPORT_FPS,features=features(),
        robot_type='unitree_g1_dex1',use_videos=True,video_backend='pyav',vcodec='libsvtav1',encoder_threads=2)
    try:
        for selected,quality in plans:
            for i,(row,images) in enumerate(selected):
                dataset.add_frame(make_frame(row,images,task=quality['task'],
                    last=i==len(selected)-1,success=outcome=='success'))
            dataset.save_episode(parallel_encoding=False)
        dataset.finalize()
        reopened = LeRobotDataset(repo_id=repo_id,root=stage,video_backend='pyav')
        if len(reopened) != sum(len(rows) for rows,_ in plans):
            raise ValueError('LeRobot frame count mismatch')
        start = 0
        for selected,_ in plans:
            for index in (start,start+len(selected)-1):
                sample = reopened[index]
                for key in ('observation.state_joint_angles','action_joint_angles',
                            'observation.state_eef_absolute','action_eef_absolute','action_eef_delta'):
                    if tuple(sample[key].shape) != (16,):raise ValueError('LeRobot vector shape mismatch')
                for role in ROLES:
                    if tuple(sample[f'observation.images.{IMAGE_KEYS[role]}'].shape) != (3,*IMAGE_SHAPES[role][:2]):
                        raise ValueError('LeRobot video readback shape mismatch')
            start += len(selected)
        audit = dict(format='LeRobot v3.0',lerobot_version='0.4.4',outcome=outcome,
            converter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            joint_order=NAMES,motor_indices=list(INDICES),
            eef_conventions=conventions(),video_codec='av1',video_encoder='libsvtav1',
            units='arms: rad/rad_s; grippers: native motor position/s; tau_est is uncalibrated motor effort',
            action='absolute position actually submitted after limits/contact logic; zero feedforward',
            sampling='30Hz grid, first submitted action at/after each grid point; source timestamps retained',
            camera_roles=dict(top='USB2.0_Camera_RGB',left='261523071841',right='261523072307'),
            image_transform='top: resize full 640x480 source to 640x360 using Lanczos; left/right unchanged',
            raw_to_export_camera_keys=IMAGE_KEYS,
            episodes=[dict(q,outcome=outcome) for _,q in plans],reader_validation='first/last frame of each episode, all 3 videos')
        for index,session in enumerate(sessions):
            provenance=stage/'meta/g1_sources'/f'episode_{index:06d}'
            provenance.mkdir(parents=True)
            for name in ('session.json','summary.json','cameras/metadata.json','cameras/summary.json'):
                (provenance/name.replace('/','_')).write_bytes((session/name).read_bytes())
        (stage/'meta/g1_export_kinematics.json').write_bytes(MODEL_PATH.read_bytes())
        (stage/'meta/g1_recording.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
        # Re-read is complete before exposing the final dataset directory.
        stage.rename(root)
        return audit
    except BaseException as exc:
        try:dataset.finalize()
        except Exception:pass
        (stage/'conversion_error.json').write_text(json.dumps(dict(error=f'{type(exc).__name__}: {exc}')))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,action='append',required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--repo-id',default='local/unitree_g1_quest')
    parser.add_argument('--outcome',choices=('success','failure'),required=True)
    parser.add_argument('--allow-shadow',action='store_true')
    parser.add_argument('--include-failed',action='store_true')
    parser.add_argument('--max-age-s',type=float,default=.2)
    args = parser.parse_args()
    audit = convert(args.session,args.root,args.repo_id,outcome=args.outcome,
        allow_shadow=args.allow_shadow,include_failed=args.include_failed,max_age_s=args.max_age_s)
    print(json.dumps(dict(root=str(args.root),episodes=len(audit['episodes']),
                         frames=sum(q['frames'] for q in audit['episodes'])),indent=2))


if __name__ == '__main__':
    main()
