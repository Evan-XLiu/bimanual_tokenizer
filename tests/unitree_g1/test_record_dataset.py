"""Synthetic recording fixtures. No camera, SSH or DDS is started by these tests."""
import io
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/unitree_g1'))
from convert_dataset import IMAGE_KEYS, IMAGE_SHAPES, INDICES, ROLES, convert, make_frame, plan_episode
from record_cameras import check_camera_status, read_packet, record


def fixture(root, frames=5):
    (root/'cameras').mkdir(parents=True)
    session = dict(nonce='synthetic-offline-fixture', hardware_motion_enabled=False,
        recording=dict(enabled=True,fps=30,shape=[480,640,3],cameras=list(ROLES),task='pick up the test block'))
    (root/'session.json').write_text(json.dumps(session))
    (root/'summary.json').write_text(json.dumps(dict(error=None,camera_recorder_returncode=0)))
    (root/'cameras/summary.json').write_text(json.dumps(dict(error=None)))
    (root/'cameras/metadata.json').write_text(json.dumps(dict(synthetic=True)))
    commands, cameras = [], []
    for i in range(frames):
        stamp = 10+i/30
        motors = [dict(index=j,q=j*.01+i*.001,dq=j*.001,tau_est=j*.002,motorstate=0) for j in range(35)]
        commands.append(dict(phase='ACTIVE',command_send_started=stamp,command_send_finished=stamp+.001,
            q=[j*.01+.1 for j in range(14)],gripper_command=dict(q=[-.3,-.4],picked_up=[True,True]),
            gripper_fault=None,observation_before_command=dict(received=stamp-.002,tick=i+100,motors=motors)))
        for j,role in enumerate(ROLES):
            path=root/'cameras'/role/f'{i+1:08d}.jpg'
            path.parent.mkdir(exist_ok=True)
            color=[0,0,0];color[j]=180+i
            Image.new('RGB',(640,480),tuple(color)).save(path)
            cameras.append(dict(role=role,file=f'{role}/{i+1:08d}.jpg',sequence=i+1,shape=[480,640,3],
                host_received=stamp-.01+j*.001,estimated_read_end=stamp-.02,
                remote_read_start=stamp-100-.025,remote_read_end=stamp-100-.02))
    (root/'commands.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in commands))
    (root/'cameras/frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cameras))
    return commands,cameras


class RecordingTests(unittest.TestCase):
    def test_unvalidated_whole_body_modes_cannot_use_legacy_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root)
            metadata = json.loads((root/'session.json').read_text())
            for mode in ('whole_body', 'whole_body_simulation', 'future_unknown'):
                metadata['control_mode'] = mode
                (root/'session.json').write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, 'Unvalidated control_mode'):
                    plan_episode(root, allow_shadow=True, include_failed=True)

    def test_legacy_15hz_recording_exports_30hz_with_causal_image_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,cameras=fixture(root)
            metadata=json.loads((root/'session.json').read_text())
            metadata['recording']['fps']=15
            (root/'session.json').write_text(json.dumps(metadata))
            cameras=[r for r in cameras if r['sequence'] % 2 == 1]
            (root/'cameras/frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cameras))
            rows,quality=plan_episode(root,allow_shadow=True)
            self.assertEqual(quality['fps'],30)
            self.assertEqual(quality['frames'],5)
            self.assertEqual(quality['duplicate_camera_samples'],dict.fromkeys(ROLES,2))
            for row,images in rows:
                self.assertTrue(all(f['host_received'] <= row['command_send_started'] for f,_ in images))

    def test_mock_ssh_records_three_streams_and_stops_cleanly(self):
        import signal
        import threading
        import time
        from types import SimpleNamespace
        from unittest.mock import patch
        closed=threading.Event()
        now=time.monotonic()
        def packet(header,payload=b''):
            h=json.dumps(dict(header,bytes=len(payload))).encode()
            return struct.pack('!I',len(h))+h+payload
        image=io.BytesIO();Image.new('RGB',(640,480),(1,2,3)).save(image,format='JPEG')
        data=packet(dict(type='hello',devices=dict.fromkeys(ROLES,{})))
        data+=b''.join(packet(dict(type='pong',remote_monotonic=now)) for _ in range(8))
        data+=b''.join(packet(dict(type='frame',role=role,sequence=1,shape=[480,640,3],
            remote_read_start=now-.01,remote_read_end=now),image.getvalue()) for role in ROLES)
        class Reader(io.BytesIO):
            def read(self,size=-1):
                result=super().read(size)
                if not result:closed.wait(2)
                return result
        class Writer(io.BytesIO):
            def close(self):closed.set();super().close()
        class SSH:
            stdin=Writer();stdout=Reader(data)
            def wait(self,timeout):closed.wait(timeout);return 0
            def terminate(self):closed.set()
            def kill(self):closed.set()
        handlers={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
        try:
            with tempfile.TemporaryDirectory() as tmp,patch('record_cameras.subprocess.Popen',return_value=SSH()) as launch:
                output=Path(tmp)/'camera_recording'
                record(SimpleNamespace(output=output,ssh_key=Path('/unused/test-key'),host='unused',duration=.15))
                self.assertEqual(launch.call_count,1)
                rows=[json.loads(line) for line in (output/'frames.jsonl').read_text().splitlines()]
                self.assertEqual({row['role'] for row in rows},set(ROLES))
                self.assertTrue(all((output/row['file']).is_file() for row in rows))
                self.assertIsNone(json.loads((output/'summary.json').read_text())['error'])
                self.assertFalse(json.loads((output/'status.json').read_text())['ready'])
        finally:
            for sig,handler in handlers.items():signal.signal(sig,handler)

    def test_protocol_and_status_detect_corruption_or_missing_camera(self):
        header=json.dumps(dict(bytes=3,role='head')).encode()
        decoded,payload=read_packet(io.BytesIO(struct.pack('!I',len(header))+header+b'abc'))
        self.assertEqual(decoded['role'],'head');self.assertEqual(payload,b'abc')
        with self.assertRaises(EOFError):read_packet(io.BytesIO(b'\0'))
        with self.assertRaises(ValueError):read_packet(io.BytesIO(struct.pack('!I',999999)))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'status.json'
            status=dict(time=10.,ready=True,latest=dict.fromkeys(ROLES,10.),error=None)
            path.write_text(json.dumps(status));check_camera_status(path,10.1)
            with self.assertRaises(ValueError):check_camera_status(path,11.)
            status['latest'].pop('head');path.write_text(json.dumps(status))
            with self.assertRaises(ValueError):check_camera_status(path,10.1)

    def test_mapping_causal_actions_rgb_and_episode_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            rows,quality=plan_episode(root,allow_shadow=True)
            frame=make_frame(*rows[0],task=quality['task'],last=False,success=True)
            np.testing.assert_allclose(frame['observation.state_joint_angles'],np.asarray(INDICES)*.01)
            self.assertAlmostEqual(frame['action_joint_angles'][7],-.3,places=6)
            self.assertAlmostEqual(frame['action_joint_angles'][15],-.4,places=6)
            self.assertFalse(np.array_equal(frame['action_joint_angles'],frame['observation.state_joint_angles']))
            self.assertEqual(frame['observation.full_body_q'].shape,(35,))
            self.assertGreater(frame['observation.images.top'][100,100,0],170)
            self.assertLess(frame['observation.images.top'][100,100,2],10)
            self.assertFalse(frame['next.done'][0])
            self.assertEqual(quality['frames'],5)
            self.assertEqual(quality['duplicate_camera_samples'],dict.fromkeys(ROLES,0))
            with self.assertRaisesRegex(ValueError,'Shadow'):plan_episode(root)

    def test_future_stale_missing_and_post_command_state_rejected(self):
        for case in ('future','stale','missing','post_state','nan_action','sequence','fault'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);commands,cameras=fixture(root)
                if case=='future':
                    for row in cameras:row['host_received']+=1.
                elif case=='stale':
                    for row in cameras:row['host_received']-=1.
                elif case=='missing':cameras=[row for row in cameras if row['role']!='head']
                elif case=='post_state':commands[0]['observation_before_command']['received']=11.
                elif case=='nan_action':commands[0]['q'][0]=math.nan
                elif case=='sequence':cameras[-1]['sequence']=1
                elif case=='fault':
                    (root/'summary.json').write_text(json.dumps(dict(error='tracking_error',camera_recorder_returncode=0)))
                (root/'commands.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in commands))
                (root/'cameras/frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cameras))
                with self.assertRaises(ValueError):plan_episode(root,allow_shadow=True)

    def test_point_two_age_limit_accepts_older_data_but_remains_bounded(self):
        for source in ('state','camera'):
            with self.subTest(source=source),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);commands,cameras=fixture(root,frames=2)
                if source=='state':
                    for row in commands:row['observation_before_command']['received']=row['command_send_started']-.15
                else:
                    for row in cameras:row['host_received']-=.14
                def save():
                    (root/'commands.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in commands))
                    (root/'cameras/frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cameras))
                save()
                _,quality=plan_episode(root,allow_shadow=True)
                self.assertEqual(quality['configured_max_age_s'],.2)
                with self.assertRaises(ValueError):plan_episode(root,allow_shadow=True,max_age_s=.1)
                with self.assertRaises(ValueError):plan_episode(root,allow_shadow=True,max_age_s=.201)
                if source=='state':
                    for row in commands:row['observation_before_command']['received']-=.1
                else:
                    for row in cameras:row['host_received']-=.1
                save()
                with self.assertRaises(ValueError):plan_episode(root,allow_shadow=True)

    def test_pickup_prefix_trim_and_large_action_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);commands,_=fixture(root)
            commands[0]['gripper_command']['picked_up']=[True,False]
            (root/'commands.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in commands))
            rows,q=plan_episode(root,allow_shadow=True)
            self.assertEqual(q['excluded_active_commands_before_dual_pickup'],1)
            self.assertEqual(len(rows),4)
            commands=commands[:2]+commands[-1:]
            (root/'commands.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in commands))
            with self.assertRaisesRegex(ValueError,'gap'):plan_episode(root,allow_shadow=True)

    def test_180_second_sampling_grid_without_hardware_or_large_video_files(self):
        import copy
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);commands,cameras=fixture(root,frames=2)
            with (root/'commands.jsonl').open('w') as command_log, (root/'cameras/frames.jsonl').open('w') as camera_log:
                for i in range(5400):
                    stamp=10+i/30
                    row=copy.deepcopy(commands[0])
                    row.update(command_send_started=stamp,command_send_finished=stamp+.001)
                    row['observation_before_command'].update(received=stamp-.002,tick=i+100)
                    command_log.write(json.dumps(row)+'\n')
                    for source in cameras[:3]:
                        frame=dict(source,sequence=i+1,host_received=stamp-.01,
                                   estimated_read_end=stamp-.02)
                        camera_log.write(json.dumps(frame)+'\n')
            selected,quality=plan_episode(root,allow_shadow=True)
            self.assertEqual(len(selected),5400)
            self.assertEqual(quality['fps'],30)
            self.assertLess(quality['active_duration_s'],180.)
            row['command_send_started']=191.
            with (root/'commands.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'180 seconds'):plan_episode(root,allow_shadow=True)


class LeRobotRoundTripTests(unittest.TestCase):
    def test_two_episodes_videos_parquet_and_official_reader(self):
        try:
            import importlib.metadata
            if importlib.metadata.version('lerobot')!='0.4.4':self.skipTest('Requires data env LeRobot 0.4.4')
        except importlib.metadata.PackageNotFoundError:self.skipTest('Use .venv-g1-data for real writer test')
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            a,b=tmp/'a',tmp/'b';fixture(a);fixture(b,frames=3)
            root=tmp/'dataset'
            audit=convert([a,b],root,'local/g1_offline_test',outcome='success',allow_shadow=True)
            ds=LeRobotDataset('local/g1_offline_test',root=root,video_backend='pyav')
            self.assertEqual(len(ds),8)
            self.assertEqual(ds.num_episodes,2)
            self.assertEqual(ds.meta.info['codebase_version'],'v3.0')
            self.assertEqual(ds.meta.info['fps'],30)
            self.assertEqual({k for k in ds.meta.info['features'] if k.startswith('observation.images.')},
                             {'observation.images.top','observation.images.left','observation.images.right'})
            self.assertEqual(len(list(root.rglob('*.mp4'))),3)
            import av
            for path in root.rglob('*.mp4'):
                with av.open(str(path)) as video:
                    self.assertEqual(video.streams.video[0].codec_context.codec.id,av.codec.Codec('av1','r').id)
            self.assertTrue((root/'meta/g1_export_kinematics.json').is_file())
            for i in range(len(ds)):
                sample=ds[i]
                self.assertEqual(sample['action_joint_angles'].shape,(16,))
                for key in ('observation.state_eef_absolute','action_eef_absolute','action_eef_delta'):
                    self.assertEqual(sample[key].shape,(16,))
                    self.assertTrue(np.isfinite(sample[key].numpy()).all())
                self.assertEqual(sample['observation.full_body_q'].shape,(35,))
                self.assertEqual(sample['next.done'].item(),i in (4,7))
                self.assertEqual(sample['next.success'].item(),i in (4,7))
                self.assertEqual(sample['episode_index'].item(),0 if i<5 else 1)
                for j,role in enumerate(ROLES):
                    image=sample[f'observation.images.{IMAGE_KEYS[role]}']
                    self.assertEqual(image.shape,(3,*IMAGE_SHAPES[role][:2]))
                    self.assertGreater(image[j,100,100].item(),.6)
            self.assertFalse(audit['episodes'][0]['hardware_motion_enabled'])
            with self.assertRaises(FileExistsError):convert([a],root,'local/g1_offline_test',outcome='failure',allow_shadow=True)


if __name__ == '__main__':
    unittest.main()
