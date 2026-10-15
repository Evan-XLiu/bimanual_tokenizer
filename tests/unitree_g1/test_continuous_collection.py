"""No hardware: routing, discard boundaries, count semantics, faults, persistent VR."""
import io
import json
import socket
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/unitree_g1'))
from continuous_input import RoutedEvents, write_json, check_service
from collect_episodes import Collector, discard_attempt, cleanup_confirmed, save_attempt, validate_args
from collect_episodes import ensure_robot_network, check_vr_address


class VRAddressTests(unittest.TestCase):
    def test_active_listener_is_rejected(self):
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('127.0.0.1', 0))
            server.listen(1)
            with self.assertRaises(OSError):
                check_vr_address(*server.getsockname())

    def test_closed_server_with_unclosed_peer_can_be_reused(self):
        with socket.socket() as server, socket.socket() as client:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('127.0.0.1', 0))
            address = server.getsockname()
            server.listen(1)
            client.connect(address)
            connection, _ = server.accept()
            connection.close()
            server.close()
            self.assertEqual(client.recv(1), b'')
            check_vr_address(*address)


class NetworkTests(unittest.TestCase):
    def run_network(self,addresses,settings=None):
        settings=settings or {}
        def run(command,**kwargs):
            if command[0]=='ip':
                value=next(addresses)
                output=json.dumps([dict(addr_info=[dict(family='inet',local=a) for a in value])])
            elif '-g' in command:
                output=settings[command[2]]
            else:
                output='activated'
            return SimpleNamespace(stdout=output,stderr='',returncode=0)
        return patch('collect_episodes.subprocess.run',side_effect=run)

    def test_ready_network_needs_no_mutation(self):
        with self.run_network(iter([['192.168.123.223']])) as run:
            self.assertFalse(ensure_robot_network('enp3s0','g1')['activated'])
            self.assertEqual(run.call_count,1)

    def test_missing_address_activates_and_verifies_profile(self):
        settings={'connection.interface-name':'enp3s0','ipv4.method':'manual',
                  'ipv4.never-default':'yes','ipv4.gateway':'','ipv4.addresses':'192.168.123.223/24'}
        with self.run_network(iter([[],['192.168.123.223']]),settings) as run:
            self.assertTrue(ensure_robot_network('enp3s0','g1')['activated'])
            self.assertIn(['nmcli','--wait','20','connection','up','id','g1','ifname','enp3s0'],
                          [call.args[0] for call in run.call_args_list])

    def test_unrelated_address_and_wrong_interface_are_not_changed(self):
        for addresses,settings in [([['10.0.0.2']],{}),([[]],{'connection.interface-name':'other'})]:
            with self.run_network(iter(addresses),settings) as run:
                with self.assertRaises(RuntimeError):ensure_robot_network('enp3s0','g1')
                self.assertFalse(any('up' in call.args[0] for call in run.call_args_list))


def clean_summary():
    return dict(error=None,arm_sdk_release_submitted=True,external_gripper_zero_submitted=[True,True],
                camera_recorder_returncode=0)


class RoutingTests(unittest.TestCase):
    def test_input_main_routed_mode_writes_only_attempt_and_closes(self):
        import televuer_input
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);service=root/'vr';attempt=root/'pending/attempt_000001';attempt.mkdir(parents=True)
            cert=root/'cert';key=root/'key';cert.touch();key.touch()
            closed=[]
            def create(args,events,dropped):
                now=time.monotonic()
                write_json(service/'route.json',dict(output=str(attempt/'input'),nonce='one',since=now))
                for kind in ('CAMERA_MOVE','CONTROLLER_MOVE'):
                    events.put(json.dumps(dict(kind=kind,session='persistent-websocket',valid=True,
                                               host_receive_monotonic_s=time.monotonic(),raw={})))
                return SimpleNamespace(process=SimpleNamespace(is_alive=lambda:True),close=lambda:closed.append(True))
            argv=['input','--cert',str(cert),'--key',str(key),'--output',str(service),'--routed','--duration','.15']
            with patch('sys.argv',argv),patch('televuer_input.create_tv',side_effect=create):
                self.assertEqual(televuer_input.main(),0)
            self.assertEqual(len((attempt/'input/events.jsonl').read_text().splitlines()),2)
            self.assertFalse((service/'events.jsonl').exists())
            self.assertTrue(closed)
            self.assertEqual(json.loads((service/'route_status.json').read_text())['route']['nonce'],'one')

    def test_one_service_routes_two_episodes_and_discards_idle_and_old_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);service=run/'vr';service.mkdir();(run/'pending').mkdir()
            (service/'session.json').write_text('{}')
            write_json(service/'route.json',{})
            router=RoutedEvents(service)
            event=lambda t:json.dumps(dict(host_receive_monotonic_s=t))
            router.write(event(1.))
            self.assertFalse((service/'events.jsonl').exists())
            for i in (1,2):
                attempt=run/'pending'/f'attempt_{i:06d}';attempt.mkdir()
                route=dict(output=str(attempt/'input'),nonce=str(i),since=10.*i)
                write_json(service/'route.json',route)
                router.write(event(10.*i-1))
                router.write(event(10.*i+1))
                write_json(service/'route.json',{})
                router.sync()
                self.assertIsNone(router.stream)
                self.assertEqual(json.loads((service/'route_status.json').read_text())['route'],{})
                lines=(attempt/'input/events.jsonl').read_text().splitlines()
                self.assertEqual(len(lines),1)
                self.assertEqual(json.loads(lines[0])['host_receive_monotonic_s'],10.*i+1)
                if i==1:discard_attempt(run,attempt)
            self.assertFalse((run/'pending/attempt_000001').exists())
            self.assertTrue((run/'pending/attempt_000002/input/events.jsonl').exists())

    def test_route_escape_and_stale_or_dropped_service_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);service=p/'vr';service.mkdir();(service/'session.json').write_text('{}')
            write_json(service/'route.json',dict(output=str(p/'outside/input'),nonce='x',since=0))
            with self.assertRaises(ValueError):RoutedEvents(service).sync()
            for status in (dict(time=time.monotonic()-10,queue_dropped=0),dict(time=time.monotonic(),queue_dropped=1)):
                write_json(service/'status.json',status)
                with self.assertRaises(RuntimeError):check_service(service)


class CollectorTests(unittest.TestCase):
    def test_real_lerobot_export_after_acceptance_uses_final_source_path(self):
        import importlib.metadata
        try:
            if importlib.metadata.version('lerobot')!='0.4.4':self.skipTest('Requires data environment')
        except importlib.metadata.PackageNotFoundError:self.skipTest('Requires data environment')
        from test_record_dataset import fixture
        from convert_dataset import convert
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'episodes').mkdir()
            attempt=root/'pending/attempt_000001';fixture(attempt,frames=3)
            def export(destination,**kwargs):
                convert([destination],destination/'lerobot','local/accepted_fixture',outcome='success',allow_shadow=True)
                return dict(exported=True)
            destination=save_attempt(root,attempt,0)
            self.assertFalse((destination/'lerobot').exists())
            export(destination)
            ds=LeRobotDataset('local/accepted_fixture',root=destination/'lerobot',video_backend='pyav')
            self.assertEqual(len(ds),3)
            self.assertTrue(ds[2]['next.success'].item())
            provenance=json.loads((destination/'lerobot/meta/g1_recording.json').read_text())
            self.assertEqual(provenance['episodes'][0]['source_session'],str(destination.resolve()))
            self.assertFalse(attempt.exists())

    def args(self,root,**overrides):
        values=dict(output=root/'run',episodes=2,task='test',active_seconds=180.,wait_seconds=300.,
            camera_host='robot',camera_ssh_key=root/'key',interface='test0',
            min_free_gb=1.,execute=True,preflight_only=False,scene_confirmation='True')
        values.update(overrides)
        return SimpleNamespace(**values)

    def fake_collector(self,root,choices,summary=None,export_ok=True,preflight_only=False,fresh=True,starts=None,
                       repairs=None,summaries=None,repair_failures=0):
        collector=Collector(self.args(root,preflight_only=preflight_only,execute=not preflight_only))
        collector.device_preflight=lambda index:None
        collector.camera_preflight=lambda:None
        collector.decision=lambda:next(choices)
        collector.start_prompts=[]
        def wait_for_start():
            collector.start_prompts.append(collector.state['phase'])
            return next(starts) if starts is not None else 'n'
        collector.wait_for_start=wait_for_start
        collector.repair_prompts=[]
        def prompt(message,allowed,**kwargs):
            collector.repair_prompts.append(collector.state['phase'])
            return next(repairs) if repairs is not None else 'q'
        collector.prompt=prompt
        def camera_check():
            nonlocal repair_failures
            if collector.state['phase']=='checking_repair' and repair_failures:
                repair_failures-=1
                raise RuntimeError('Camera still unavailable')
        collector.camera_preflight=camera_check
        collector.episode_command=lambda attempt,service:['record',str(attempt)]
        calls=[]
        def start():
            calls.append('start_vr')
            process=SimpleNamespace(returncode=None)
            process.poll=lambda:process.returncode
            process.terminate=lambda:setattr(process,'returncode',0)
            process.wait=lambda timeout:process.returncode
            collector.input_process=process
            collector.input_log=io.StringIO()
            return collector.args.output/'vr'
        collector.start_input=start
        def run(command,log_path,timeout=None):
            if command[0]!='record':return 0
            attempt=Path(command[1]);attempt.mkdir()
            write_json(attempt/'summary.json',next(summaries) if summaries is not None else summary or clean_summary())
            (attempt/'raw.jpg').write_bytes(b'fixture')
            calls.append(str(attempt))
            return 0
        collector.run_process=run
        def exporter(destination,**kwargs):
            self.assertEqual(collector.input_process.poll(),0)
            self.assertIn(collector.state['phase'],('exporting_batch',))
            self.assertTrue(all(Path(p).is_dir() for p in collector.state['accepted']))
            if export_ok:(destination/'lerobot').mkdir()
            return dict(exported=export_ok)
        with patch('collect_episodes.ensure_service',return_value={}) as ensure,patch('collect_episodes.local_preflight',return_value={}),patch('collect_episodes.set_route'),patch('collect_episodes.check_service',return_value=dict(receive_fresh=fresh)),patch('collect_episodes.export_recording',side_effect=exporter):
            code=collector.run()
            self.assertEqual(ensure.call_count,0 if preflight_only else 1)
        return collector,code,calls

    def test_s_f_s_counts_only_two_saved_and_keeps_vr_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,code,calls=self.fake_collector(Path(tmp),iter(['s','f','s']))
            self.assertEqual(code,0)
            self.assertEqual(c.state['phase'],'complete')
            self.assertEqual(c.state['accepted_count'],2)
            self.assertEqual([r['status'] for r in c.state['attempts']],['saved','discarded','saved'])
            self.assertEqual(calls.count('start_vr'),1)
            self.assertEqual(c.start_prompts,['awaiting_n']*3)
            self.assertTrue(all(r['start_confirmation']=='n' for r in c.state['attempts']))
            self.assertEqual(len(calls),4)
            self.assertEqual(list((c.args.output/'pending').iterdir()),[])
            self.assertTrue((c.args.output/'episodes/episode_000001/lerobot').is_dir())

    def test_quit_before_n_does_not_start_first_or_retry_attempt(self):
        for decisions,starts,attempt_count in (([],['q'],0),(['f'],['n','q'],1),(['s'],['n','q'],1)):
            with self.subTest(starts=starts,decisions=decisions),tempfile.TemporaryDirectory() as tmp:
                c,code,calls=self.fake_collector(Path(tmp),iter(decisions),starts=iter(starts))
                self.assertEqual(code,0)
                self.assertEqual(c.state['phase'],'stopped')
                self.assertEqual(len(c.state['attempts']),attempt_count)
                self.assertEqual(len(calls),1+attempt_count)

    def test_start_prompt_requires_n_and_accepts_quit_eof_or_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            for text,expected in (('s\nf\n\nn\n','n'),('q\n','q'),('','q')):
                c=Collector(self.args(Path(tmp)))
                c.input_process=SimpleNamespace(poll=lambda:None)
                stream=io.StringIO(text)
                with patch('collect_episodes.sys.stdin',stream),patch('collect_episodes.select.select',return_value=([stream],[],[])):
                    self.assertEqual(c.wait_for_start(),expected)
            c.stop=True
            self.assertEqual(c.wait_for_start(),'q')

    def test_stale_headset_cannot_start_next_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,code,calls=self.fake_collector(Path(tmp),iter([]),fresh=False)
            self.assertEqual(code,0)
            self.assertEqual(c.repair_prompts,['awaiting_repair'])
            self.assertEqual(calls,['start_vr'])
            self.assertEqual(c.state['attempts'],[])

    def test_interrupt_requests_graceful_child_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=Collector(self.args(Path(tmp)));c.stop=True
            class Process:
                returncode=None
                terminated=False
                def poll(self):return self.returncode
                def terminate(self):self.terminated=True;self.returncode=0
                def wait(self,timeout):return self.returncode
            process=Process()
            with patch('collect_episodes.subprocess.Popen',return_value=process):
                with self.assertRaises(InterruptedError):
                    c.run_process(['fake'],Path(tmp)/'child.log')
            self.assertTrue(process.terminated)
            self.assertIsNone(c.child)

    def test_fault_does_not_prompt_or_rearm_and_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary=dict(clean_summary(),error='tracking_error')
            c,code,calls=self.fake_collector(Path(tmp),iter([]),summary=summary)
            self.assertEqual(code,0)
            self.assertEqual(c.repair_prompts,['awaiting_repair'])
            self.assertEqual(len(calls),2)
            self.assertEqual(c.state['accepted_count'],0)
            self.assertTrue((c.args.output/'pending/attempt_000001/raw.jpg').is_file())

    def test_camera_fault_rechecks_then_requires_new_n_and_preserves_failed_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            fault=dict(clean_summary(),error='camera stopped',camera_recorder_returncode=1)
            c,code,calls=self.fake_collector(Path(tmp),iter(['s','s']),
                summaries=iter([fault,clean_summary(),clean_summary()]),
                repairs=iter(['r','r']),repair_failures=1,starts=iter(['n','n','n']))
            self.assertEqual(code,0)
            self.assertEqual(c.state['phase'],'complete')
            self.assertEqual(c.state['accepted_count'],2)
            self.assertEqual(c.start_prompts,['awaiting_n']*3)
            self.assertEqual(c.repair_prompts,['awaiting_repair']*2)
            self.assertEqual([x['passed'] for x in c.state['recovery_checks']],[False,True])
            self.assertEqual(calls.count('start_vr'),1)
            self.assertTrue((c.args.output/'pending/attempt_000001/collection_fault.json').is_file())
            self.assertEqual(c.state['attempts'][0]['status'],'hardware_or_cleanup_fault')
            self.assertTrue(c.state['attempts'][0]['recovery_preflight_passed'])

    def test_explicit_failure_before_control_can_recheck(self):
        summary = dict(error='Stale body feedback', command_frames=0,
                       control_publish_not_started=True, camera_recorder_returncode=0)
        self.assertTrue(cleanup_confirmed(summary))
        for change in (dict(control_publish_not_started=False), dict(command_frames=1),
                       dict(unload_error='fault'), dict(external_gripper_cleanup_error='fault')):
            self.assertFalse(cleanup_confirmed(dict(summary, **change)))
        with tempfile.TemporaryDirectory() as tmp:
            c, code, calls = self.fake_collector(Path(tmp), iter([]), summary=summary,
                repairs=iter(['r']), starts=iter(['n', 'q']))
            self.assertEqual(code, 0)
            self.assertTrue(c.state['recovery_checks'][0]['passed'])
            self.assertEqual(len(c.state['attempts']), 1)

    def test_unconfirmed_motor_or_camera_cleanup_never_rearms(self):
        for changes in (dict(arm_sdk_release_submitted=False),
                        dict(external_gripper_zero_submitted=[True,False]),
                        dict(unload_error='failed'),dict(camera_recorder_returncode=None)):
            with self.subTest(changes=changes),tempfile.TemporaryDirectory() as tmp:
                summary=dict(clean_summary(),error='fault',**changes)
                c,code,calls=self.fake_collector(Path(tmp),iter([]),summary=summary,
                    repairs=iter(['r','r','q']),starts=iter(['n']))
                self.assertEqual(code,0)
                self.assertEqual(len(c.state['attempts']),1)
                self.assertEqual(c.state['accepted_count'],0)
                self.assertEqual([x['passed'] for x in c.state['recovery_checks']],[False,False])
                self.assertTrue((c.args.output/'pending/attempt_000001/raw.jpg').exists())

    def test_successful_recheck_alone_does_not_start_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,code,calls=self.fake_collector(Path(tmp),iter([]),
                summary=dict(clean_summary(),error='tracking_error'),
                repairs=iter(['r']),starts=iter(['n','q']))
            self.assertEqual(code,0)
            self.assertEqual(len(c.state['attempts']),1)
            self.assertTrue(c.state['recovery_checks'][0]['passed'])
            self.assertEqual(len(calls),2)

    def test_missing_summary_cannot_be_cleared_by_sensor_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=Collector(self.args(Path(tmp)))
            c.args.output.mkdir()
            with patch.object(c,'prompt',side_effect=['r','q']),patch.object(c,'device_preflight') as check:
                self.assertFalse(c.wait_for_repair(Path(tmp)/'vr',dict(path=str(Path(tmp)/'missing'))))
                check.assert_not_called()
                self.assertFalse(c.state['recovery_checks'][0]['passed'])

    def test_repeated_device_checks_have_unique_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=Collector(self.args(Path(tmp)))
            (c.args.output/'checks').mkdir(parents=True)
            paths=[]
            def run(command,log_path,timeout):
                path=Path(command[command.index('--output')+1])
                self.assertFalse(path.exists())
                write_json(path,dict(preflight_passed=True))
                paths.append(path)
                return 0
            with patch.object(c,'run_process',side_effect=run):
                c.device_preflight(1)
                c.device_preflight(1)
            self.assertNotEqual(paths[0],paths[1])

    def test_export_failure_keeps_accepted_raw_data_and_runs_only_after_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,code,calls=self.fake_collector(Path(tmp),iter(['s','s']),export_ok=False)
            self.assertEqual(code,1)
            self.assertEqual(c.state['accepted_count'],2)
            self.assertEqual(c.state['exported_count'],0)
            self.assertEqual(c.state['phase'],'export_failed')
            self.assertEqual(len(c.state['export_results']),2)
            self.assertEqual(len(calls),3)
            self.assertTrue((c.args.output/'episodes/episode_000000/raw.jpg').is_file())
            self.assertTrue((c.args.output/'episodes/episode_000001/raw.jpg').is_file())

    def test_quit_exports_only_previously_accepted_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,code,calls=self.fake_collector(Path(tmp),iter(['s']),starts=iter(['n','q']))
            self.assertEqual(code,0)
            self.assertEqual(c.state['accepted_count'],1)
            self.assertEqual(c.state['exported_count'],1)
            self.assertEqual(c.state['phase'],'stopped')
            self.assertEqual(len(c.state['export_results']),1)

    def test_quit_after_fault_exports_prior_acceptance_not_faulted_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,code,_=self.fake_collector(Path(tmp),iter(['s']),
                summaries=iter([clean_summary(),dict(clean_summary(),error='input stale')]),
                repairs=iter(['q']),starts=iter(['n','n']))
            self.assertEqual(code,0)
            self.assertEqual(c.state['accepted_count'],1)
            self.assertEqual(c.state['exported_count'],1)
            self.assertFalse((c.args.output/'pending/attempt_000002/lerobot').exists())

    def test_ctrl_c_end_still_exports_accepted_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=Collector(self.args(Path(tmp)))
            episode=c.args.output/'episodes/episode_000000'
            episode.mkdir(parents=True)
            def collect():
                c.stop=True
                c.state['accepted']=[str(episode)]
                c.checkpoint('stopped')
                return 1
            with patch.object(c,'collect',side_effect=collect),patch('collect_episodes.export_recording',return_value=dict(exported=True)) as export:
                self.assertEqual(c.run(),1)
                export.assert_called_once_with(episode,outcome='success')
                self.assertEqual(c.state['exported_count'],1)
                self.assertEqual(c.state['phase'],'stopped')

    def test_q_and_preflight_only_do_not_start_next(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,code,calls=self.fake_collector(Path(tmp),iter(['q']))
            self.assertEqual(code,0);self.assertEqual(c.state['phase'],'stopped')
            self.assertTrue((c.args.output/'pending/attempt_000001/raw.jpg').is_file())
        with tempfile.TemporaryDirectory() as tmp:
            c,code,calls=self.fake_collector(Path(tmp),iter([]),preflight_only=True)
            self.assertEqual(code,0);self.assertEqual(c.state['phase'],'preflight_passed')
            self.assertEqual(calls,[])

    def test_discard_is_restricted_and_cleanup_must_be_confirmed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);outside=root/'outside';outside.mkdir();(root/'pending').mkdir()
            link=root/'pending/attempt_symlink';link.symlink_to(outside,target_is_directory=True)
            for path in (outside,link,root/'pending'):
                with self.assertRaises(ValueError):discard_attempt(root,path)
            self.assertTrue(outside.exists())
        self.assertTrue(cleanup_confirmed(clean_summary()))
        for change in (dict(arm_sdk_release_submitted=False),dict(external_gripper_zero_submitted=[True,False]),
                       dict(camera_recorder_returncode=1),dict(input_route_cleanup_error='timeout')):
            self.assertFalse(cleanup_confirmed(dict(clean_summary(),**change)))

    def test_invalid_config_cannot_start(self):
        with tempfile.TemporaryDirectory() as tmp,patch('sys.stdin.isatty',return_value=True):
            args=self.args(Path(tmp));validate_args(args)
            for changes in (dict(episodes=0),dict(active_seconds=float('nan')),dict(task=' '),
                            dict(scene_confirmation=''),dict(scene_confirmation='False'),
                            dict(scene_confirmation=None),dict(scene_confirmation='test fixture only'),
                            dict(execute=False,preflight_only=False)):
                with self.assertRaises(ValueError):validate_args(self.args(Path(tmp),**changes))


if __name__=='__main__':unittest.main()
