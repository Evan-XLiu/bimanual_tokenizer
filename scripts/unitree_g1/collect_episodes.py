#!/usr/bin/env python3
"""Continuous Quest collection: n/start each attempt, s/save or f/discard."""
import argparse
import ipaddress
import json
import math
import os
from pathlib import Path
import select
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import threading
import time

from continuous_input import check_service, set_route, write_json
from export_recording import export_recording
from dex1_service import ensure_service

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT/'scripts/unitree_g1'


def read_json(path):
    return json.loads(Path(path).read_text())


def ensure_robot_network(interface, connection):
    """Activate an existing dedicated NM profile when the robot subnet is absent."""
    subnet=ipaddress.ip_network('192.168.123.0/24')
    def addresses():
        result=subprocess.run(['ip','-j','-4','addr','show','dev',interface],
                              capture_output=True,text=True,check=True,timeout=10)
        return [entry['local'] for device in json.loads(result.stdout)
                for entry in device.get('addr_info',[]) if entry.get('family')=='inet']
    current=addresses()
    if any(ipaddress.ip_address(address) in subnet for address in current):
        return dict(interface=interface,addresses=current,activated=False)
    if current:
        raise RuntimeError(f'{interface} has unrelated IPv4 addresses {current}; configure its robot network explicitly')
    if not connection:
        raise RuntimeError(f'{interface} has no robot IPv4 address; enable its NetworkManager connection')
    def setting(name):
        return subprocess.run(['nmcli','-g',name,'connection','show',connection],
                              capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    if (setting('connection.interface-name')!=interface
            or setting('ipv4.method')!='manual'
            or setting('ipv4.never-default')!='yes'
            or setting('ipv4.gateway')):
        raise RuntimeError(f'{connection}: expected a manual, gateway-free profile bound to {interface}')
    profile_addresses=setting('ipv4.addresses').split(',')
    if not all(ipaddress.ip_interface(address.strip()).network==subnet for address in profile_addresses):
        raise RuntimeError(f'{connection}: expected addresses in {subnet}')
    print(f'Network: activating {connection} on {interface}',flush=True)
    result=subprocess.run(['nmcli','--wait','20','connection','up','id',connection,'ifname',interface],
                          capture_output=True,text=True,timeout=25)
    if result.returncode:
        raise RuntimeError(f'Network activation failed: {result.stderr.strip() or result.stdout.strip()}')
    current=addresses()
    if not any(ipaddress.ip_address(address) in subnet for address in current):
        raise RuntimeError(f'{interface} still has no address in {subnet}')
    return dict(interface=interface,addresses=current,activated=True,connection=connection)


def validate_args(args):
    if args.episodes < 1 or not args.task.strip():
        raise ValueError('Positive --episodes and nonempty --task required')
    for value,limit in ((args.active_seconds,180),(args.wait_seconds,600),(args.min_free_gb,100000)):
        if not math.isfinite(value) or not 0 < value <= limit:
            raise ValueError('Invalid duration or disk-space limit')
    if args.execute and args.preflight_only:
        raise ValueError('--execute and --preflight-only are mutually exclusive')
    if not args.execute and not args.preflight_only:
        raise ValueError('Choose --preflight-only (read only) or --execute')
    if args.execute and args.scene_confirmation != 'True':
        raise ValueError('--execute requires --scene-confirmation True')
    if args.execute and not sys.stdin.isatty():
        raise ValueError('Formal collection needs an interactive terminal for s/f')
    if args.output.exists():
        raise FileExistsError('Use a new output directory; existing runs are never overwritten')


def check_vr_address(host, port=8012):
    # Match the web server's address reuse: closed VR connections may linger
    # in FIN_WAIT/TIME_WAIT. An active listener must still block a new run.
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(1)


def local_preflight(args):
    from record_cameras import load_config
    load_config(args.camera_config)
    for path in (args.cert,args.key,args.camera_ssh_key,ROOT/'.venv-g1-data/bin/python'):
        if not path.is_file():
            raise FileNotFoundError(path)
    context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(args.cert),str(args.key))
    if args.interface not in {name for _,name in socket.if_nameindex()}:
        raise ValueError('Robot network interface does not exist')
    network=ensure_robot_network(args.interface,args.network_connection)
    # Bind checks that the headset address belongs to this host and port is free.
    check_vr_address(args.host_address)
    check_disk(args)
    return dict(tls_pair=True,interface=args.interface,network=network,host_address=args.host_address,
                vr_port=8012,camera_config=str(args.camera_config))


def check_disk(args):
    free=shutil.disk_usage(args.output).free
    if free < args.min_free_gb*1024**3:
        raise RuntimeError(f'Insufficient disk space: {free/1024**3:.1f} GiB free')


def cleanup_confirmed(summary):
    return (motion_cleanup_confirmed(summary)
            and not summary.get('input_route_cleanup_error')
            and summary.get('camera_recorder_returncode') == 0)


def motion_cleanup_confirmed(summary):
    """Camera failure is recoverable, but never substitutes for motor release."""
    if (summary.get('control_publish_not_started') is True
            and summary.get('command_frames') == 0
            and not summary.get('external_gripper_cleanup_error')
            and not summary.get('unload_error')):
        return True
    return (summary.get('arm_sdk_release_submitted') is True
            and summary.get('external_gripper_zero_submitted') == [True,True]
            and not summary.get('external_gripper_cleanup_error')
            and not summary.get('unload_error'))


def discard_attempt(run, attempt):
    """Delete only this run's direct pending attempt, after its writers have stopped."""
    run,attempt=Path(run).resolve(),Path(attempt)
    if attempt.is_symlink() or attempt.resolve().parent != run/'pending' or not attempt.name.startswith('attempt_'):
        raise ValueError('Refusing to discard outside the current pending directory')
    shutil.rmtree(attempt)


def save_attempt(run,attempt,index):
    destination=Path(run)/'episodes'/f'episode_{index:06d}'
    if destination.exists():
        raise FileExistsError(destination)
    attempt.rename(destination)
    write_json(destination/'operator_label.json',dict(decision='s',outcome='success',accepted_index=index))
    return destination


class Collector:
    def __init__(self,args):
        self.args=args
        self.stop=False
        self.child=None
        self.input_process=None
        self.input_cleanup_thread=None
        self.input_cleanup_error=None
        self.robot_checks=0
        self.camera_checks=0
        self.state=dict(schema='g1-continuous-collection-v2',target=args.episodes,accepted=[],
                        exported=[],export_results=[],attempts=[],phase='preflight')

    def checkpoint(self,phase):
        self.state.update(phase=phase,accepted_count=len(self.state['accepted']),
                          exported_count=len(self.state['exported']))
        write_json(self.args.output/'collection.json',self.state)

    def terminate(self,process,timeout=20):
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
                raise RuntimeError('Child cleanup timed out; collection stopped, no automatic rearm')

    def run_process(self,command,log_path,timeout=None):
        ready_path = (Path(command[command.index('--output')+1])/'collection_ready.json'
                      if '--shared-input' in command else None)
        ready_announced = False
        env=dict(os.environ,PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
                 HF_HOME=str(ROOT/'.local/unitree_g1/hf-data'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        env.pop('PYTHONPATH',None)
        with log_path.open('x') as log:
            self.child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
            deadline=time.monotonic()+timeout if timeout else math.inf
            try:
                while self.child.poll() is None:
                    if ready_path is not None and ready_path.exists() and not ready_announced:
                        print(f'{self.episode_label()} | Ready. Release both side grips, then hold both.',flush=True)
                        ready_announced=True
                    if self.stop:
                        self.terminate(self.child)
                        raise InterruptedError('Operator stopped collection')
                    if self.input_process is not None and self.input_process.poll() is not None:
                        self.terminate(self.child)
                        raise RuntimeError('Persistent VR service exited')
                    if time.monotonic()>deadline:
                        self.terminate(self.child)
                        raise TimeoutError(f'Process timeout; inspect {log_path}')
                    time.sleep(.1)
                return self.child.returncode
            finally:
                self.terminate(self.child)
                self.child=None

    def device_preflight(self,index):
        check_disk(self.args)
        path=self.args.output/'checks'/f'robot_{self.robot_checks:06d}.json'
        self.robot_checks+=1
        result=self.run_process([sys.executable,str(SCRIPTS/'external_dex1.py'),
            '--interface',self.args.interface,'--output',str(path)],path.with_suffix('.log'),timeout=20)
        if result or not read_json(path).get('preflight_passed'):
            detail=read_json(path).get('error','No diagnostic error recorded')
            raise RuntimeError(f'Robot/gripper/official-controller/network preflight failed: {detail}; report: {path}')

    def camera_preflight(self):
        a=self.args
        name='cameras' if self.camera_checks==0 else f'cameras_{self.camera_checks:06d}'
        self.camera_checks+=1
        path=a.output/'checks'/name
        log_path=a.output/'checks'/f'{name}.log'
        code=self.run_process([sys.executable,str(SCRIPTS/'record_cameras.py'),
            '--host',a.camera_host,'--ssh-key',str(a.camera_ssh_key),'--config',str(a.camera_config),
            '--duration','10','--output',str(path)],log_path,timeout=45)
        if (code or not (path/'summary.json').is_file()
                or read_json(path/'summary.json').get('error')
                or not (path/'frames.jsonl').is_file()):
            raise RuntimeError(f'Camera/SSH preflight failed; inspect {log_path}')
        from PIL import Image
        from record_cameras import CAMERAS, load_config
        specs=load_config(a.camera_config)['cameras']
        counts=dict.fromkeys(CAMERAS,0)
        for line in (path/'frames.jsonl').open():
            frame=json.loads(line)
            with Image.open(path/frame['file']) as image:
                image.load()
                if image.size != tuple(specs[frame['role']]['shape'][1::-1]):raise ValueError('Camera preflight resolution changed')
            counts[frame['role']]+=1
        if min(counts.values()) < 10:
            raise RuntimeError('Camera preflight did not receive enough frames from all three views')
        write_json(a.output/'checks'/f'{name}_validation.json',dict(decoded_frames=counts,passed=True))
        # Probe images are temporary diagnostics, never training episodes.
        shutil.rmtree(path)

    def start_input(self):
        a=self.args
        service=a.output/'vr'
        self.input_log=(a.output/'vr.log').open('x')
        self.input_process=subprocess.Popen([sys.executable,str(SCRIPTS/'televuer_input.py'),
            '--cert',str(a.cert),'--key',str(a.key),'--output',str(service),
            '--host-address',a.host_address,'--timeout','1.0','--routed','--gripper-status',str(service)],
            stdout=self.input_log,stderr=subprocess.STDOUT,start_new_session=True)
        print(f'VR (keep open): https://{a.host_address}:8012/?ws=wss://{a.host_address}:8012',flush=True)
        deadline=time.monotonic()+a.wait_seconds
        while time.monotonic()<deadline:
            if self.stop:raise InterruptedError('Stopped while waiting for VR')
            if self.input_process.poll() is not None:raise RuntimeError('VR server failed; inspect vr.log')
            try:
                if check_service(service)['receive_fresh']:
                    return service
            except (OSError,ValueError,KeyError,RuntimeError):pass
            time.sleep(.1)
        raise TimeoutError('No fresh headset and controller input before wait deadline')

    def prompt(self,message,allowed,*,check_vr=True):
        # Buffered input from an earlier phase cannot start or label an episode.
        if sys.stdin.isatty():
            import termios
            termios.tcflush(sys.stdin,termios.TCIFLUSH)
        print(message,flush=True)
        while not self.stop:
            if check_vr and self.input_process.poll() is not None:raise RuntimeError('VR service exited while waiting for operator input')
            if select.select([sys.stdin],[],[],.2)[0]:
                value=sys.stdin.readline()
                if not value:return 'q'
                value=value.strip().lower()
                if value in allowed:return value
                print('Use '+ '/'.join(allowed)+' + Enter.',flush=True)
        return 'q'

    def wait_for_repair(self,service,record=None):
        """Read-only recovery gates; passing them never starts an episode."""
        while not self.stop:
            self.checkpoint('awaiting_repair')
            choice=self.prompt('Collection paused. Repair the issue, then r: recheck | q: quit [Enter]',
                               ('r','q'),check_vr=False)
            if choice!='r' or self.stop:
                return False
            evidence=dict(time=time.time(),passed=False)
            self.state.setdefault('recovery_checks',[]).append(evidence)
            self.checkpoint('checking_repair')
            try:
                if self.child is not None and self.child.poll() is None:
                    raise RuntimeError('Attempt process is still running; no rearm')
                if record is not None:
                    summary=read_json(Path(record['path'])/'summary.json')
                    if not motion_cleanup_confirmed(summary):
                        raise RuntimeError('Arm/gripper release is not confirmed; sensor preflight cannot clear this block')
                    if type(summary.get('camera_recorder_returncode')) is not int:
                        raise RuntimeError('Camera recorder termination is not confirmed; no rearm')
                if self.input_process is None or self.input_process.poll() is not None:
                    raise RuntimeError('VR service is not running; no rearm')
                set_route(service)
                self.device_preflight(len(self.state['attempts'])+1)
                self.camera_preflight()
                if not check_service(service)['receive_fresh']:
                    raise RuntimeError('Headset/controller input is not fresh')
                if self.stop:
                    return False
                evidence['passed']=True
                if record is not None:
                    record['recovery_preflight_passed']=True
                self.checkpoint('repair_ready')
                print('Recheck passed. No motion started; enter n at the next prompt to prepare a new attempt.',flush=True)
                return True
            except Exception as exc:
                evidence['error']=f'{type(exc).__name__}: {exc}'
                print(f'Recheck failed: {evidence["error"]}',flush=True)
                self.checkpoint('awaiting_repair')
        return False

    def episode_label(self,*,next_attempt=False):
        attempt=len(self.state['attempts'])+int(next_attempt)
        episode=len(self.state['accepted'])+1
        return f'Try {attempt} | Episode {episode}/{self.args.episodes}'

    def wait_for_start(self):
        return self.prompt(f'{self.episode_label(next_attempt=True)} | Reset scene. n: prepare/open grippers | q: quit [Enter]',('n','q'))

    def decision(self):
        return self.prompt(f'{self.episode_label()} | Finished. s: save | f: discard | q: quit [Enter]',('s','f','q'))

    def episode_command(self,attempt,service):
        a=self.args
        return [sys.executable,str(SCRIPTS/'quest_bridge.py'),'--interface',a.interface,
            '--cert',str(a.cert),'--key',str(a.key),'--host-address',a.host_address,
            '--output',str(attempt),'--execute','--scene-confirmation',a.scene_confirmation,
            '--control-mode','arm_sdk','--grippers','--gripper-backend','external-dex1',
            '--record','--task',a.task,'--active-seconds',str(a.active_seconds),
            '--wait-seconds',str(a.wait_seconds),'--camera-host',a.camera_host,
            '--camera-ssh-key',str(a.camera_ssh_key),'--camera-config',str(a.camera_config),
            '--shared-input',str(service),'--defer-export']

    def run(self):
        try:
            return self.collect_and_export()
        finally:
            if self.input_cleanup_thread is not None:
                self.input_cleanup_thread.join()
                if self.input_cleanup_error:
                    self.state['input_cleanup_error']=self.input_cleanup_error
                    print(f'VR cleanup warning: {self.input_cleanup_error}; export results retained',
                          file=sys.stderr,flush=True)
                self.checkpoint(self.state['phase'])

    def cleanup_input(self):
        """Reap input-only VR in parallel with export; never write collection state here."""
        try:
            if self.input_process.poll() is None:
                self.input_process.terminate()
                try:
                    self.input_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.input_process.pid,signal.SIGKILL)
                    self.input_process.wait(timeout=5)
                    self.input_cleanup_error='VR exit timed out after 10s; process group killed'
        except Exception as exc:
            self.input_cleanup_error=f'{type(exc).__name__}: {exc}'
        finally:
            self.input_log.close()

    def collect_and_export(self):
        # Recording/motion cleanup finishes first. Input-only VR exits in parallel.
        result=self.collect()
        if not self.state['accepted']:
            return result
        collection_phase=self.state['phase']
        self.state['collection_end_phase']=collection_phase
        self.checkpoint('exporting_batch')
        print(f'Collection ended. Exporting {len(self.state["accepted"])} accepted episodes...',flush=True)
        failed=False
        for index,path in enumerate(self.state['accepted']):
            print(f'Export {index+1}/{len(self.state["accepted"])}: {path} (see export.log)',flush=True)
            try:
                status=export_recording(Path(path),outcome='success')
            except Exception as exc:
                status=dict(exported=False,error=f'{type(exc).__name__}: {exc}')
            self.state['export_results'].append(dict(session=path,**status))
            if status.get('exported'):
                self.state['exported'].append(str(Path(path)/'lerobot'))
                print(f'Export {index+1}: complete',flush=True)
            else:
                failed=True
                print(f'Export {index+1}: failed; raw data retained at {path}',flush=True)
            self.checkpoint('exporting_batch')
        self.checkpoint('export_failed' if failed else collection_phase)
        return 1 if failed else result

    def collect(self):
        a=self.args
        a.output.mkdir(parents=True,exist_ok=False)
        for name in ('checks','pending','episodes','logs'):(a.output/name).mkdir()
        self.checkpoint('preflight')
        print(f'Output: {a.output}',flush=True)
        service=None
        try:
            local=local_preflight(a)
            # Check writer/encoder dependencies before any robot motion.
            env_code="import importlib.metadata, av; from lerobot.datasets.lerobot_dataset import LeRobotDataset; assert importlib.metadata.version('lerobot') == '0.4.4'; av.codec.Codec('libsvtav1','w')"
            if self.run_process([str(ROOT/'.venv-g1-data/bin/python'),'-c',env_code],a.output/'checks/data_environment.log',timeout=40):
                raise RuntimeError('LeRobot/AV1 data environment unavailable')
            if a.execute:
                print('Checking robot Dex1 service startup',flush=True)
                service_status=ensure_service(a.camera_host,a.camera_ssh_key)
                write_json(a.output/'checks/dex1_service.json',service_status)
            self.device_preflight(0)
            self.camera_preflight()
            write_json(a.output/'checks/preflight.json',dict(local=local,passed=True,read_only=True))
            if a.preflight_only:
                self.checkpoint('preflight_passed');return 0
            service=self.start_input()
            while len(self.state['accepted']) < a.episodes and not self.stop:
                number=len(self.state['attempts'])+1
                self.checkpoint('awaiting_n')
                if self.wait_for_start() != 'n' or self.stop:
                    break
                self.checkpoint('checking_next_episode')
                precheck_started = time.monotonic()
                try:
                    self.device_preflight(number)
                    if not check_service(service)['receive_fresh']:
                        raise RuntimeError('Headset/controller input is not fresh; next episode was not started')
                except Exception as exc:
                    self.state['last_preflight_error']=f'{type(exc).__name__}: {exc}'
                    print(self.state['last_preflight_error'],flush=True)
                    if not self.wait_for_repair(service):break
                    continue
                precheck_s = time.monotonic()-precheck_started
                attempt=a.output/'pending'/f'attempt_{number:06d}'
                record=dict(attempt=number,path=str(attempt),start_confirmation='n',decision=None,
                            precheck_s=precheck_s)
                self.state['attempts'].append(record)
                self.checkpoint('recording')
                print(f'{self.episode_label()} | Preparing',flush=True)
                code=None
                summary={}
                failures=[]
                try:
                    # The child attempts bounded unloading in its finally block
                    # before run_process returns (including graceful termination).
                    code=self.run_process(self.episode_command(attempt,service),a.output/'logs'/f'attempt_{number:06d}.log')
                except Exception as exc:
                    failures.append(f'{type(exc).__name__}: {exc}')
                try:
                    set_route(service)  # acknowledgement closes the input route.
                except Exception as exc:
                    failures.append(f'Input route cleanup: {type(exc).__name__}: {exc}')
                try:
                    summary=read_json(attempt/'summary.json')
                except (OSError,ValueError) as exc:
                    failures.append(f'Attempt summary unavailable: {exc}')
                record['returncode']=code
                if failures or code or summary.get('error') or not cleanup_confirmed(summary):
                    record['status']='hardware_or_cleanup_fault'
                    record['error']=summary.get('error')
                    record['collector_errors']=failures
                    record['motion_cleanup_confirmed']=motion_cleanup_confirmed(summary)
                    attempt.mkdir(parents=True,exist_ok=True)
                    write_json(attempt/'collection_fault.json',dict(record,summary=summary))
                    print(f'Attempt failed; data retained: {attempt}\n'
                          f'Motion cleanup confirmed: {record["motion_cleanup_confirmed"]}. '
                          f'Error: {summary.get("error") or failures or code}',flush=True)
                    if not self.wait_for_repair(service,record):break
                    continue
                self.checkpoint('awaiting_s_or_f')
                choice=self.decision()
                record['decision']=choice
                if choice=='q':
                    record['status']='unlabelled_retained';break
                if choice=='f':
                    discard_attempt(a.output,attempt)
                    record['status']='discarded'
                    record.pop('path',None)
                    self.checkpoint('retrying')
                    continue
                self.checkpoint('saving_raw')
                try:
                    destination=save_attempt(a.output,attempt,len(self.state['accepted']))
                except Exception:
                    destination=a.output/'episodes'/f'episode_{len(self.state["accepted"]):06d}'
                    record['status']='save_failed'
                    if destination.exists():record['path']=str(destination)
                    raise
                record.update(status='saved',path=str(destination))
                self.state['accepted'].append(str(destination))
                self.checkpoint('saved')
                print(f'Try {number} | Episode {len(self.state["accepted"])}/{a.episodes} | Saved raw data; export queued until collection ends',flush=True)
            self.checkpoint('complete' if len(self.state['accepted'])==a.episodes else 'stopped')
            return 0
        except BaseException as exc:
            self.state['error']=f'{type(exc).__name__}: {exc}'
            self.checkpoint('stopped' if self.stop else 'error')
            print(self.state['error'],file=sys.stderr,flush=True)
            return 1
        finally:
            try:
                self.terminate(self.child)
            finally:
                if self.input_process:
                    self.input_cleanup_thread=threading.Thread(target=self.cleanup_input,
                                                              name='vr-cleanup')
                    self.input_cleanup_thread.start()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes',type=int,required=True,help='Number of accepted episodes, not attempts')
    parser.add_argument('--task',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--interface',required=True)
    parser.add_argument('--network-connection',default='g1-direct-thinkcentre',
                        help='Existing dedicated NetworkManager profile to activate if IPv4 is absent; empty disables activation')
    parser.add_argument('--host-address',required=True)
    parser.add_argument('--cert',type=Path,required=True)
    parser.add_argument('--key',type=Path,required=True)
    parser.add_argument('--camera-host',default='unitree@192.168.123.164')
    parser.add_argument('--camera-ssh-key',type=Path,default=ROOT/'.local/unitree_g1/ssh/id_ed25519')
    parser.add_argument('--camera-config',type=Path,default=ROOT/'artifacts/unitree_g1/camera_capture_profile.json')
    parser.add_argument('--active-seconds',type=float,default=180.)
    parser.add_argument('--wait-seconds',type=float,default=300.)
    parser.add_argument('--min-free-gb',type=float,default=10.)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--scene-confirmation',choices=('True','False'),
                        help='True confirms the scene is ready for this collection run')
    args=parser.parse_args()
    for name in ('output','cert','key','camera_ssh_key','camera_config'):
        setattr(args,name,getattr(args,name).resolve())
    try:validate_args(args)
    except (ValueError,OSError) as exc:parser.error(str(exc))
    collector=Collector(args)
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,lambda *_:setattr(collector,'stop',True))
    return collector.run()


if __name__=='__main__':
    raise SystemExit(main())
