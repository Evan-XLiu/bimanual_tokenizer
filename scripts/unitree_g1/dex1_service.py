#!/usr/bin/env python3
"""Install the fixed Dex1 startup repair, or invoke it before collection."""
import argparse
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def ssh_command(host,key):
    if host.startswith('-'):
        raise ValueError('Invalid SSH host')
    return ['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
            '-o','ConnectTimeout=5','-o','IdentitiesOnly=yes','-i',str(key),host]


def ensure_service(host,key):
    result=subprocess.run(ssh_command(host,key)+['sudo -n /usr/local/sbin/g1-ensure-dex1'],
                          capture_output=True,text=True,timeout=40)
    if result.returncode:
        raise RuntimeError('Dex1 startup repair failed. Run scripts/unitree_g1/dex1_service.py '
                           '--install once with robot sudo authentication. Details: '+
                           (result.stderr.strip() or result.stdout.strip()))
    return dict(host=host,service='dex1_gripper.service',details=result.stdout.strip())


def install(host,key):
    ssh=ssh_command(host,key)
    remote=subprocess.run(ssh+['mktemp /tmp/g1-dex1-install.XXXXXXXX'],
                          capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    if not remote.startswith('/tmp/g1-dex1-install.') or '\n' in remote:
        raise RuntimeError('Unexpected remote installation path')
    quoted=shlex.quote(remote)
    try:
        source=Path(__file__).with_name('install_dex1_service.sh').read_text()
        subprocess.run(ssh+['cat > '+quoted],input=source,text=True,check=True,timeout=10)
        # Keep stdin attached to the operator terminal for the robot's sudo prompt.
        subprocess.run(ssh[:1]+['-t']+ssh[1:]+['sudo /bin/sh '+quoted],check=True)
    finally:
        subprocess.run(ssh+['rm -f -- '+quoted],timeout=10,check=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='unitree@192.168.123.164')
    parser.add_argument('--ssh-key',type=Path,default=ROOT/'.local/unitree_g1/ssh/id_ed25519')
    parser.add_argument('--install',action='store_true')
    args=parser.parse_args()
    if args.install:
        install(args.host,args.ssh_key)
    else:
        print(ensure_service(args.host,args.ssh_key))
