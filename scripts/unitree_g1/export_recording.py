"""Run the offline BiCAT/G1 export only after recording and robot cleanup finish."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def export_recording(session, *, outcome='unknown', allow_shadow=False):
    session = Path(session)
    destination = session/'lerobot'
    command = [str(ROOT/'.venv-g1-data/bin/python'),
               str(ROOT/'scripts/unitree_g1/convert_dataset.py'),
               '--session',str(session),'--root',str(destination),
               '--repo-id','local/'+session.name,'--outcome',outcome]
    if allow_shadow:
        command.append('--allow-shadow')
    env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               HF_HOME=str(ROOT/'.local/unitree_g1/hf-data'),
               PYTHONNOUSERSITE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    env.pop('PYTHONPATH',None)
    try:
        with (session/'export.log').open('x') as log:
            result = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        status = dict(exported=result.returncode == 0, returncode=result.returncode,
                      root=str(destination), outcome=outcome, log=str(session/'export.log'))
    except Exception as exc:
        status = dict(exported=False,error=f'{type(exc).__name__}: {exc}',root=str(destination))
    (session/'export_status.json').write_text(json.dumps(status,indent=2)+'\n')
    return status
