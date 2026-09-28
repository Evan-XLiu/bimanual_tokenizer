import json, glob
from pathlib import Path
import numpy as np
import h5py

root=Path('data/robotwin_new_demo_clean/demo_clean'); out=Path('artifacts/robotwin_new_data_audit'); out.mkdir(parents=True,exist_ok=True)
names=['left_joint_'+str(i) for i in range(6)]+['left_gripper']+['right_joint_'+str(i) for i in range(6)]+['right_gripper']
allx=[]; task_stats={}; periodic=[]; jumps=[]; bad=[]
for td in sorted(p for p in root.iterdir() if p.is_dir()):
 eps=[]
 for f in sorted((td/'aloha_agilex'/'data').glob('episode_*.hdf5')):
  with h5py.File(f,'r') as h: x=np.asarray(h['action/joint_states'],dtype=np.float64)
  if x.ndim!=2 or x.shape[1]!=14: bad.append({'file':str(f),'shape':list(x.shape)}); continue
  allx.append(x); eps.append((f,x)); d=np.abs(np.diff(x,axis=0))
  for j in range(14):
   ix=np.where(np.abs(d[:,j]-2*np.pi)<0.02)[0]
   if len(ix): jumps.append({'task':td.name,'episode':f.stem,'joint':names[j],'count':int(len(ix)),'max_delta':float(d[:,j].max())})
 for i,(fa,a) in enumerate(eps):
  for fb,b in eps[i+1:]:
   m=min(len(a),len(b)); diff=np.median(a[:m]-b[:m],axis=0)
   for j,v in enumerate(diff):
    k=round(v/(2*np.pi))
    if k in (-1,1) and abs(v-k*2*np.pi)<0.02: periodic.append({'task':td.name,'episode_a':fa.stem,'episode_b':fb.stem,'joint':names[j],'offset':float(v),'residual':float(v-k*2*np.pi),'frames_compared':m})
 if eps: task_stats[td.name]={'episodes':len(eps),'frames':int(sum(len(x) for _,x in eps))}
x=np.concatenate(allx); ps=[.001,.01,.05,.1,.25,.5,.75,.9,.95,.99,.999]; q=np.quantile(x,ps,axis=0); stats=[]
for j,n in enumerate(names): stats.append(dict(joint=n,count=int(len(x)),min=float(x[:,j].min()),max=float(x[:,j].max()),mean=float(x[:,j].mean()),std=float(x[:,j].std()),**{f'q{int(p*1000):03d}':float(q[k,j]) for k,p in enumerate(ps)}))
report={'dataset_root':str(root.resolve()),'tasks':len(task_stats),'episodes':sum(v['episodes'] for v in task_stats.values()),'frames':int(len(x)),'joint_names':names,'joint_stats':stats,'periodic_2pi_pair_flags':periodic,'within_episode_2pi_jump_flags':jumps,'invalid_files':bad,'task_stats':task_stats,'thresholds':{'offset_abs_tol_rad':0.02,'jump_abs_tol_rad':0.02}}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n'); keys=['joint','count','min','max','mean','std']+[f'q{int(p*1000):03d}' for p in ps]
with (out/'joint_stats.csv').open('w') as f:
 f.write(','.join(keys)+'\n')
 for s in stats: f.write(','.join(str(s[k]) for k in keys)+'\n')
print(json.dumps({'tasks':len(task_stats),'episodes':report['episodes'],'frames':report['frames'],'periodic_pair_flags':len(periodic),'within_episode_jump_flags':len(jumps),'invalid_files':len(bad),'report':str((out/'report.json').resolve()),'csv':str((out/'joint_stats.csv').resolve())},indent=2))
