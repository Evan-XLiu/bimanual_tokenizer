import json
from pathlib import Path
import numpy as np, h5py
root=Path('data/robotwin_new_demo_clean/demo_clean'); out=Path('artifacts/robotwin_new_data_audit'); names=[f'left_joint_{i}' for i in range(6)]+['left_gripper']+[f'right_joint_{i}' for i in range(6)]+['right_gripper']
vals=[]; records=[]
for td in sorted(p for p in root.iterdir() if p.is_dir()):
 for f in sorted((td/'aloha_agilex'/'data').glob('episode_*.hdf5')):
  with h5py.File(f,'r') as h: x=np.asarray(h['action/joint_states'],float)
  d=np.abs(np.diff(x,axis=0)); vals.append(d)
  for j in range(14):
   for i in np.where(d[:,j]>0.5)[0]: records.append({'task':td.name,'episode':f.stem,'frame_from':int(i),'frame_to':int(i+1),'joint':names[j],'delta':float(d[i,j]),'from_value':float(x[i,j]),'to_value':float(x[i+1,j])})
d=np.concatenate(vals); qs=[.9,.95,.99,.999,.9999]; summary=[]
for j,n in enumerate(names): summary.append({'joint':n,'max':float(d[:,j].max()),'mean':float(d[:,j].mean()),**{f'q{int(q*10000):05d}':float(np.quantile(d[:,j],q)) for q in qs},**{f'count_gt_{t}':int((d[:,j]>t).sum()) for t in [.1,.25,.5,1.,1.5]}})
report={'frames_differences':int(len(d)),'joint_summary':summary,'events_gt_0.5rad':sorted(records,key=lambda r:r['delta'],reverse=True)}
(out/'jump_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'events_gt_0.5':len(records),'events_gt_1.0':sum(r['delta']>1 for r in records),'events_gt_1.5':sum(r['delta']>1.5 for r in records),'top10':records[:10],'report':str((out/'jump_report.json').resolve())},indent=2))
for s in summary: print(s)
