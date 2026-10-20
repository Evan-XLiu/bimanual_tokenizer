import glob, json, torch, numpy as np, pyarrow.parquet as pq
from pathlib import Path
from praxis.codecs.factory import make_codec
root=Path('/scratch/xuanweiliu/bimanual_tokenizer/data/BiCAT-BimanualYAM-Dataset')
ck='/scratch/xuanweiliu/bimanual_tokenizer/realworld_BiCAT/AT/BTT_tokenizer_step_150000'
codec=make_codec({'kind':'tokens','name':'btt','pretrained_path':ck})
codec.eval(); device='cpu'; codec.to(device)
print('codebook_size',codec.codebook_size,'seq_len',codec.max_code_seq_len, flush=True)
counts=np.zeros(codec.codebook_size,dtype=np.int64); n=0; chunks=0
with torch.inference_mode():
 for src in ['box','bottles','pen','plate']:
  files=glob.glob(str(root/src/'data'/'**'/'*.parquet'),recursive=True)
  for fi,f in enumerate(files):
   tab=pq.read_table(f,columns=['action_joint_angles','episode_index'])
   acts=np.asarray(tab['action_joint_angles'].to_pylist(),dtype=np.float32); eps=np.asarray(tab['episode_index'])
   for ep in np.unique(eps):
    a=acts[eps==ep]; m=(len(a)//32)*32
    for i in range(0,m,32):
     x=torch.from_numpy(a[i:i+32]).unsqueeze(0)
     tok=codec.tokenize(x).numpy().reshape(-1)
     counts += np.bincount(tok,minlength=codec.codebook_size); chunks+=1
   if fi%10==0: print(src,fi,'chunks',chunks,flush=True)
used=int((counts>0).sum()); total=int(counts.sum()); p=counts[counts>0]/total
print(json.dumps({'chunks':chunks,'tokens':total,'used_codes':used,'codebook_size':codec.codebook_size,'utilization':used/codec.codebook_size,'perplexity':float(np.exp(-(p*np.log(p)).sum())),'top20':sorted([(int(i),int(c)) for i,c in enumerate(counts)],key=lambda x:-x[1])[:20]},indent=2))
np.save('/tmp/btt_counts.npy',counts)
