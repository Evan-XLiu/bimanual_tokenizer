# Evaluate the 3000-step FM box checkpoint from a workstation

The checkpoint remains at
`outputs/BiCAT_box_Molmo2ER_FM_30k/67256/train/checkpoints/recent/step_3000/pretrained_model`
on the shared server filesystem. The GPU node loads it
and serves one `/act` HTTP endpoint. The request uses the same `json_numpy`
array format as the Jianman server, but its state and actions are 14-D **joint**
values. Do not pass the 16-D EEF state expected by the Jianman box baseline.

The policy has been loaded and run successfully on one RTX A5000 on crane0.
Submit the one-GPU server there with:

```bash
cd /data/home/shailesh/Evan/bimanual_tokenizer
sbatch --nodelist=crane0 artifacts/bicat_box/serve_fm_box_step3000.sbatch
```

The server passed a local inference test on crane0. At the time of that test,
storage could not reach `crane0:8213` directly, so the following workstation
tunnel requires the cluster network path to be fixed or replaced by a working
forwarding route. It has not been validated end to end. Keep this command
running after connectivity is established:

```bash
ssh -N -L 8213:crane0:8213 shailesh@storage
```

On the workstation, `http://127.0.0.1:8213/act` reaches the policy. A quick
health check shows the exact checkpoint and action dimensions:

```bash
curl http://127.0.0.1:8213/act
```

The workstation sends one observation as `json_numpy.dumps(payload)`:

```python
import json_numpy
import numpy as np
import requests

payload = {
    "top": top_rgb_uint8,      # H x W x 3
    "left": left_rgb_uint8,    # H x W x 3
    "right": right_rgb_uint8,  # H x W x 3
    "state": np.asarray(joint_angles_14, dtype=np.float32),
    "instruction": "Put all blocks into the box.",
    "num_steps": 10,
}
response = requests.post(
    "http://127.0.0.1:8213/act",
    data=json_numpy.dumps(payload),
    headers={"Content-Type": "application/json"},
    timeout=180,
)
response.raise_for_status()
result = json_numpy.loads(response.text)
predicted = result["actions"]       # (32, 14), continuous joint actions
execution_prefix = predicted[:16]   # the configured receding-horizon prefix
```

The server also accepts camera keys `top_cam`, `left_cam`, and `right_cam`.
`GET /act` reports the checkpoint and configuration; `POST /act` returns the
full 32-step prediction. Each request predicts a fresh chunk. There is no
model upload and no action tokenizer. Check the controller's joint ordering,
units, and gripper convention against the training dataset before any physical
execution.

Port 8213 is separate from the Jianman baseline on port 8212. The existing
public ngrok domain can point to only one upstream at a time; this setup does
not change it. Keep access through SSH unless a public endpoint is needed.
