# Evaluate

Praxis evaluation is a wrapper around `praxis-eval`. Praxis loads the policy;
`praxis-eval` owns the benchmark rollout.

## Local Evaluation

```bash
POLICY_CKPT=/absolute/path/to/policy/pretrained_model

HYDRA_FULL_ERROR=1 MUJOCO_GL=egl uv run praxis-eval \
  policy=bar_oat_pali \
  policy.pretrained_path="${POLICY_CKPT}" \
  env.type=libero env.task=libero \
  eval.mode=local_inproc \
  eval.num_eval_per_task=2 eval.num_parallel_env=2 \
  eval.record_episodes_per_task=1 \
  eval.output_dir=outputs/eval/libero_bar_oat_pali_smoke
```

Use local mode when simulator and policy dependencies can share one
environment.

## Remote Evaluation

Remote mode connects to a running policy server:

```bash
HYDRA_FULL_ERROR=1 MUJOCO_GL=egl uv run praxis-eval \
  env.type=libero env.task=libero \
  eval.mode=remote_grpc \
  server_address=127.0.0.1:50051 \
  eval.num_eval_per_task=2 eval.num_parallel_env=2 \
  eval.output_dir=outputs/eval/libero_remote_smoke
```

Use remote mode when policy and simulator dependencies should be isolated.

## Output

Evaluation output is benchmark-owned and typically includes:

```text
results.json
media/
```

Additional traces, logs, videos, or benchmark-native artifacts may appear
depending on the driver.
