# Serve

Serving exposes a Praxis checkpoint over the remote policy protocol used by
`praxis-eval`.

## Start A Server

```bash
POLICY_CKPT=/absolute/path/to/policy/pretrained_model

uv run praxis-serve \
  policy=bar_oat_pali \
  policy.pretrained_path="${POLICY_CKPT}" \
  host=0.0.0.0 \
  port=50051
```

The server owns policy loading and inference. The evaluator owns rollout.

## Evaluate Against The Server

```bash
HYDRA_FULL_ERROR=1 MUJOCO_GL=egl uv run praxis-eval \
  env.type=libero env.task=libero \
  eval.mode=remote_grpc \
  server_address=127.0.0.1:50051 \
  eval.num_eval_per_task=2 eval.num_parallel_env=2 \
  eval.output_dir=outputs/eval/libero_remote_smoke
```

## When To Use Serving

Use serving when:

- the simulator and policy require incompatible dependencies;
- the policy should run on a different machine or accelerator;
- you want a stable reset/act transport for evaluator-side experiments.

Do not expose the server to untrusted networks without additional deployment
controls. It is a research runtime, not a hardened public service.
