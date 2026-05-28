# Evaluate A Policy

Praxis evaluation is a wrapper around `praxis-eval`.

For LIBERO, Praxis accepts `env.task=libero` as the user-facing full-suite
selector before handing the expanded evaluator task list to `praxis-eval`.

Simulator evaluation is a compute-node job. Use `--cfg job` on a login node to
inspect composition without loading a policy or starting environments:

```bash
uv run praxis-eval \
  policy=bar_oat_pali \
  policy.pretrained_path=/tmp/policy \
  env.type=libero env.task=libero \
  eval.mode=local_inproc \
  --cfg job
```

## Local Checkpoint Evaluation

```bash
POLICY_CKPT="$(find outputs/train_policy -path '*/checkpoints/latest/pretrained_model' -type d | sort | tail -n 1)"
test -n "${POLICY_CKPT}"

uv run praxis-eval \
  policy=bar_oat_pali \
  policy.pretrained_path="${POLICY_CKPT}" \
  env.type=libero env.task=libero \
  eval.mode=local_inproc \
  eval.num_eval_per_task=20 \
  eval.num_parallel_env=10 \
  eval.output_dir=outputs/eval/libero_bar_oat_pali
```

Run the real command inside an interactive or batch compute allocation.

The policy checkpoint must be self-contained with:

```text
policy_io.json
norm_stats.pt
```

## Remote Evaluation

Remote evaluation connects to a running `praxis-serve` endpoint:

```bash
uv run praxis-eval \
  env.type=libero env.task=libero \
  eval.mode=remote_grpc \
  server_address=127.0.0.1:50051 \
  eval.output_dir=outputs/eval/libero_remote
```

## Results

`praxis-eval` writes benchmark-owned results:

```text
results.json
media/
```

Praxis should not parse simulator internals directly. It should consume the evaluator result and log metrics.

## Eval Knobs

Common knobs:

```text
eval.num_eval_per_task
eval.num_parallel_env
eval.record_episodes_per_task
eval.step_timeout_sec
eval.rollout_failure_retries
eval.policy_kwargs.decode_keep_k
eval.policy_kwargs.decode_temperature
eval.policy_kwargs.decode_top_k
```

Policy kwargs are passed through the adapter to the policy.
