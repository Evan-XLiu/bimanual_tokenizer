# CLI Reference

Praxis exposes four installable commands.

## `praxis-train`

Train a policy or codec.

Inspect the policy-training template:

```bash
uv run praxis-train --cfg job
```

This prints the composed job without launching training. Remove `--cfg job` only
when launching real training on a compute node.

Codec training:

```bash
uv run praxis-train --config-name train_codecs codecs@codec=tokens/oat --cfg job
```

The entry point is:

```text
src/praxis/scripts/train.py
```

## `praxis-eval`

Evaluate a local or remote policy through `praxis-eval`.

Local:

```bash
POLICY_CKPT="$(find outputs/train_policy -path '*/checkpoints/latest/pretrained_model' -type d | sort | tail -n 1)"
test -n "${POLICY_CKPT}"

uv run praxis-eval \
  policy=bar_oat_pali \
  policy.pretrained_path="${POLICY_CKPT}" \
  env.type=libero env.task=libero \
  eval.output_dir=outputs/eval/bar_oat_pali_smoke
```

Remote:

```bash
uv run praxis-eval \
  eval.mode=remote_grpc \
  server_address=127.0.0.1:50051 \
  env.type=libero env.task=libero \
  eval.output_dir=outputs/eval/remote
```

The entry point is:

```text
src/praxis/scripts/eval.py
```

## `praxis-serve`

Serve a checkpoint over gRPC:

```bash
POLICY_CKPT="$(find outputs/train_policy -path '*/checkpoints/latest/pretrained_model' -type d | sort | tail -n 1)"
test -n "${POLICY_CKPT}"

uv run praxis-serve \
  policy=bar_oat_pali \
  policy.pretrained_path="${POLICY_CKPT}" \
  host=0.0.0.0 \
  port=50051
```

The entry point is:

```text
src/praxis/scripts/serve.py
```

## `praxis-eval-monitor`

Monitor checkpoints and launch evaluations. This is training infrastructure, not the main conceptual path for understanding methods, backbones, or codecs.

The entry point is:

```text
src/praxis/scripts/eval_monitor.py
```
