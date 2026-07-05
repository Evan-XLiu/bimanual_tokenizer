# CLI

Praxis installs four command entry points.

## `praxis-train`

Train codecs or policies.

```bash
uv run praxis-train --config-name train_codecs --help
uv run praxis-train --config-name train_policy --help
```

Common forms:

```bash
uv run praxis-train --config-name train_codecs codecs@codec=tokens/oat
uv run praxis-train policy=bar_oat_pali policy.action_tokenizer.pretrained_path=/path/to/codec
uv run praxis-train policy=fm_pali
```

Use `--cfg job` to inspect composed Hydra config:

```bash
uv run praxis-train policy=bar_oat_pali --cfg job
```

## `praxis-eval`

Evaluate a local checkpoint or remote policy server through `praxis-eval`.

```bash
uv run praxis-eval \
  policy=bar_oat_pali \
  policy.pretrained_path=/path/to/policy \
  env.type=libero env.task=libero \
  eval.mode=local_inproc
```

```bash
uv run praxis-eval \
  env.type=libero env.task=libero \
  eval.mode=remote_grpc \
  server_address=127.0.0.1:50051
```

## `praxis-serve`

Serve a policy checkpoint for remote evaluation.

```bash
uv run praxis-serve \
  policy=bar_oat_pali \
  policy.pretrained_path=/path/to/policy \
  host=0.0.0.0 \
  port=50051
```

## `praxis-eval-monitor`

Monitor evaluation candidates and launch evaluator jobs. This is an operational
helper; use the workflow docs for the core local/remote evaluation path.
