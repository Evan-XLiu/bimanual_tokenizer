# Training Artifacts

Praxis writes self-contained artifacts for codecs and policies.

## Codec Runs

Default root:

```text
outputs/train_codecs/<date>/<run_name>/
```

Important files:

```text
train_config.yaml
train_metrics.jsonl
checkpoints/latest/pretrained_model/
checkpoints/best/<step>/pretrained_model/
```

The codec `pretrained_model` contains:

```text
config.json
policy_io.json
norm_stats.pt
weights or tokenizer assets
```

## Policy Runs

Default root:

```text
outputs/train_policy/<date>/<run_name>/
```

Important files:

```text
train_config.yaml
train_metrics.jsonl
policy_io.json
checkpoints/latest/pretrained_model/
checkpoints/best/
eval/by_step/
```

The policy `pretrained_model` contains enough information for `praxis-eval` local mode to load the policy through `make_policy(...)`.

## Evaluation Runs

Evaluation output is owned by `praxis-eval`:

```text
results.json
media/
.hydra/
```

Praxis passes the policy adapter and config into `praxis_eval.evaluate(...)`; the evaluator owns rollout artifacts.
