# Artifacts

Praxis artifacts are directories that combine model weights with sidecars.

## Codec Artifact

Codec training writes:

```text
outputs/train_codecs/<date>/<run_name>/checkpoints/latest/pretrained_model
```

Expected sidecars:

```text
policy_io.json
norm_stats.pt
```

BAR and KI policies reference this path through:

```text
policy.action_tokenizer.pretrained_path=/path/to/pretrained_model
```

## Policy Artifact

Policy training writes:

```text
outputs/train_policy/<date>/<run_name>/checkpoints/latest/pretrained_model
```

Expected sidecars:

```text
policy_io.json
norm_stats.pt
```

The policy artifact is the path used by evaluation and serving:

```text
policy.pretrained_path=/path/to/pretrained_model
```

## Evaluation Artifact

Evaluation output is owned by `praxis-eval`:

```text
outputs/eval/<run_name>/
  results.json
  media/
```

Benchmark-specific drivers may add logs, traces, videos, or native output
files.
