# BAR OAT Pali Example

This guide uses one Praxis policy configuration:

```text
bar_oat_pali
```

It combines:

| Axis | Choice |
| --- | --- |
| Policy objective | BAR |
| Action codec | OAT |
| VLM backbone | PaliGemma2 |

This cell is useful as a worked example because it exercises the main contracts
in the repository: an action-token codec, a token-generating VLM policy, a
backbone adapter, a trained policy checkpoint, and the `praxis-eval` adapter.
It is an example path, not a project recommendation.

Run training and evaluation commands on a compute node with a GPU. On a login
node, use `--cfg job` to inspect Hydra composition.

## Train OAT

```bash
uv run praxis-train --config-name train_codecs \
  codecs@codec=tokens/oat \
  dataset.name=libero dataset.repo_id=lerobot/libero \
  device=cuda \
  trainer.max_steps=500 trainer.val_every=200 trainer.checkpoint_every=200 \
  trainer.use_wandb=false
```

The codec artifact is written under:

```text
outputs/train_codecs/<date>/<run_name>/checkpoints/latest/pretrained_model
```

## Train BAR OAT Pali

```bash
CODEC_CKPT="$(find outputs/train_codecs -path '*/checkpoints/latest/pretrained_model' -type d | sort | tail -n 1)"
test -n "${CODEC_CKPT}"

uv run praxis-train \
  policy=bar_oat_pali \
  policy.action_tokenizer.pretrained_path="${CODEC_CKPT}" \
  dataset.name=libero dataset.repo_id=lerobot/libero \
  env.type=libero env.task=libero \
  device=cuda \
  trainer.max_steps=120 \
  trainer.val_every=100 trainer.reconst_every=100 trainer.checkpoint_every=100 \
  trainer.sim_eval_every=100 trainer.sim_eval.num_eval_per_task=2 \
  trainer.sim_eval.num_parallel_env=2 \
  trainer.use_wandb=false trainer.wandb_upload_eval_videos=false
```

The policy artifact is written under:

```text
outputs/train_policy/<date>/libero_bar_oat_pali/checkpoints/latest/pretrained_model
```

## Evaluate

```bash
POLICY_CKPT="$(find outputs/train_policy -path '*/checkpoints/latest/pretrained_model' -type d | sort | tail -n 1)"
test -n "${POLICY_CKPT}"

uv run praxis-eval \
  policy=bar_oat_pali \
  policy.pretrained_path="${POLICY_CKPT}" \
  env.type=libero env.task=libero \
  eval.mode=local_inproc \
  eval.num_eval_per_task=2 eval.num_parallel_env=2 \
  eval.record_episodes_per_task=1 \
  eval.output_dir=outputs/eval/bar_oat_pali
```

## Follow-Up Comparisons

Change one axis at a time:

```text
bar_oat_pali -> bar_oat_qwen      # backbone
bar_oat_pali -> bar_fast_pali     # codec
bar_oat_pali -> ki_oat_pali       # policy objective
bar_oat_pali -> fm_pali           # objective without token codec
```

For each comparison, report at least:

- codec reconstruction loss, when a codec is present;
- token CE for BAR or KI;
- decoded action error;
- closed-loop success from `praxis-eval`;
- action bounds or clipping rate when reported by the benchmark;
- wall-clock and accelerator memory.
