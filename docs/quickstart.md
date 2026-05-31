# Quickstart

This quickstart runs one compact policy cell:

```text
policy=bar_oat_pali
```

That expands to:

| Coordinate | Choice |
| --- | --- |
| objective | BAR |
| action representation | OAT |
| VLM backbone | PaliGemma2 |

Use this path to verify the repository surfaces. It is not a claim that this
cell is the best choice for your study.

## Install

From the repository root:

```bash
uv sync --group dev --extra libero --extra pali --extra serving --extra logging
```

Verify the package imports:

```bash
uv run python -c "import praxis; print('praxis import OK')"
```

## Train The OAT Codec

BAR and KI policies require a pretrained token codec. The codec checkpoint
carries parameters plus action-normalization sidecars.

```bash
HYDRA_FULL_ERROR=1 uv run praxis-train --config-name train_codecs \
  codecs@codec=tokens/oat \
  dataset.name=libero dataset.repo_id=lerobot/libero \
  device=cuda \
  trainer.max_steps=500 trainer.val_every=200 trainer.checkpoint_every=200 \
  trainer.use_wandb=false
```

The codec path has this shape:

```text
outputs/train_codecs/<date>/<run_name>/checkpoints/latest/pretrained_model
```

## Train The Policy

Point the policy at the codec checkpoint:

```bash
CODEC_CKPT="$(find outputs/train_codecs -path '*/checkpoints/latest/pretrained_model' -type d | sort | tail -n 1)"
test -n "${CODEC_CKPT}"

HYDRA_FULL_ERROR=1 MUJOCO_GL=egl uv run praxis-train \
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

The policy checkpoint has this shape:

```text
outputs/train_policy/<date>/libero_bar_oat_pali/checkpoints/latest/pretrained_model
```

## Evaluate Locally

Local evaluation loads the checkpoint in the same Python environment as the
simulator:

```bash
POLICY_CKPT="$(find outputs/train_policy -path '*/checkpoints/latest/pretrained_model' -type d | sort | tail -n 1)"
test -n "${POLICY_CKPT}"

HYDRA_FULL_ERROR=1 MUJOCO_GL=egl uv run praxis-eval \
  policy=bar_oat_pali \
  policy.pretrained_path="${POLICY_CKPT}" \
  env.type=libero env.task=libero \
  eval.mode=local_inproc \
  eval.num_eval_per_task=2 eval.num_parallel_env=2 \
  eval.record_episodes_per_task=1 \
  eval.output_dir=outputs/eval/libero_bar_oat_pali_smoke
```

Results are written under:

```text
outputs/eval/libero_bar_oat_pali_smoke/
```

## Serve And Evaluate Remotely

Use the remote path when policy and simulator dependencies should not share one
environment.

```bash
POLICY_CKPT="$(find outputs/train_policy -path '*/checkpoints/latest/pretrained_model' -type d | sort | tail -n 1)"
test -n "${POLICY_CKPT}"

uv run praxis-serve \
  policy=bar_oat_pali \
  policy.pretrained_path="${POLICY_CKPT}" \
  host=0.0.0.0 \
  port=50051
```

In another terminal:

```bash
HYDRA_FULL_ERROR=1 MUJOCO_GL=egl uv run praxis-eval \
  env.type=libero env.task=libero \
  eval.mode=remote_grpc \
  server_address=127.0.0.1:50051 \
  eval.num_eval_per_task=2 eval.num_parallel_env=2 \
  eval.output_dir=outputs/eval/libero_remote_smoke
```

## Next

Read [Policy Cells](/foundations/policy-cells) to understand the name
`bar_oat_pali`, then [Comparison Design](/axes/comparison-design) before
changing any coordinate for a real study.
