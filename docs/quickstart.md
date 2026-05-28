# Quickstart

This page gives a minimal executable path using one policy configuration:

```text
policy=bar_oat_pali
```

That means:

- method: [BAR](/methods/bar)
- action codec: [OAT](/codecs/oat)
- VLM backbone: [PaliGemma2](/backbones/paligemma)

## Install

From the repo root:

```bash
uv sync --group dev --extra libero --extra pali --extra serving --extra logging
```

Verify that the package imports:

```bash
uv run python -c "import praxis; print('praxis import OK')"
```

## Train Or Bootstrap An Action Codec

Run training and evaluation commands on a compute node with a GPU. On a login
node, append `--cfg job` to inspect Hydra composition without launching the job.
The examples below are real training/evaluation commands; do not run them as
heavy jobs on a login node.

BAR and KI policies need a pretrained action codec checkpoint. For this example
path, train OAT:

```bash
HYDRA_FULL_ERROR=1 uv run praxis-train --config-name train_codecs \
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

The policy constructor requires this codec checkpoint because it carries both
codec parameters and action-normalization sidecars.

## Train The Example Policy

Use the OAT checkpoint from the previous step:

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

The trained policy artifact is written under:

```text
outputs/train_policy/<date>/libero_bar_oat_pali/checkpoints/latest/pretrained_model
```

## Evaluate The Checkpoint

Praxis delegates benchmark rollout to `praxis-eval`. The local evaluation mode
loads a Praxis policy checkpoint, wraps it with `PolicyEvalAdapter`, and calls
`praxis_eval.evaluate(...)`.

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

The evaluator writes:

```text
outputs/eval/libero_bar_oat_pali_smoke/results.json
outputs/eval/libero_bar_oat_pali_smoke/media/
```

## Serve The Checkpoint

Start a gRPC policy server:

```bash
POLICY_CKPT="$(find outputs/train_policy -path '*/checkpoints/latest/pretrained_model' -type d | sort | tail -n 1)"
test -n "${POLICY_CKPT}"

uv run praxis-serve \
  policy=bar_oat_pali \
  policy.pretrained_path="${POLICY_CKPT}" \
  host=0.0.0.0 \
  port=50051
```

Then evaluate through remote transport:

```bash
HYDRA_FULL_ERROR=1 MUJOCO_GL=egl uv run praxis-eval \
  env.type=libero env.task=libero \
  eval.mode=remote_grpc \
  server_address=127.0.0.1:50051 \
  eval.num_eval_per_task=2 eval.num_parallel_env=2 \
  eval.output_dir=outputs/eval/libero_remote_smoke
```

## Next

- [PolicyIO Contract](/concepts/policy-io) defines the data and normalization contract.
- [Comparison Axes](/concepts/comparison-axes) defines controlled experiment cells.
- [Codec Training](/codecs/training) explains codec checkpoints and validation.
