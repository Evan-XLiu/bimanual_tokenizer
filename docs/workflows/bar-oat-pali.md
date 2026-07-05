# BAR OAT Pali Path

`bar_oat_pali` is a compact smoke path because it touches the main Praxis
surfaces:

```text
OAT codec checkpoint
  -> BAR token policy
  -> PaliGemma2 backbone adapter
  -> policy checkpoint
  -> local or remote evaluation
```

## Step 1: Train OAT

```bash
HYDRA_FULL_ERROR=1 uv run praxis-train --config-name train_codecs \
  codecs@codec=tokens/oat \
  dataset.name=libero dataset.repo_id=lerobot/libero \
  device=cuda \
  trainer.max_steps=500 trainer.val_every=200 trainer.checkpoint_every=200 \
  trainer.use_wandb=false
```

## Step 2: Train BAR OAT Pali

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

## Step 3: Evaluate

```bash
POLICY_CKPT="$(find outputs/train_policy -path '*/checkpoints/latest/pretrained_model' -type d | sort | tail -n 1)"
test -n "${POLICY_CKPT}"

HYDRA_FULL_ERROR=1 MUJOCO_GL=egl uv run praxis-eval \
  policy=bar_oat_pali \
  policy.pretrained_path="${POLICY_CKPT}" \
  env.type=libero env.task=libero \
  eval.mode=local_inproc \
  eval.num_eval_per_task=2 eval.num_parallel_env=2 \
  eval.output_dir=outputs/eval/libero_bar_oat_pali_smoke
```

## Next Comparisons

After the smoke path works, change one coordinate:

```text
bar_oat_pali  -> bar_fast_pali     # action representation
bar_oat_pali  -> bar_oat_qwen      # backbone
bar_oat_pali  -> ki_oat_pali       # objective
```
