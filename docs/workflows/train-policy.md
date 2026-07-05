# Train A Policy

Policy training starts from a public policy cell.

## BAR Or KI

BAR and KI require a codec checkpoint:

```bash
CODEC_CKPT=/absolute/path/to/codec/pretrained_model

HYDRA_FULL_ERROR=1 MUJOCO_GL=egl uv run praxis-train \
  policy=bar_oat_pali \
  policy.action_tokenizer.pretrained_path="${CODEC_CKPT}" \
  dataset.name=libero dataset.repo_id=lerobot/libero \
  env.type=libero env.task=libero \
  device=cuda \
  trainer.max_steps=120 \
  trainer.use_wandb=false
```

Swap the policy name to change the cell:

```text
policy=bar_fast_pali
policy=ki_oat_qwen
policy=bar_boat_smol
```

## FM

FM does not use a codec checkpoint:

```bash
HYDRA_FULL_ERROR=1 uv run praxis-train \
  policy=fm_pali \
  dataset.name=libero dataset.repo_id=lerobot/libero \
  device=cuda \
  trainer.max_steps=120 \
  trainer.use_wandb=false
```

## Output

Policy artifacts are written under:

```text
outputs/train_policy/<date>/<run_name>/checkpoints/latest/pretrained_model
```

The checkpoint must carry `policy_io.json` and `norm_stats.pt` so serving and
evaluation can load the policy safely.
