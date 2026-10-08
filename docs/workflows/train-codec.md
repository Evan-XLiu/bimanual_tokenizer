# Train A Codec

Train a codec before training BAR or KI. The codec checkpoint defines the
action-token language used by the policy.

## Command Shape

```bash
HYDRA_FULL_ERROR=1 uv run praxis-train --config-name train_codecs \
  codecs@codec=tokens/oat \
  dataset.name=libero dataset.repo_id=lerobot/libero \
  device=cuda \
  trainer.max_steps=500 trainer.val_every=200 trainer.checkpoint_every=200 \
  trainer.use_wandb=false
```

Swap `codecs@codec=...` to train another representation:

```text
codecs@codec=tokens/bin
codecs@codec=tokens/fast
codecs@codec=tokens/oat
codecs@codec=tokens/boat
codecs@codec=tokens/quest
codecs@codec=tokens/acodec
```

## BiCAT YAM Joint Actions

Use `dataset=bicat_box` or `dataset=bicat_bottles` for the downloaded local
BiCAT tasks. Both configurations explicitly select `action_joint_angles`
(14 dimensions: left six joints and gripper, then right six joints and gripper).
The other EEF action columns remain in the source dataset and are not training
targets. `BICAT_DATA_ROOT` can override the parent directory containing both tasks.

```bash
uv run praxis-train --config-name train_codecs \
  codecs@codec=tokens/btt dataset=bicat_box
```

For policy training, use the same dataset choice with your policy and a codec
checkpoint trained on the same joint-action representation:

```bash
uv run praxis-train --config-name train_policy \
  policy=ki_btt_molmo dataset=bicat_box \
  policy.action_tokenizer.pretrained_path=/absolute/path/to/BiCAT/codec/pretrained_model
```

The policy uses `observation.state_joint_angles` and the top, left, and right
cameras. The local datasets supply their own FPS and metadata statistics.
Camera views with different resolutions are center-padded to a common canvas
before the VLM stacks them; the downloaded videos remain unchanged.
For other datasets with multiple action columns, set `dataset.action_key`
explicitly (`+dataset.action_key=...` when that config does not declare it).
Robot-specific bimanual sign conventions still need to match the YAM robot.
BTT uses its configured `shared_action_min` / `shared_action_max` internally;
check these ranges for YAM rather than assuming metadata statistics replace them.

## Output

Codec artifacts are written under:

```text
outputs/train_codecs/<date>/<run_name>/checkpoints/latest/pretrained_model
```

The `pretrained_model` directory should include:

```text
policy_io.json
norm_stats.pt
```

## Use In Policy Training

Pass the codec checkpoint to BAR or KI:

```bash
policy.action_tokenizer.pretrained_path=/absolute/path/to/pretrained_model
```

Do not train a policy with an uninitialized tokenizer. The policy constructor
requires a checkpoint because action-token ids need the codec's normalization
and decode semantics.
