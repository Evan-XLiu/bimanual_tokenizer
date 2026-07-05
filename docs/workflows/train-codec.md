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
