# Codec Training

Standalone codec training estimates an action representation before token-based
policy training. The entry point is:

```bash
uv run praxis-train --config-name train_codecs \
  codecs@codec=tokens/oat \
  dataset.name=libero dataset.repo_id=lerobot/libero \
  device=cuda \
  trainer.use_wandb=false
```

Run real codec training on a compute node with a GPU. On a login node, use
`--cfg job` to inspect Hydra composition without starting the training job.

## Configuration Roots

The trainer config is:

```text
src/praxis/configs/train_codecs.yaml
```

Codec configs live under:

```text
src/praxis/configs/codecs/tokens/
src/praxis/configs/codecs/latents/
```

The selected object is `cfg.codec`. Standalone codec training should not grow a
parallel `cfg.tokenizer` root.

## Trainer Procedure

`CodecsTrainer` follows this sequence:

```python
policy_io = build_policy_io(dataset)
norm_stats = estimate_or_load_action_norm_stats(dataset)

codec = make_codec(cfg.codec)
codec.set_policy_io(policy_io)
codec.set_norm_stats(norm_stats)

if codec.has_trainable_parameters:
    optimize_reconstruction(codec, dataset)
elif codec.requires_fit:
    codec.fit(dataset_actions)

codec.save_pretrained(checkpoint_dir / "pretrained_model")
```

This procedure makes the codec artifact self-contained with respect to the
dataset schema and action scale.

## Artifact Format

A codec checkpoint contains:

```text
config.json
model.safetensors or pytorch_model.bin
policy_io.json
norm_stats.pt
optional tokenizer assets
```

FAST may save BPE assets. Learned codecs save neural weights and quantizer
state. The `policy_io.json` and `norm_stats.pt` sidecars are required for later
policy construction.

## Validation Metrics

The basic reconstruction metric is:

$$
\mathrm{MSE}_{\mathrm{reconst}}
= {1\over BHD}\sum_{b,h,d}
  (\hat{a}_{b,h,d}-a_{b,h,d})^2.
$$

For token-based policy work, also inspect:

- token sequence length and vocabulary size;
- invalid-decode behavior under arbitrary token arrays;
- prefix reconstruction when the codec claims prefix decode support;
- BAR/KI token cross entropy after policy training;
- decoded action error in policy validation;
- closed-loop success from `praxis-eval`.

Codec reconstruction is a representation metric, not a complete policy metric.
