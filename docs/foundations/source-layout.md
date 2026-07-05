# Source Layout

The source tree is organized around the same variables and contracts described
by the docs.

## Main Packages

| Package | Responsibility |
| --- | --- |
| `praxis.policies` | BAR, FM, KI policies, VLM adapters, policy catalog, policy factory, evaluation adapter |
| `praxis.codecs` | action representations, tokenizers, latent autoencoders, codec factory |
| `praxis.training` | trainers, checkpointing, distributed runtime, metrics, inline evaluation |
| `praxis.configs` | Hydra composition roots for policies, codecs, datasets, trainers, recipes |
| `praxis.serving` | gRPC policy server and remote inference runtime |
| `praxis.data` | dataset metadata and LeRobot-facing utilities |
| `praxis.nn` | reusable neural network layers and architecture utilities |
| `praxis.scripts` | installable CLI entry points |

## Policy Tree

```text
src/praxis/policies/
  catalog.py              # public policy matrix
  factory.py              # config and checkpoint construction
  eval_adapter.py         # BasePolicy -> praxis-eval adapter
  backbones/              # shared VLM-family adapters
  bar/                    # BAR objective family
  fm/                     # FM objective family
  ki/                     # KI objective family
  core/                   # shared policy lifecycle and utilities
```

## Codec Tree

```text
src/praxis/codecs/
  factory.py              # make_codec, register_codec, available_codecs
  core/                   # shared codec lifecycle
  tokens/                 # policy-facing action tokenizers
  latents/                # learned latent autoencoders
```

## Configuration Tree

```text
src/praxis/configs/
  train_policy.yaml
  train_codecs.yaml
  eval_policy.yaml
  serve.yaml
  policy/
  codecs/
  dataset/
  train_recipe/
```

Hydra aliases follow the policy matrix:

```text
policy=bar_oat_pali
policy=ki_fast_qwen
policy=fm_smol
```
