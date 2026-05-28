# Source Layout

The conceptual core of Praxis is `src/praxis`. Documentation should treat this
tree as the source of truth for policy construction, action representations,
training, checkpointing, serving, and evaluation adaptation.

## Main Packages

| Package | Responsibility |
| --- | --- |
| `praxis.policies` | BAR, FM, and KI policies; VLM backbone adapters; policy catalog; policy factory; `praxis-eval` adapter. |
| `praxis.codecs` | Action codecs, latent autoencoders, tokenizers, codec factory, and codec checkpoint format. |
| `praxis.training` | Policy and codec trainers, distributed training, optimizer grouping, checkpoint management, inline evaluation scheduling. |
| `praxis.configs` | Hydra composition roots for policies, codecs, datasets, trainers, and method-specific train recipes. |
| `praxis.serving` | Policy server runtime and gRPC integration. |
| `praxis.data` | Dataset metadata, feature extraction, and LeRobot-facing utilities. |
| `praxis.nn` | Neural components with demonstrated cross-family reuse. |
| `praxis.scripts` | Installable CLI entry points such as `praxis-train`, `praxis-eval`, and `praxis-serve`. |

## Policy Tree

```text
src/praxis/policies/
  catalog.py              # supported method x codec x backbone cells
  factory.py              # config and checkpoint policy construction
  eval_adapter.py         # BasePolicy -> praxis-eval adapter
  backbones/              # shared VLM-family loaders and adapters
  bar/                    # BAR objective family
  fm/                     # FM objective family
  ki/                     # KI objective family
  core/                   # shared policy lifecycle, CE helpers, action expert
```

The catalog is the public inventory for names such as `bar_oat_pali` and
`ki_fast_qwen`. Method-specific packages own method behavior; backbone-specific
packages own VLM-family details.

## Codec Tree

```text
src/praxis/codecs/
  factory.py              # make_codec, register_codec, available_codecs
  core/                   # BaseCodec, BaseTokenizer, shared codec lifecycle
  tokens/                 # discrete action-token codecs
  latents/                # continuous latent autoencoders
```

Discrete action representations inherit from `BaseTokenizer`. Learned token
codecs such as OAT, BOAT, QueST, and ActionCodec may use latent autoencoders
internally, but the policy-facing object is still the token codec.

## Configuration Tree

```text
src/praxis/configs/
  train_policy.yaml
  train_codecs.yaml
  eval_policy.yaml
  serve.yaml
  policy/
  codecs/
  train_recipe/
```

Policy aliases follow the matrix names:

```text
policy=bar_oat_pali
policy=bar_bin_qwen
policy=fm_smol
policy=ki_acodec_pali
```

Standalone codec training selects a codec group:

```text
codecs@codec=tokens/oat
codecs@codec=tokens/bin
codecs@codec=latents/oat
```

## Evaluation Boundary

Praxis owns policy construction, training, checkpoint loading, action codecs,
serving, and `PolicyEvalAdapter`. `praxis-eval` owns benchmark environment
setup, rollout, metrics, media artifacts, and benchmark-specific observation and
action validation.

This boundary keeps simulator dependencies out of the policy framework and lets
new VLA methods be evaluated without importing benchmark internals.
