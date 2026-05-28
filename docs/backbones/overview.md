# Backbone Adapters

A VLM backbone adapter converts observation context into the prefix states used
by BAR, FM, and KI. For observation context $o$, the adapter implements:

$$
h_b(o)=\mathrm{BackboneAdapter}_b(o),
$$

where $b$ is one of `pali`, `qwen`, or `smol`.

## Adapter Responsibility

Backbone adapters own VLM-family-specific behavior:

| Responsibility | Examples |
| --- | --- |
| Model and processor loading | Hugging Face model id, tokenizer, image processor |
| Observation formatting | image ordering, task text, optional state prefix |
| Image preprocessing | resize, normalization, padding, processor-specific metadata |
| Attention convention | bidirectional or causal prefix behavior |
| Hidden-state geometry | hidden width, layer outputs, cache shape |
| Fine-tuning controls | freezing, LoRA targets, projector ownership |

These details are not global policy facts. They differ across PaliGemma2,
Qwen3-VL, and SmolVLM, so they are scoped to the backbone family and, when
needed, to the method-specific adapter.

## Implemented Backbones

| Key | Backbone | Prefix convention | Role in comparisons |
| --- | --- | --- | --- |
| `pali` | PaliGemma2 | bidirectional | compact bidirectional-prefix backbone |
| `qwen` | Qwen3-VL | causal | Qwen-family VLM comparison |
| `smol` | SmolVLM | causal | smaller-backbone and SmolVLA-style comparison |

The quickstart example `bar_oat_pali` uses the PaliGemma2 adapter.

## Source Layout

Backbone code appears at two levels:

```text
src/praxis/policies/backbones/<backbone>/
src/praxis/policies/{bar,fm,ki}/backbones/<backbone>/
```

The shared path owns VLM-family plumbing. The method-specific path owns how a
method consumes the VLM states, logits, and caches.

## Comparison Protocol

A backbone comparison should hold method, action representation, data, and
evaluation fixed:

```text
bar_oat_pali vs bar_oat_qwen
bar_oat_pali vs bar_oat_smol
ki_fast_pali vs ki_fast_qwen
fm_pali vs fm_smol
```

The interpretation is:

$$
\Delta_{\mathrm{backbone}}
=J(\pi_{m,c,b_1})-J(\pi_{m,c,b_2}),
$$

where $J$ is the chosen validation or rollout metric. This comparison is only
meaningful when the dataset, action horizon, train recipe, and evaluation
budget are controlled.
