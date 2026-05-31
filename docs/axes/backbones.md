# Backbones

The backbone axis controls how observations become VLM prefix states:

$$
h_b(o)=\mathrm{BackboneAdapter}_b(o).
$$

Backbone adapters own family-specific behavior. That includes processor
loading, image formatting, prompt convention, attention geometry, hidden-state
shape, cache behavior, freezing, and LoRA targets.

## Implemented Backbones

| Key | Backbone | Prefix convention | Role |
| --- | --- | --- | --- |
| `pali` | PaliGemma2 | bidirectional prefix | compact bidirectional VLM path |
| `qwen` | Qwen3-VL | causal prefix | Qwen-family VLM comparison |
| `smol` | SmolVLM | causal prefix | smaller VLM and SmolVLA-style preprocessing comparison |

## Adapter Responsibilities

| Responsibility | Examples |
| --- | --- |
| model loading | Hugging Face model id, processor, tokenizer |
| observation formatting | image ordering, task text, optional state prefix |
| image preprocessing | resize, normalization, padding, processor metadata |
| attention convention | bidirectional prefix or causal prefix |
| hidden state geometry | hidden width, cache shape, selected layers |
| fine-tuning controls | freeze policy, LoRA targets, projector ownership |

These are axis-local details. They should not leak into codec code or evaluator
code.

## Comparison Pattern

Backbone comparisons should hold objective, action representation, data, and
evaluation fixed:

```text
bar_oat_pali vs bar_oat_qwen
bar_oat_pali vs bar_oat_smol
ki_fast_pali vs ki_fast_qwen
fm_pali      vs fm_smol
```

The interpretation is only clean if the observation keys, image preprocessing
policy, action horizon, train recipe, and evaluation budget are controlled.

## Implementation

Backbone code exists at two levels:

```text
src/praxis/policies/backbones/<backbone>/
src/praxis/policies/{bar,fm,ki}/backbones/<backbone>/
```

The shared path owns VLM-family plumbing. The method-specific path owns how an
objective consumes VLM states, logits, caches, and prefix geometry.

## Backbone Comparison Checklist

Before comparing backbones, declare:

| Contract | Example |
| --- | --- |
| objective | BAR, FM, or KI |
| action representation | OAT checkpoint, FAST checkpoint, or no codec for FM |
| observation keys | image keys and optional state keys |
| prompt convention | task text and chat/prefix formatting |
| train recipe | freeze policy, LoRA settings, optimizer groups |
| evaluator | same benchmark, task selector, rollout budget |
