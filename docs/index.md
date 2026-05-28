---
layout: home

hero:
  name: "Praxis"
  text: "A framework for comparing vision-language-action systems"
  tagline: "Praxis lets researchers train, evaluate, and extend VLA policies by varying the policy objective, VLM backbone, and action representation under a shared training and evaluation surface."
  actions:
    - theme: brand
      text: "Read the Introduction"
      link: /introduction
    - theme: alt
      text: "Run the Quickstart"
      link: /quickstart
    - theme: alt
      text: "BAR OAT Pali Example"
      link: /guides/bar-oat-pali-example

features:
  - title: "Compare VLA Design Axes"
    details: "Run controlled comparisons across policy objectives, VLM backbones, and action codecs instead of treating a VLA policy as one fixed recipe."
    link: /concepts/comparison-axes
    linkText: "Study the axes"
  - title: "Swap Action Representations"
    details: "Evaluate Binning, FAST, OAT, BOAT, QueST, and ActionCodec through a shared codec lifecycle and policy-facing tokenizer interface."
    link: /codecs/overview
    linkText: "Open codec docs"
  - title: "Swap VLM Backbones"
    details: "Use PaliGemma2, Qwen3-VL, and SmolVLM through backbone adapters that preserve family-specific preprocessing and prefix conventions."
    link: /backbones/overview
    linkText: "Open backbone docs"
  - title: "Study BAR, FM, and KI"
    details: "Compare token generation, continuous flow matching, and knowledge-insulated policies with their objectives and inference paths stated explicitly."
    link: /methods/overview
    linkText: "Open method docs"
  - title: "Build New Components"
    details: "Add a new codec, backbone, or policy method by wiring the registry, Hydra config, implementation class, and focused tests at the correct boundary."
    link: /development/adding-codecs
    linkText: "Start extending"
  - title: "Evaluate Through praxis-eval"
    details: "Keep benchmark rollout, task selection, metrics, and simulator setup in praxis-eval while Praxis owns policy construction and checkpoints."
    link: /concepts/eval-boundary
    linkText: "Read the boundary"
---

## What Praxis Is

Praxis is a modular research codebase for VLA policy comparison. It makes the
policy objective, action representation, and VLM backbone explicit so
experiments can change one axis at a time instead of treating a VLA policy as
one fixed recipe.

The documentation is organized for three reader tasks:

| Reader task | Start here |
| --- | --- |
| Understand the project | [Introduction](/introduction) |
| Run a small end-to-end path | [Quickstart](/quickstart) |
| Compare methods, codecs, or backbones | [Comparison Axes](/concepts/comparison-axes) |
| Add a new action codec | [New Action Codec](/development/adding-codecs) |
| Add a new VLM backbone | [New VLM Backbone](/development/adding-backbones) |

## Quickstart Example

```text
bar_oat_pali = BAR objective + OAT codec + PaliGemma2 backbone
```

The quickstart uses `bar_oat_pali` because it exercises the main Praxis
surfaces in one runnable path:

- action codec training and checkpointing;
- token-generating VLA policy training;
- VLM backbone adaptation;
- policy checkpoint loading;
- local and remote evaluation through `praxis-eval`.

It is an example configuration, not a project recommendation. For the full
explanation, read [Introduction](/introduction). For commands, use
[Quickstart](/quickstart).
