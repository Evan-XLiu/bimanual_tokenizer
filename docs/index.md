---
layout: home

hero:
  name: "Praxis"
  text: "Controlled VLA policy research, as code"
  tagline: "Praxis makes objectives, action representations, VLM backbones, and evaluators explicit so VLA experiments can say what changed, what stayed fixed, and why the result is interpretable."
  actions:
    - theme: brand
      text: "Read The Introduction"
      link: /introduction
    - theme: alt
      text: "Run The Quickstart"
      link: /quickstart
    - theme: alt
      text: "Design A Comparison"
      link: /axes/comparison-design

features:
  - title: "Policy Cells"
    details: "Use public names such as bar_oat_pali and fm_qwen as executable coordinates in a policy matrix."
    link: /foundations/policy-cells
    linkText: "Learn the cell model"
  - title: "Explicit Contracts"
    details: "Keep observations, actions, horizons, normalization, sidecars, and checkpoints visible across training, serving, and evaluation."
    link: /foundations/policy-io
    linkText: "Read PolicyIO"
  - title: "Research Axes"
    details: "Study BAR, FM, KI, action tokenizers, learned codecs, and VLM backbones as declared variables."
    link: /axes/overview
    linkText: "Open the axes"
  - title: "Operational Workflows"
    details: "Train codecs, train policies, serve checkpoints, and evaluate locally or remotely through praxis-eval."
    link: /workflows/overview
    linkText: "Run workflows"
  - title: "Extension Paths"
    details: "Add a codec, backbone, objective, or public cell at the module that owns that axis."
    link: /extension/overview
    linkText: "Extend Praxis"
  - title: "Reference Surface"
    details: "Look up CLI commands, Hydra composition, artifact formats, and factory behavior."
    link: /reference/cli
    linkText: "Open reference"
---

## The Problem

VLA policy research has outgrown one-architecture comparisons. A reported
rollout score may change the objective, action tokenizer, VLM backbone,
preprocessing, normalization, serving mode, simulator dependencies, benchmark
adapter, and metric code at the same time.

That stack-level result can track capability, but it is weak evidence for a
specific claim such as "this tokenizer is better" or "this backbone is
stronger." Praxis exists because controlled VLA research needs a codebase where
the variables are named before the measurement is made.

## What This Repository Is

This repository is `praxis-vla`, the policy side of Praxis. It owns policy
construction, action representations, VLM adapters, training, checkpointing,
serving, and the adapter used by `praxis-eval`.

`praxis-eval` owns closed-loop benchmark execution: simulator setup, task
selection, rollout, metrics, media, and benchmark-specific action validation.
The split keeps benchmark integration from becoming an unreported policy
variable.

## The Mental Model

Token-based policies are policy cells:

$$
\pi_{m,c,b}
= \mathrm{Policy}(\text{objective}=m,\ \text{codec}=c,\ \text{backbone}=b).
$$

Continuous FM policies have no discrete codec coordinate:

$$
\pi_{\mathrm{FM},\varnothing,b}.
$$

The quickstart cell is:

```text
bar_oat_pali = BAR objective + OAT codec + PaliGemma2 backbone
```

It is a compact runnable path, not a recommendation. It exercises codec
training, token-policy training, VLM adaptation, checkpoint loading, serving,
and evaluation through `praxis-eval`.

## Where To Start

| Goal | Page |
| --- | --- |
| Understand the repo purpose | [Introduction](/introduction) |
| Learn the core architecture | [Foundations Overview](/foundations/overview) |
| Run one end-to-end path | [Quickstart](/quickstart) |
| Design a controlled ablation | [Comparison Design](/axes/comparison-design) |
| Train or evaluate models | [Workflows Overview](/workflows/overview) |
| Add a new research coordinate | [Extension Overview](/extension/overview) |
| Look up command/config details | [Reference](/reference/cli) |
