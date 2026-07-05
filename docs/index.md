---
layout: home

hero:
  name: "Praxis"
  text: "Controlled VLA policy research, as code"
  tagline: "Technical documentation for building, comparing, evaluating, and extending Praxis policy cells."
  actions:
    - theme: brand
      text: "Read The Introduction"
      link: /introduction
    - theme: alt
      text: "Run The Quickstart"
      link: /quickstart

features:
  - title: "Introduction"
    details: "Understand the benchmark-centric comparison problem, repo purpose, and policy-cell mental model."
    link: /introduction
    linkText: "Start reading"
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
---

## Documentation Map

Praxis has two related artifacts. The paper explains the high-level motivation:
why VLA research needs controlled design-space comparisons. These docs explain
the codebase: interfaces, artifacts, commands, modules, and extension paths.

| Goal | Page |
| --- | --- |
| Understand why this repository exists | [Introduction](/introduction) |
| Learn the policy-cell model | [Policy Cells](/foundations/policy-cells) |
| Learn the core architecture | [Foundations Overview](/foundations/overview) |
| Run one end-to-end path | [Quickstart](/quickstart) |
| Design a controlled ablation | [Comparison Design](/axes/comparison-design) |
| Train or evaluate models | [Workflows Overview](/workflows/overview) |
| Add a new research coordinate | [Extension Overview](/extension/overview) |
| Look up command/config details | [Reference](/reference/cli) |
