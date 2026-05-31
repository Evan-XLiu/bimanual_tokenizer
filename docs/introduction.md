# Introduction

Praxis is a controlled research substrate for vision-language-action (VLA)
policy work. The paper explains the motivation at a high level: VLA research
needs a way to separate design variables from the surrounding training,
serving, and evaluation stack. These docs are the technical companion: they
explain how to use and extend the repository.

## Why Praxis Exists

Benchmark-centric progress is useful, but it often hides what changed. A VLA
rollout score is produced by a full stack: dataset, objective, action
representation, VLM backbone, preprocessing, prompt format, normalization,
action decode, checkpoint loading, simulator setup, task selector, serving
mode, metric code, and artifact reporting.

If those factors move together, a method claim becomes hard to interpret. A
tokenizer comparison may also be a backbone comparison. A backbone comparison
may also be an evaluation-adapter comparison. An objective comparison may also
change the action horizon or normalization state.

Praxis makes the intended comparison explicit:

```text
change this coordinate
hold these contracts fixed
measure through this evaluation boundary
```

## What `praxis-vla` Owns

This repository is the policy-side package. It owns:

| Surface | Responsibility |
| --- | --- |
| Policy matrix | public policy cells and construction aliases |
| Objective families | BAR, FM, and KI training/inference implementations |
| Action representations | token codecs, learned codec checkpoints, codec factory |
| VLM backbones | PaliGemma2, Qwen3-VL, and SmolVLM adapters |
| Training | policy and codec trainers, recipes, checkpointing, metrics |
| Runtime | checkpoint loading, local inference adapter, gRPC serving |

`praxis-eval` owns benchmark execution: benchmark drivers, simulator
dependencies, task selection, rollout, metrics, media artifacts, and
benchmark-specific observation/action validation.

## The Core Abstraction

A token-based VLA policy is a policy cell:

$$
\pi_{m,c,b}
= \mathrm{Policy}(\text{objective}=m,\ \text{codec}=c,\ \text{backbone}=b),
$$

where $m$ is the objective, $c$ is the action representation, and $b$ is the
VLM backbone. FM policies remove the discrete codec from the inference path and
therefore use a null codec coordinate:

$$
\pi_{\mathrm{FM},\varnothing,b}.
$$

The public config names encode this cell:

```text
bar_oat_pali
ki_fast_qwen
fm_smol
```

The name is not cosmetic. It is the first statement of the experiment.

## What The Docs Are For

The docs are organized by reader task:

| Reader task | Start here |
| --- | --- |
| Understand the lab structure | [Foundations Overview](/foundations/overview) |
| Understand public policy names | [Policy Cells](/foundations/policy-cells) |
| Understand data/checkpoint contracts | [PolicyIO](/foundations/policy-io) |
| Compare objectives, codecs, or backbones | [Design Axes](/axes/overview) |
| Run training/evaluation commands | [Workflows](/workflows/overview) |
| Add a new research coordinate | [Extension](/extension/overview) |
| Look up command and artifact details | [Reference](/reference/cli) |

The docs do not try to reproduce the paper. The paper gives the motivation and
research framing. The docs give the implementation map, commands, artifact
contracts, and extension rules.
