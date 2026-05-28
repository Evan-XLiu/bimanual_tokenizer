# Introduction

Praxis is a research framework for building and comparing
vision-language-action policies. It is designed around a simple experimental
question:

> If the dataset, action horizon, training loop, and evaluator are fixed, what
> changes when we swap the policy objective, the VLM backbone, or the action
> representation?

The central abstraction is a policy cell:

$$
\pi_{m,c,b}
= \mathrm{Policy}(\text{method}=m,\ \text{codec}=c,\ \text{backbone}=b),
$$

where $m$ is the policy objective, $c$ is the action codec, and $b$ is the
pretrained VLM backbone. One runnable example cell is:

```text
bar_oat_pali = BAR objective + OAT codec + PaliGemma2 backbone
```

`bar_oat_pali` appears in quickstart-style commands because it exercises the
main system surfaces. It is not a claim that BAR, OAT, or PaliGemma2 is the
best or preferred choice for every setting.

## What You Can Do With Praxis

Praxis supports three common workflows.

| Workflow | What changes | Where to start |
| --- | --- | --- |
| Run an end-to-end example | OAT codec, then BAR OAT Pali policy | [Quickstart](/quickstart) |
| Run controlled comparisons | method, codec, or backbone axis | [Comparison Axes](/concepts/comparison-axes) |
| Extend the framework | new codec, backbone, or policy method | [Extension](/development/adding-codecs) |

The goal is to make comparisons interpretable. A useful experiment changes one
axis and keeps the rest of the system fixed:

```text
bar_oat_pali  vs  bar_oat_qwen      # backbone comparison
bar_oat_pali  vs  bar_fast_pali     # action-codec comparison
bar_oat_pali  vs  ki_oat_pali       # policy-objective comparison
```

## Implemented Design Axes

| Axis | Implemented choices |
| --- | --- |
| Policy objectives | BAR, FM, KI |
| VLM backbones | PaliGemma2, Qwen3-VL, SmolVLM |
| Action codecs | Binning, FAST, OAT, BOAT, QueST, ActionCodec |

The docs use BAR OAT Pali for the first runnable path because BAR directly
exposes action-token modelability, OAT provides a learned ordered action-token
representation, and PaliGemma2 gives a compact VLM path. Other cells are
first-class configurations rather than deviations from a preferred recipe.

## System Boundary

The core source tree is `src/praxis`. It owns:

- policy classes and factories;
- action codecs and codec checkpoints;
- policy and codec training;
- checkpoint loading and sidecars;
- serving;
- `PolicyEvalAdapter`.

Benchmark rollout lives in `praxis-eval`. That package owns benchmark
environments, task selection, rollout loops, metrics, media artifacts, simulator
setup, and benchmark-specific observation/action validation.

The boundary is:

```text
Praxis policy checkpoint
  -> PolicyEvalAdapter
  -> praxis-eval rollout
  -> metrics and media artifacts
```

This split keeps simulator integration separate from policy implementation.

## Data And Checkpoint Contract

Praxis uses `PolicyIO` to make datasets, codecs, policies, checkpoints, and
evaluation agree on feature names, shapes, horizons, and normalization modes.
The structural contract is saved as:

```text
policy_io.json
```

The numerical normalization tensors are saved as:

```text
norm_stats.pt
```

Token-based policies such as BAR and KI require an action codec checkpoint:

```text
policy.action_tokenizer.pretrained_path=/absolute/path/to/codec/pretrained_model
```

The codec checkpoint is part of the policy definition because action tokens only
have meaning under the action scale and horizon used to train the codec.

## Mathematical Summary

Let $o$ denote the observation context, including images, optional state, and
the task instruction. Let $a_{1:H}\in\mathbb{R}^{H\times D}$ be a normalized
action chunk. Let $z_{1:L}$ be action tokens when a codec is present.

BAR models action tokens directly:

$$
L_{\mathrm{BAR}}
=-\sum_{i=1}^{L} \log p_\theta(z_i \mid o,z_{<i}),
\qquad
\hat{a}_{1:H}=D(\hat{z}_{1:L}).
$$

FM trains a continuous flow expert:

$$
x_t=(1-t)a+t\epsilon,
\qquad
L_{\mathrm{FM}}
=\|v_\phi(x_t,t,h_\theta(o))-(\epsilon-a)\|_2^2.
$$

KI combines action-token supervision with an insulated flow expert:

$$
L_{\mathrm{KI}}
=\lambda_{\mathrm{CE}}L_{\mathrm{CE}}
+\lambda_{\mathrm{flow}}
 \|v_\phi(x_t,t,\mathrm{sg}(h_\theta(o)))-(\epsilon-a)\|_2^2.
$$

The action codec defines:

$$
z_{1:L}=E(a_{1:H}),
\qquad
\hat{a}_{1:H}=D(z_{1:L}).
$$

For token-based methods, codec quality is therefore not only reconstruction
quality. It also determines token sequence length, vocabulary size, prefix
semantics, invalid-decode behavior, and next-token prediction difficulty.

## Reading Path

Read in this order when learning Praxis:

1. [PolicyIO Contract](/concepts/policy-io) for the data and checkpoint
   contract.
2. [Comparison Axes](/concepts/comparison-axes) for the experimental design.
3. [Codec Interface](/codecs/overview) for action representations.
4. [Policy Objectives](/methods/overview) for BAR, FM, and KI.
5. [Backbone Adapters](/backbones/overview) for PaliGemma2, Qwen3-VL, and
   SmolVLM.
6. [BAR OAT Pali Example](/guides/bar-oat-pali-example) for a complete
   training and evaluation example.
