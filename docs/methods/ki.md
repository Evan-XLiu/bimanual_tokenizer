# KI Objective

KI is the knowledge-insulated policy objective.

KI combines two signals:

1. action-token cross entropy for the VLM;
2. flow matching for a continuous action expert.

The central design choice is that the action expert receives VLM context through
a detach boundary. The flow loss trains the expert without directly rewriting
the VLM prefix through the continuous-action objective. This is the knowledge
insulation principle: pretrained VLM knowledge remains the context provider,
while continuous-control gradients are routed through a separate expert.

## Data Model

Like BAR, KI uses an action tokenizer:

$$
T_{1:L}=\mathcal{T}(a_{1:H})
$$

Like FM, KI also trains an action expert on noisy continuous actions:

$$
x_t = (1-t)a + t\epsilon,
\qquad
u^* = \epsilon - a
$$

## Objective

KI combines CE and flow:

$$
L_{\mathrm{KI}}
= \lambda_{\mathrm{CE}} L_{\mathrm{CE}}
 + \lambda_{\mathrm{flow}} L_{\mathrm{flow}}
$$

with:

$$
L_{\mathrm{CE}}
= - \sum_i m_i \log p_\theta(T_i \mid o,T_{<i})
$$

and:

$$
L_{\mathrm{flow}}
= \| v_\phi(x_t,t,\mathrm{sg}(c_\theta^{\mathrm{prefix}}(o))) - (\epsilon-a) \|_2^2
$$

Here $c_\theta^{\mathrm{prefix}}(o)$ is the selected VLM prefix context exposed to the action expert, and `sg` is the knowledge-insulation boundary. The flow expert does not condition on the full ground-truth action-token suffix.

When both branches are active, Praxis uses one integrated forward pass: the CE branch supervises action-token positions, and the action expert reads the matching VLM prefix context with `detach_expert_context=true`. In flow-only mode, the expert can also use a no-action-token FM-style prompt and detach that prefix.

## Warmstart

KI can warmstart with CE before enabling flow:

```text
if global_step < ce_warmstart_steps:
    flow_weight = 0
```

This lets the VLM learn the action-token language before the continuous expert depends on its prefix.

The effective loss is:

$$
\lambda_{\mathrm{flow}}(s)
=
\begin{cases}
0, & s < s_{\mathrm{warm}} \\
\lambda_{\mathrm{flow}}, & s \ge s_{\mathrm{warm}}
\end{cases}
$$

where $s$ is the global training step.

## Algorithm Sketch

KI is easiest to read as BAR supervision plus an insulated FM branch:

```python
# src/praxis/policies/ki/core/policy.py
tokens = action_tokenizer.tokenize(actions)
prefix, hidden = vlm_forward_with_action_token_labels(observation, tokens)
ce_loss = cross_entropy(hidden.action_logits, tokens)

expert_context = select_prefix_context(hidden).detach()
x_t, target_velocity = flow_training_sample(actions)
pred_velocity = action_expert(x_t, t, expert_context)
flow_loss = mse(pred_velocity, target_velocity)

if global_step < ce_warmstart_steps:
    flow_weight = 0.0
loss = ce_weight * ce_loss + flow_weight * flow_loss
```

The real implementation keeps the one-forward adapter in
`src/praxis/policies/ki/core/one_forward.py` and the CE plus flow objective in
`src/praxis/policies/ki/core/policy.py`.

## Experimental Role

BAR asks the VLM to directly emit action tokens. FM asks the VLM to support a
continuous action expert. KI studies the intermediate design:

- token CE teaches the VLM action structure;
- flow matching handles smooth continuous control;
- detach prevents the flow objective from dominating the VLM representation.

This makes KI the relevant method for testing whether action-token supervision
improves continuous-action policies.

## Inference

KI inference uses the VLM prefix/KV context to condition the action expert. It does not require the VLM to finish an autoregressive token decode at inference time in the same way BAR does.

The high-level path is:

```text
build VLM prefix
  -> build prefix cache/context for action expert
  -> sample continuous action chunk by flow integration
  -> unnormalize actions
```

## Supported KI Matrix

KI supports:

```text
codecs:    oat, acodec, fast, bin, quest
backbones: pali, qwen, smol
```

BOAT is not currently part of KI.

Examples:

```text
ki_oat_pali
ki_fast_qwen
ki_acodec_smol
```

## Implementation Modules

| Source path | Role |
| --- | --- |
| `src/praxis/policies/ki/variants.py` | Concrete KI method x codec x backbone classes. |
| `src/praxis/policies/ki/core/policy.py` | KI CE plus flow objective wiring. |
| `src/praxis/policies/ki/core/config.py` | KI weights, warmstart, and expert config. |
| `src/praxis/policies/ki/core/one_forward.py` | Adapter for CE loss from hidden states in integrated paths. |
| `src/praxis/policies/core/vlm/training.py` | Shared action-token CE helpers. |
| `src/praxis/policies/core/action_expert/` | Shared flow expert lifecycle. |

## Configuration

Public aliases:

```text
src/praxis/configs/policy/ki_<codec>_<backbone>.yaml
```

Fragments:

```text
src/praxis/configs/policy/ki/backbone/
src/praxis/configs/policy/ki/codec/
```

Training recipe:

```text
src/praxis/configs/train_recipe/policy/ki.yaml
```
