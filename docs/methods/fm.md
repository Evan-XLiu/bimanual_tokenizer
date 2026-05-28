# FM Objective

FM is the continuous-action flow-matching policy objective.

FM does not use a discrete action tokenizer. A policy name is only:

```text
fm_<backbone>
```

Examples:

```text
fm_pali
fm_qwen
fm_smol
```

## Data Model

Let $a \in R^{H \times D}$ be a normalized action chunk and $\epsilon \sim N(0,I)$ be Gaussian noise.

Praxis samples a time $t \in [t_{\min},1]$ and builds:

$$
x_t = (1-t)a + t\epsilon
$$

The target velocity is:

$$
u^* = \epsilon - a
$$

The code uses this convention because $t=1$ is noise and $t=0$ is data.

## Objective

FM trains an action expert to predict the velocity:

$$
L_{\mathrm{FM}}
= \| v_\phi(x_t,t,h_\theta(o)) - u^* \|_2^2
$$

where:

- $h_\theta(o)$ is the VLM prefix representation of the observation;
- $v_\phi$ is the action expert;
- $\theta$ and $\phi$ are trained end-to-end.

Unlike BAR, the VLM is not asked to generate action tokens. It learns to produce useful conditioning states for a continuous action expert.

## Inference

Inference samples noise and integrates backward from $t=1$ to $t=0$:

$$
x_{t+\Delta t} = x_t + v_\phi(x_t,t,h_\theta(o)) \Delta t
$$

with:

$$
\Delta t = -{1 \over N}
$$

After $N$ Euler steps, the final $x_0$ is the predicted normalized action chunk.

## Algorithm Sketch

FM is the simplest policy method in Praxis because there is no action-token
codec in the policy path:

```python
# src/praxis/policies/fm/core/policy.py
prefix = vlm_prefix(observation)
noise = torch.randn_like(actions)
t = sample_time(batch)
x_t = (1 - t) * actions + t * noise
target_velocity = noise - actions
pred_velocity = action_expert(x_t, t, prefix)
loss = mse(pred_velocity, target_velocity)
```

At inference, `src/praxis/policies/core/action_expert/sampling.py` starts from
Gaussian noise and applies the same expert repeatedly with an Euler sampler.

## Experimental Role

FM answers a different question from BAR:

> If discrete action tokens are removed entirely, can the VLM still condition a
> strong continuous action generator?

This objective is useful when the comparison should isolate:

- tokenization error;
- continuous-control quality;
- the value of VLM prefix conditioning without token supervision;
- token-based VLA methods against a continuous-action baseline under the same
  backbone and data.

## Implementation Modules

| Source path | Role |
| --- | --- |
| `src/praxis/policies/fm/variants.py` | Concrete FM backbone variants. |
| `src/praxis/policies/fm/core/policy.py` | FM objective wiring. |
| `src/praxis/policies/fm/core/config.py` | FM config defaults. |
| `src/praxis/policies/core/action_expert/policy.py` | Shared VLM/action-expert lifecycle. |
| `src/praxis/policies/core/action_expert/flow.py` | Flow training sample construction. |
| `src/praxis/policies/core/action_expert/sampling.py` | Euler sampling from noise to data. |

## Configuration

Public aliases:

```text
src/praxis/configs/policy/fm_pali.yaml
src/praxis/configs/policy/fm_qwen.yaml
src/praxis/configs/policy/fm_smol.yaml
```

Backbone fragments:

```text
src/praxis/configs/policy/fm/backbone/
```

Training recipe:

```text
src/praxis/configs/train_recipe/policy/fm.yaml
```
