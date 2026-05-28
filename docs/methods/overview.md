# Policy Objectives

Praxis implements three VLA policy objectives: BAR, FM, and KI. They share the
same observation contract and differ in the supervision applied to actions and
VLM prefix states.

Let $o$ denote the observation context, including images, optional state, and
the task instruction. Let $a_{1:H}\in\mathbb{R}^{H\times D}$ be a normalized
action chunk. When a discrete action codec is present, let
$z_{1:L}=E(a_{1:H})$.

## Objective Families

| Method | Action representation | VLM supervision | Action-generation path |
| --- | --- | --- | --- |
| BAR | discrete action tokens | action-token CE | VLM generates tokens, codec detokenizes |
| FM | continuous action chunks | flow conditioning only | action expert integrates a continuous flow |
| KI | tokens plus continuous actions | action-token CE plus detached flow context | action expert samples continuous actions from VLM prefix context |

## BAR

BAR trains the VLM as an action-token model:

$$
L_{\mathrm{BAR}}
=-\sum_{i=1}^{L}m_i\log p_\theta(z_i\mid o,z_{<i}).
$$

Inference produces tokens and decodes them:

$$
\hat{z}_{1:L}\sim p_\theta(\cdot\mid o),
\qquad
\hat{a}_{1:H}=D(\hat{z}_{1:L}).
$$

The BAR implementation also supports blockwise schedules, where a stage
predicts a group $G_s=z_{b_{s-1}+1:b_s}$ in parallel from the realized prefix.

## FM

FM removes action tokens from the policy path. It samples Gaussian noise
$\epsilon\sim\mathcal{N}(0,I)$ and interpolation time $t$, then defines:

$$
x_t=(1-t)a+t\epsilon,\qquad
u^*=\epsilon-a.
$$

The flow objective is:

$$
L_{\mathrm{FM}}
=\|v_\phi(x_t,t,h_\theta(o))-u^*\|_2^2.
$$

Inference starts from noise and integrates the learned velocity field from
$t=1$ to $t=0$.

## KI

KI combines BAR-style token supervision with an FM-style continuous expert. The
expert receives VLM context through a stop-gradient boundary:

$$
L_{\mathrm{KI}}
=\lambda_{\mathrm{CE}}L_{\mathrm{CE}}(z\mid o)
+\lambda_{\mathrm{flow}}
  \|v_\phi(x_t,t,\mathrm{sg}(h_\theta(o)))-(\epsilon-a)\|_2^2.
$$

This objective studies whether action-token supervision improves the VLM prefix
representation while continuous control is produced by a separate expert.

## Method Selection

| Experimental question | Method |
| --- | --- |
| How well can a VLM directly model the action-token language? | BAR |
| How strong is continuous action generation without tokenization? | FM |
| Does token supervision improve a continuous-action expert? | KI |

The quickstart path uses BAR because it is a direct way to show action-codec
training followed by token-generating policy training. FM and KI are included to
separate representation effects from continuous-control generation effects.

## Shared Implementation

| Concept | Files |
| --- | --- |
| Method matrix | `src/praxis/policies/catalog.py` |
| BAR variants | `src/praxis/policies/bar/variants.py` |
| FM variants | `src/praxis/policies/fm/variants.py` |
| KI variants | `src/praxis/policies/ki/variants.py` |
| Shared action-token CE | `src/praxis/policies/core/vlm/training.py` |
| Shared action expert | `src/praxis/policies/core/action_expert/` |
| VLM prefix contracts | `src/praxis/policies/core/vlm/prefix.py` |

## Minimal Pseudocode

```python
batch = build_policy_batch(dataset_examples, policy_io)
prefix = backbone_adapter.encode_observation(batch.observation)

if method == "bar":
    tokens = action_tokenizer.tokenize(batch.actions)
    loss = token_cross_entropy(prefix, tokens)
elif method == "fm":
    x_t, target = flow_sample(batch.actions)
    loss = mse(action_expert(x_t, t, prefix), target)
elif method == "ki":
    tokens = action_tokenizer.tokenize(batch.actions)
    ce = token_cross_entropy(prefix, tokens)
    x_t, target = flow_sample(batch.actions)
    flow = mse(action_expert(x_t, t, prefix.detach()), target)
    loss = ce_weight * ce + flow_weight * flow
```
