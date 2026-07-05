# Objectives

The objective axis defines supervision, inference, and gradient flow.

| Method | Action representation | Supervision | Inference |
| --- | --- | --- | --- |
| BAR | discrete tokens | token CE with token-wise or blockwise schedule | generate tokens, then detokenize |
| FM | continuous actions | flow-matching velocity regression | integrate a continuous action flow |
| KI | tokens plus continuous expert | token CE plus stop-gradient-insulated flow loss | sample continuous actions from an expert conditioned on VLM context |

## BAR

BAR treats action generation as scheduled action-token prediction. Given action
tokens $z_{1:L}$, BAR minimizes token cross entropy:

$$
L_{\mathrm{BAR}}
=-\sum_i \log p_\theta(z_i\mid o,z_{<i}).
$$

The schedule can be token-wise, blockwise, or parallel. At inference time, BAR
generates a token sequence and calls the codec decoder:

$$
\hat{a}_{1:H}=D(\hat{z}_{1:L}).
$$

Use BAR when the study asks how modelable an action-token language is under VLM
token prediction.

Technical notes:

| Concern | Where it lives |
| --- | --- |
| schedule and token sequence handling | `src/praxis/policies/bar/core/` |
| VLM-backed BAR policy | `src/praxis/policies/bar/core/vlm/` |
| FAST-specific decode path | `src/praxis/policies/bar/codecs/fast/` |
| concrete public variants | `src/praxis/policies/bar/variants.py` |

## FM

FM removes the discrete token coordinate from inference. For action chunk $a$,
noise $\epsilon$, and interpolation time $t$:

$$
x_t=(1-t)a+t\epsilon,
\qquad
u^\star=\epsilon-a.
$$

The objective is:

$$
L_{\mathrm{FM}}
=\|v_\phi(x_t,t,h_\theta(o))-u^\star\|_2^2.
$$

At inference time, the action expert integrates from noise toward the data
endpoint. Use FM as the continuous-control reference point when no tokenizer
should be in the policy path.

FM is useful in comparisons because it removes tokenization error, invalid
token sequences, and discrete decode latency from deployed control. The tradeoff
is that the VLM no longer receives an action-token language as structured
supervision.

Technical notes:

| Concern | Where it lives |
| --- | --- |
| policy implementation | `src/praxis/policies/fm/core/` |
| action expert utilities | `src/praxis/policies/core/action_expert/` |
| concrete public variants | `src/praxis/policies/fm/variants.py` |

## KI

KI tests whether action-token supervision helps VLM context even when deployed
control is continuous. It combines token CE with a detached flow branch:

$$
L_{\mathrm{KI}}
=\lambda_{\mathrm{CE}}L_{\mathrm{CE}}
+\lambda_{\mathrm{flow}}
\|v_\phi(x_t,t,\mathrm{sg}(h_\theta(o)))-(\epsilon-a)\|_2^2.
$$

The stop-gradient boundary insulates the continuous expert from directly
overwriting the VLM representation. Use KI when the study asks whether action
tokens are useful as representation supervision rather than as the final
deployment-time action language.

KI has two protocol-sensitive parts. First, the CE branch needs a valid token
codec checkpoint. Second, the flow branch receives VLM context through a
stop-gradient boundary, so changing that boundary changes the objective family
rather than a minor hyperparameter.

Technical notes:

| Concern | Where it lives |
| --- | --- |
| KI policy logic | `src/praxis/policies/ki/core/` |
| one-forward helper | `src/praxis/policies/ki/core/one_forward.py` |
| action expert utilities | `src/praxis/policies/core/action_expert/` |
| concrete public variants | `src/praxis/policies/ki/variants.py` |

## Implementation

| Concept | Files |
| --- | --- |
| public methods | `src/praxis/policies/catalog.py` |
| BAR | `src/praxis/policies/bar/` |
| FM | `src/praxis/policies/fm/` |
| KI | `src/praxis/policies/ki/` |
| shared VLM utilities | `src/praxis/policies/core/vlm/` |
| shared action expert | `src/praxis/policies/core/action_expert/` |

## Objective Comparison Checklist

Before comparing objectives, declare:

| Contract | Example |
| --- | --- |
| action representation | OAT checkpoint path, token length, decode behavior |
| backbone | PaliGemma2, Qwen3-VL, or SmolVLM |
| action horizon | `policy.action_horizon` |
| train recipe | optimizer, LR schedule, warmstart choices |
| rollout protocol | local/remote mode, benchmark task selector, evaluation budget |
