# Why Praxis

Praxis exists because VLA policy research has outgrown one-architecture
comparisons. A reported rollout score may change the objective, action
tokenizer, VLM backbone, preprocessing, normalization, serving mode, simulator
dependencies, benchmark adapter, and metric code at the same time.

That stack-level result can track capability, but it is weak evidence for a
specific claim such as "this tokenizer is better" or "this backbone is
stronger." Controlled VLA research needs a codebase where the variables are
named before the measurement is made.

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

## What Praxis Tries To Fix

Praxis does not claim that one objective, tokenizer, backbone, or benchmark is
best. It tries to make those claims easier to test by exposing:

| Need | Praxis response |
| --- | --- |
| Name the variable under study | public policy cells such as `bar_oat_pali` |
| Preserve data and action assumptions | `PolicyIO`, `policy_io.json`, and `norm_stats.pt` |
| Keep action-token meaning explicit | codec checkpoints referenced by token policies |
| Avoid benchmark glue as a hidden variable | `PolicyEvalAdapter` plus `praxis-eval` ownership |
| Add new research coordinates cleanly | axis-local extension paths for codecs, backbones, and objectives |

The practical goal is simple: a study should be able to say what changed, what
stayed fixed, and why the resulting evidence is interpretable.
