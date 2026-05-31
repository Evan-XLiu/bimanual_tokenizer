# Design Axes Overview

Praxis exposes three policy-side axes:

| Axis | Question |
| --- | --- |
| objective | where should action generation live? |
| action representation | what action language should the policy predict or use for supervision? |
| backbone | how should image, state, and language context enter the policy? |

The axes are not independent in a mathematical sense. They interact. Praxis
does not remove those interactions; it gives them names and implementation
locations so a study can inspect what changed.

## Cell Form

Token-based policies:

$$
\pi_{m,c,b}
= \mathrm{Policy}(\text{objective}=m,\ \text{codec}=c,\ \text{backbone}=b).
$$

Continuous FM policies:

$$
\pi_{\mathrm{FM},\varnothing,b}.
$$

## Axis Pages

| Page | Use it for |
| --- | --- |
| [Objectives](/axes/objectives) | BAR, FM, KI supervision and inference |
| [Action Representations](/axes/action-representations) | Binning, FAST, OAT, BOAT, QueST, ActionCodec |
| [Backbones](/axes/backbones) | PaliGemma2, Qwen3-VL, SmolVLM adapters |
| [Comparison Design](/axes/comparison-design) | deciding what to hold fixed |
