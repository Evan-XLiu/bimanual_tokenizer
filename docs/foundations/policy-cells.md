# Policy Cells

A policy cell is the comparison unit in Praxis. It names the variables that a
study intends to control.

## Cell Notation

Token-based cells are:

$$
\pi_{m,c,b}
= \mathrm{Policy}(\text{objective}=m,\ \text{codec}=c,\ \text{backbone}=b).
$$

FM cells do not decode a discrete action representation at inference time:

$$
\pi_{\mathrm{FM},\varnothing,b}.
$$

The public config name follows the same structure:

```text
<method>_<codec>_<backbone>
fm_<backbone>
```

## Current Public Matrix

The public release contains 36 cells:

| Method | Codecs | Backbones | Count |
| --- | --- | --- | --- |
| `bar` | `oat`, `acodec`, `boat`, `fast`, `bin`, `quest` | `pali`, `qwen`, `smol` | 18 |
| `ki` | `oat`, `acodec`, `fast`, `bin`, `quest` | `pali`, `qwen`, `smol` | 15 |
| `fm` | none | `pali`, `qwen`, `smol` | 3 |

BOAT is BAR-only because its block schedule is part of the representation and
policy coupling used by BAR.

## Examples

| Name | Meaning |
| --- | --- |
| `bar_oat_pali` | BAR objective, OAT codec, PaliGemma2 backbone |
| `bar_fast_qwen` | BAR objective, FAST codec, Qwen3-VL backbone |
| `ki_acodec_smol` | KI objective, ActionCodec codec, SmolVLM backbone |
| `fm_qwen` | FM objective, no token codec, Qwen3-VL backbone |

## From Name To Object

Construction follows one path:

```text
public name
  -> catalog PolicySpec
  -> Hydra policy config
  -> policy factory class resolution
  -> policy object
  -> checkpoint artifact
```

The public inventory lives in `src/praxis/policies/catalog.py`. The runtime
constructor lives in `src/praxis/policies/factory.py`.

## What A Cell Does Not Guarantee

A cell name does not make a comparison fair by itself. It only names the
intended coordinates. The study still needs to hold contracts fixed: dataset,
action horizon, normalization, train recipe, codec checkpoint, evaluation
budget, and success metric.
