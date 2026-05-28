# SmolVLM

SmolVLM is the smaller VLM backbone family in Praxis.

Example policies:

```text
bar_oat_smol
bar_fast_smol
ki_oat_smol
fm_smol
```

## Experimental Role

SmolVLM is used when the research question involves model size, preprocessing,
and efficiency. It gives a different tradeoff from PaliGemma2 and Qwen3-VL:

- smaller backbone;
- SmolVLA-style preprocessing path;
- causal prefix convention;
- potentially different batch-size and memory behavior.

## Preprocessing

Smol has shared preprocessing under:

```text
src/praxis/policies/backbones/smol/preprocessing.py
```

That path exists because Smol preprocessing is a backbone-family detail, not a global policy helper.

## Prefix Convention

Smol uses causal prefix attention, like Qwen:

```text
prefix tokens follow causal attention
action suffix follows the method-specific schedule
```

## Implementation

| Source path | Role |
| --- | --- |
| `src/praxis/policies/backbones/smol/vlm.py` | Shared Smol model/processor loading. |
| `src/praxis/policies/backbones/smol/prefix.py` | Smol prefix adapter. |
| `src/praxis/policies/backbones/smol/preprocessing.py` | Shared Smol preprocessing. |
| `src/praxis/policies/bar/backbones/smol/` | BAR-specific Smol path. |
| `src/praxis/policies/fm/backbones/smol/` | FM-specific Smol path. |
| `src/praxis/policies/ki/backbones/smol/` | KI-specific Smol path. |
| `src/praxis/configs/policy/*/backbone/smol.yaml` | Hydra backbone fragments. |

## Comparison Pattern

Use Smol to test whether a method or codec still works when the VLM family changes:

```text
bar_oat_pali vs bar_oat_smol
bar_fast_pali vs bar_fast_smol
ki_oat_pali vs ki_oat_smol
```

When comparing Smol to larger backbones, report image resolution and effective batch size. Those can be real confounders.
