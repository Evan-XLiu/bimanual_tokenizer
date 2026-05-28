# Qwen3-VL

Qwen3-VL is the Qwen-family VLM backbone in Praxis.

Example policies:

```text
bar_oat_qwen
bar_fast_qwen
ki_oat_qwen
fm_qwen
```

## Experimental Role

Qwen3-VL provides a different pretrained VLM prior, image processor,
tokenization path, and attention convention from PaliGemma2. It is used to test
whether an observed result belongs to a policy method or action codec rather
than to a single VLM family.

## Prefix Convention

Qwen uses causal prefix attention:

```text
prefix tokens follow the backbone's causal convention
action suffix follows the method-specific action-token schedule
```

This is intentionally different from Pali. The shared BAR/KI action-token helpers support the common action suffix behavior, but each VLM family declares its own prefix convention.

## Implementation

| Source path | Role |
| --- | --- |
| `src/praxis/policies/backbones/qwen/vlm.py` | Shared Qwen model/processor loading and batching. |
| `src/praxis/policies/backbones/qwen/prefix.py` | Qwen prefix adapter and attention geometry. |
| `src/praxis/policies/bar/backbones/qwen/` | BAR-specific Qwen path. |
| `src/praxis/policies/fm/backbones/qwen/` | FM-specific Qwen path. |
| `src/praxis/policies/ki/backbones/qwen/` | KI-specific Qwen path. |
| `src/praxis/configs/policy/*/backbone/qwen.yaml` | Hydra backbone fragments. |

## Comparison Pattern

To compare Qwen against Pali:

```text
bar_oat_pali vs bar_oat_qwen
bar_fast_pali vs bar_fast_qwen
fm_pali vs fm_qwen
```

Use the same dataset, horizon, codec checkpoint quality, and train recipe. If image resolution changes, treat that as a separate ablation.
