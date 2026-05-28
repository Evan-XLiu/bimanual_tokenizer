# PaliGemma2

PaliGemma2 is one of the VLM backbone families supported by Praxis.

Example policy:

```text
bar_oat_pali
```

## Experimental Role

PaliGemma2 provides a compact image-language model, a stable processor path,
and a useful visual-token budget for common robot-image resolutions.

It also exercises the main Praxis contracts:

- shared Pali VLM loading;
- Pali prefix adapter logic;
- BAR action-token CE;
- OAT tokenization;
- local eval through `praxis-eval`.

## Prefix Convention

Pali uses bidirectional prefix attention:

```text
prefix tokens attend bidirectionally
action suffix follows the method-specific action-token schedule
```

This means the observation/instruction prefix is treated as a fully visible context before action generation begins.

## State Prefix

BAR defaults to discrete text state for Pali-style token-generation paths. FM and KI default to continuous projected state when configured that way. The common rule is that state conditioning belongs in the VLM prefix, not hidden inside benchmark-specific code.

## Implementation

| Source path | Role |
| --- | --- |
| `src/praxis/policies/backbones/pali/vlm.py` | Shared Pali loading, processor batching, and vision freezing. |
| `src/praxis/policies/backbones/pali/prefix.py` | Pali prefix adapter for hidden/KV geometry. |
| `src/praxis/policies/bar/backbones/pali/` | BAR-specific Pali input construction and config. |
| `src/praxis/policies/fm/backbones/pali/` | FM-specific Pali input construction and config. |
| `src/praxis/policies/ki/backbones/pali/` | KI-specific Pali input construction and config. |
| `src/praxis/configs/policy/*/backbone/pali.yaml` | Hydra backbone fragments. |

## Initial Comparisons

Start with:

```text
bar_oat_pali
```

Then compare:

```text
bar_fast_pali
bar_bin_pali
bar_quest_pali
ki_oat_pali
fm_pali
```

Those comparisons isolate tokenizer and method behavior while keeping the VLM family fixed.
