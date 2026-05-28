# Policy Matrix

The supported policy matrix is defined in `src/praxis/policies/catalog.py`.
Policy names encode the experimental cell:

```text
<method>_<codec>_<backbone>
```

FM does not use a discrete action codec, so FM aliases use:

```text
fm_<backbone>
```

## Supported Cells

| Method | Codecs | Backbones |
| --- | --- | --- |
| `bar` | `oat`, `acodec`, `boat`, `fast`, `bin`, `quest` | `pali`, `qwen`, `smol` |
| `ki` | `oat`, `acodec`, `fast`, `bin`, `quest` | `pali`, `qwen`, `smol` |
| `fm` | none | `pali`, `qwen`, `smol` |

BOAT is currently BAR-only because its block schedule is part of the
representation-policy coupling used by BAR.

## Examples

| Policy | Interpretation |
| --- | --- |
| `bar_oat_pali` | BAR objective, OAT codec, PaliGemma2 backbone. |
| `bar_fast_qwen` | BAR objective, FAST codec, Qwen3-VL backbone. |
| `bar_boat_smol` | BAR objective, BOAT codec, SmolVLM backbone. |
| `ki_acodec_pali` | KI objective, ActionCodec codec, PaliGemma2 backbone. |
| `fm_qwen` | FM objective, no token codec, Qwen3-VL backbone. |

## Wiring Locations

| Layer | Files |
| --- | --- |
| Matrix and public names | `src/praxis/policies/catalog.py` |
| Runtime policy construction | `src/praxis/policies/factory.py` |
| BAR concrete classes | `src/praxis/policies/bar/variants.py` |
| FM concrete classes | `src/praxis/policies/fm/variants.py` |
| KI concrete classes | `src/praxis/policies/ki/variants.py` |
| Hydra aliases | `src/praxis/configs/policy/*.yaml` |
| Backbone fragments | `src/praxis/configs/policy/{bar,fm,ki}/backbone/*.yaml` |
| Codec fragments | `src/praxis/configs/policy/{bar,ki}/codec/*.yaml` |

## Adding A Public Cell

A new public cell should be introduced in all of the following places:

1. the catalog;
2. the method `variants.py`;
3. a Hydra alias under `src/praxis/configs/policy/`;
4. tests proving that the name resolves and composes.

The catalog keeps public policy names finite and explicit while leaving
family-specific implementation details local to the method, codec, or backbone
package that owns them.
