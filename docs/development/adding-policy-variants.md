# Add A Policy Variant

A policy variant is a public matrix cell such as:

```text
bar_oat_pali
ki_fast_qwen
fm_smol
```

Add a variant when an existing method, codec, and backbone combination is truly supported.

## Files To Touch

1. `src/praxis/policies/catalog.py`
2. `src/praxis/policies/<method>/variants.py`
3. `src/praxis/configs/policy/<variant>.yaml`
4. method codec/backbone fragments if they do not already exist
5. tests

## BAR Or KI Variant

BAR and KI variants combine:

```text
method + codec + backbone
```

The config class should follow the existing dataclass pattern:

```python
from dataclasses import dataclass, field

from praxis.policies.core.action_tokens import action_tokenizer_config_factory


@dataclass
class MyCodecPaliPolicyConfig(PaliPolicyConfig):
    name: str = "bar_mycodec_pali"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("mycodec")
    )
```

FAST and other variable-length codecs may need a mixin or method-specific config
fields. Keep those fields at the nearest component that owns the behavior.

## FM Variant

FM variants are backbone-only:

```text
fm_pali
fm_qwen
fm_smol
```

Do not add a fake codec axis to FM. It does not use `policy.action_tokenizer`.

## Hydra Alias

A public alias should compose method fragments:

```yaml
defaults:
  - bar/backbone@_here_: pali
  - bar/codec@_here_: mycodec
  - _self_
```

and set:

```yaml
name: bar_mycodec_pali
```

Keep aliases minimal. Behavior belongs in source modules and method fragments.

## Tests

Add tests that cover:

- `policy_name(...)`;
- `supported_policy_names()`;
- `available_policies()`;
- `get_policy_class(...)`;
- Hydra composition for the alias;
- factory construction with fake or minimal `PolicyIO` when practical.

The interface is the public policy name plus the config fields required to construct it.
