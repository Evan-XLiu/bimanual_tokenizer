# Policy Factory

The policy factory is the construction interface shared by training, serving,
and evaluation.

## Main Functions

| Function | Role |
| --- | --- |
| `available_policies()` | list registered policy names |
| `get_policy_class(name)` | resolve a public name to a `BasePolicy` subclass |
| `make_policy(policy_cfg, io=None)` | build from scratch or load from checkpoint |
| `register_policy(name, import_path)` | add an external policy class |

## From Scratch

Fresh construction requires `PolicyIO`:

```python
policy = make_policy(cfg.policy, io=policy_io)
```

BAR and KI construction also requires:

```text
policy.action_tokenizer.pretrained_path
```

The tokenizer checkpoint must include action normalization stats.

## From Pretrained

Pretrained construction loads weights plus sidecars:

```python
policy = make_policy(cfg.policy)
```

where `cfg.policy.pretrained_path` points to a policy `pretrained_model`
directory.

## Source

| Responsibility | File |
| --- | --- |
| public matrix | `src/praxis/policies/catalog.py` |
| factory | `src/praxis/policies/factory.py` |
| base policy | `src/praxis/policies/core/policy.py` |
