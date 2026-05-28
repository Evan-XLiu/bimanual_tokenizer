# Policy Factory

The policy factory lives at:

```text
src/praxis/policies/factory.py
```

It owns:

- policy class registry;
- `available_policies()`;
- `get_policy_class(name)`;
- `make_policy(policy_cfg, io=None)`;
- loading pretrained policy checkpoints;
- validating action-tokenizer checkpoint sidecars.

## From Scratch

Fresh policy construction requires `PolicyIO`:

```python
from praxis.policies.factory import make_policy

policy = make_policy(cfg.policy, io=policy_io)
```

BAR and KI fresh construction also require:

```text
policy.action_tokenizer.pretrained_path
```

## From Pretrained

Checkpoint construction loads the self-contained checkpoint:

```python
policy = make_policy(cfg.policy)
```

where:

```text
cfg.policy.pretrained_path = /absolute/path/to/policy/pretrained_model
```

The checkpoint must include:

```text
policy_io.json
norm_stats.pt
```

## Registry Source

The default registry is generated from:

```text
src/praxis/policies/catalog.py
```

Public names should resolve through the catalog rather than one-off factory registration.
