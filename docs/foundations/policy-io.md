# PolicyIO

`PolicyIO` is the structural contract shared by datasets, codecs, policies,
checkpoints, serving, and evaluation.

## What It Records

`PolicyIO` records:

| Field | Meaning |
| --- | --- |
| observation features | keys, types, shapes, and normalization modes |
| action feature | key, shape, and normalization mode |
| `n_obs_steps` | number of observation steps consumed by the policy |
| `n_action_steps` | action chunk horizon produced by the policy |

The contract is built from dataset metadata and policy config during training.
It is then saved with checkpoints so loading and evaluation do not have to
guess the feature layout.

## Checkpoint Sidecars

Policy and codec artifacts use sidecars:

```text
policy_io.json
norm_stats.pt
```

`policy_io.json` stores structural metadata. `norm_stats.pt` stores numerical
normalization state.

Token-based policies also require a codec checkpoint:

```text
policy.action_tokenizer.pretrained_path=/path/to/codec/pretrained_model
```

The codec checkpoint is part of the policy definition because action tokens are
only meaningful under the action scale, horizon, vocabulary, and decode rule
used by that codec.

## Failure Modes Caught By The Contract

The contract catches problems that otherwise look like model failures:

| Failure | Why it matters |
| --- | --- |
| missing observation key | policy receives a different context than training |
| action dimension mismatch | checkpoint cannot safely act in the environment |
| missing action normalization | token ids no longer map to the intended action scale |
| stale codec checkpoint | BAR/KI decode through the wrong action language |
| mismatched horizon | rollout consumes a different chunk shape than training |

## Where It Lives

| Responsibility | File |
| --- | --- |
| type definitions | `src/praxis/core/types.py` |
| metadata to `PolicyIO` | `src/praxis/policies/core/io_metadata.py` |
| sidecar filenames and resolution | `src/praxis/core/checkpoint_artifacts.py` |
| pretrained policy loading | `src/praxis/policies/factory.py` |
