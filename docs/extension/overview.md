# Extension Overview

Praxis should grow by extending the axis that owns the new idea. Do not add a
parallel training or evaluation stack when the existing interfaces can express
the change.

## Extension Map

| You are adding | Start here | Owning module |
| --- | --- | --- |
| action representation | [Add A Codec](/extension/add-codec) | `src/praxis/codecs/` |
| VLM family | [Add A Backbone](/extension/add-backbone) | `src/praxis/policies/backbones/` |
| objective family | [Add An Objective](/extension/add-objective) | `src/praxis/policies/` |
| public matrix entry | [Add A Public Cell](/extension/add-public-cell) | `src/praxis/policies/catalog.py` |
| Slurm cluster surface | [Add A Cluster Launcher](/extension/add-cluster-launcher) | `scripts/infra/` |
| benchmark | `praxis-eval` docs | evaluator package |

## Extension Rule

Write the study sentence first:

```text
This extension changes <axis> while preserving <contracts>.
```

Then add the code at the module that owns that axis. Tests should cross the
same interface that users cross.
