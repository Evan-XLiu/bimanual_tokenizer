# Add A Cluster Launcher

Praxis cluster launchers are thin profiles around one shared Submitit launcher
core. Adding support for a new Slurm cluster should usually mean adding one
small entrypoint under `scripts/infra/`, not copying a whole launch stack.

## Shape

```text
scripts/infra/my_cluster_launch.py
  defines partition/account/resource quirks
  builds a SlurmClusterProfile
  calls run_launcher(profile=...)

scripts/infra/launch/
  owns shared argument parsing, command construction, Submitit setup,
  stage orchestration, source checkout behavior, and eval-monitor plumbing
```

The extension sentence should be:

```text
This extension changes the Slurm cluster profile while preserving the shared
Praxis launcher stages and Hydra command construction.
```

## Minimal Entrypoint

Start from an existing launcher such as `scripts/infra/fasrc_launch.py`,
`scripts/infra/delta_launch.py`, `scripts/infra/anvil_launch.py`, or
`scripts/infra/psc_launch.py`.

A cluster entrypoint typically owns:

- supported partition names;
- the account environment variable names;
- default partition selection for tokenizer, policy, eval, and monitor stages;
- partition-specific GRES or constraint handling;
- requeue partition detection;
- optional mail settings.

The shared profile contract is:

```python
from launch.profile import SlurmClusterProfile
from launch.runner import run_launcher

MY_CLUSTER_PROFILE = SlurmClusterProfile(
    validate_partition_list=validate_partition_list,
    resolve_partition_spec=resolve_partition_spec,
    resolve_account_for_partition_spec=resolve_account_for_partition_spec,
    is_requeue_partition_spec=is_requeue_partition_spec,
)


def main() -> None:
    run_launcher(profile=MY_CLUSTER_PROFILE)
```

## Account Handling

Do not hardcode personal account strings. Read accounts from environment
variables and fail early when they are missing.

```python
from launch.partitions import resolve_required_account_env

ACCOUNT_ENV = "PRAXIS_MYCLUSTER_ACCOUNT"


def resolve_account_for_partition_spec(_partition_spec: str) -> str:
    return resolve_required_account_env(
        ACCOUNT_ENV,
        context="MyCluster Slurm account",
    )
```

For clusters where partitions map to different accounts, follow the FASRC
pattern: keep a partition-to-env-var map and resolve the submitted partition
spec through that map.

## Partitions And Resources

Keep cluster-specific behavior local to the cluster entrypoint:

- partition allowlists;
- multipartition restrictions;
- interactive partition limits;
- GPU type names required by `--gres`;
- default partitions for a stage or backbone;
- cluster-specific surface detection, as in `delta_launch.py`.

Keep shared behavior in `scripts/infra/launch/` only when it is truly common to
multiple clusters.

## Test Contract

Add focused tests under `tests/scripts/infra/`.

Cover the behavior users rely on:

- parser accepts expected verbs and rejects invalid combinations;
- `--dry-run` builds the expected Hydra command;
- account env vars are required and mapped correctly;
- partition validation catches unsupported or invalid multipartition requests;
- GRES or constraint translation is correct for the cluster;
- multi-node jobs request one task per node.

Do not freeze mutable operational choices such as active run names, dated output
paths, or one lab's preferred queue in tests.

## Verification

Run direct dry-runs before submitting real jobs:

```bash
uv run python scripts/infra/my_cluster_launch.py \
  train-tokenizer \
  --codec oat \
  --dataset libero \
  --partition gpu \
  --dry-run
```

Then run focused tests:

```bash
uv run python -m pytest tests/scripts/infra -q
uv run python -m pre_commit run --all-files
```

Real training and evaluation jobs should run on compute nodes, not login nodes.
