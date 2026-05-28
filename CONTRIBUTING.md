# Contributing

## Development Setup

```bash
git clone https://github.com/Chaoqi-LIU/praxis-vla.git
cd praxis-vla
uv sync --group dev
uv run pre-commit install
uv run pytest --strict-markers -m "not manual"
```

Install only the optional extras needed for the work you are doing, for example:

```bash
uv sync --group dev --extra pali --extra serving --extra logging --extra libero
```

For optional Slurm launcher helpers under `scripts/infra/`, also install
`--extra slurm` and set the relevant account environment variable for your
cluster, for example `PRAXIS_FASRC_ACCOUNT`, `PRAXIS_DELTAAI_ACCOUNT`,
`PRAXIS_DELTA_ACCOUNT`, `PRAXIS_ANVIL_ACCOUNT`, or `PRAXIS_PSC_ACCOUNT`.

## Project Boundary

Praxis owns policy training, action codecs, checkpoint loading, policy
preprocessing, serving adapters, and source-checkout research workflows.

`praxis-eval` owns benchmark setup, rollout execution, metrics, artifacts,
setup/verification tools, and observation/action contracts. Keep simulator- and
benchmark-specific behavior in `praxis-eval` unless the code is genuinely part
of policy training or serving.

Keep method-specific behavior in the nearest honest owner:

- codec behavior belongs in `src/praxis/codecs/`;
- BAR/KI/FM policy behavior belongs in `src/praxis/policies/`;
- trainer lifecycle and metric aggregation belong in `src/praxis/training/`;
- benchmark rollout behavior belongs in `praxis-eval`.

## License Headers

New manually maintained original Python files under `src/` should carry SPDX
headers:

```text
SPDX-FileCopyrightText: 2026 Chaoqi Liu and contributors
SPDX-License-Identifier: Apache-2.0
```

Use the current year when creating a new file. Do not update every file just
because the calendar year changed. If an existing file receives copyrightable
changes in a later year, update only that file's year range or copyright holder
when appropriate.

Do not remove third-party copyright notices or file-level license headers.
Files with existing non-Apache headers remain governed by those headers.

Tests, documentation, workflow files, project config, generated files, lockfiles,
and files that should not be edited directly are licensed through the root
`LICENSE` and `NOTICE` files, but they should not carry visible SPDX headers.

## Checks

Run the relevant subset locally before sending changes:

```bash
uv run pre-commit run --all-files
GIT_LFS_SKIP_SMUDGE=true uv run pytest --strict-markers -m "not manual"
```

For documentation changes, also run:

```bash
cd docs
npm ci --no-audit --no-fund
npm run docs:check
npm run docs:build
```
