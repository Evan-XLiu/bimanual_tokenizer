# Praxis

Official implementation of **OAT: Ordered Action Tokenization**.

[Webpage](https://ordered-action-tokenization.github.io/) |
[arXiv](https://arxiv.org/abs/2602.04215) |
[Docs](https://chaoqi-liu.com/praxis-vla)

Praxis is a research codebase for training and evaluating
vision-language-action policies with explicit action representations. It
contains action tokenizers/codecs, policy objectives, VLM backbone adapters,
training and checkpointing code, serving utilities, and the policy adapter used
by [`praxis-eval`](https://github.com/Chaoqi-LIU/praxis-eval).

## Installation

Praxis is released as a source research repository. Use a checkout and `uv`.

```bash
git clone https://github.com/Chaoqi-LIU/praxis-vla.git
cd praxis-vla
uv sync --group dev
uv run python -c "import praxis; print('praxis import OK')"
```

Install only the extras needed for your workflow. For the quickstart:

```bash
uv sync --group dev --extra libero --extra pali --extra serving --extra logging
```

Common extras include `pali`, `qwen`, `smol`, `serving`, `logging`, `libero`,
`metaworld`, `mshab`, `robocasa`, `robomimic`, `simpler`, and `slurm`.

## Quickstart

The maintained docs are the source of truth:

https://chaoqi-liu.com/praxis-vla

Start with:

- [Installation](docs/installation.md)
- [Quickstart](docs/quickstart.md)
- [BAR OAT Pali Path](docs/workflows/bar-oat-pali.md)
- [CLI Reference](docs/reference/cli.md)

The compact smoke path uses `bar_oat_pali`:

```text
BAR objective + OAT action codec + PaliGemma2 backbone
```

Inspect the composed Hydra config before launching real jobs:

```bash
uv run praxis-train policy=bar_oat_pali --cfg job
```

Training and simulator evaluation are GPU workloads. On shared clusters, run
them on compute nodes rather than login nodes.

## What's Included

| Area | Contents |
| --- | --- |
| Action representations | Binning, FAST, OAT, BOAT, QueST, ActionCodec |
| Policy objectives | BAR, FM, KI |
| VLM backbones | PaliGemma2, Qwen3-VL, SmolVLM |
| Runtime | training, validation, checkpoint export/load, serving |
| Evaluation bridge | `PolicyEvalAdapter` for `praxis-eval` |
| Cluster utilities | Submitit/Slurm launchers under `scripts/infra` |

Closed-loop benchmark execution lives in `praxis-eval`. Praxis owns policy
construction, codec training, checkpoint loading, preprocessing, and serving;
`praxis-eval` owns benchmark setup, rollout execution, metrics, artifacts, and
simulator dependencies.

## Repository Layout

```text
src/praxis/codecs/      action tokenizers and latent/action codecs
src/praxis/policies/    policy families, VLM adapters, and policy registry
src/praxis/training/    training loops, checkpointing, metrics, and eval hooks
src/praxis/serving/     remote policy server runtime
scripts/infra/          extensible Submitit/Slurm launchers
docs/                   technical documentation
tests/                  regression tests
```

For Slurm usage, see [Slurm Launchers](docs/workflows/slurm-launchers.md). To
port the launcher to another cluster, see
[Add A Cluster Launcher](docs/extension/add-cluster-launcher.md).

## External Artifacts

External datasets, model checkpoints, simulator assets, benchmark packages, and
released weights are governed by their upstream model cards, dataset cards,
licenses, and access terms. This repository's Apache-2.0 license covers the
Praxis source code, not those external artifacts.

## Citation

If Praxis or OAT supports your research, please cite:

```bibtex
@misc{liu2026oatorderedactiontokenization,
      title={OAT: Ordered Action Tokenization},
      author={Chaoqi Liu and Xiaoshen Han and Jiawei Gao and Yue Zhao and Haonan Chen and Yilun Du},
      year={2026},
      eprint={2602.04215},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.04215},
}
```

## License

Praxis source code is released under the Apache-2.0 license. See
[LICENSE](LICENSE) and [NOTICE](NOTICE).
