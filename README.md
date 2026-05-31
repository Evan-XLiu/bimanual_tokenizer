# Praxis

[![Docs](https://github.com/Chaoqi-LIU/praxis-vla/actions/workflows/docs.yml/badge.svg)](https://github.com/Chaoqi-LIU/praxis-vla/actions/workflows/docs.yml)
[![Tests](https://github.com/Chaoqi-LIU/praxis-vla/actions/workflows/test.yml/badge.svg)](https://github.com/Chaoqi-LIU/praxis-vla/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Praxis is a controlled laboratory for vision-language-action (VLA) policy
research. This repository, `praxis-vla`, is the policy-side implementation:
policy objectives, action representations, VLM backbones, training,
checkpointing, serving, and the adapter used by `praxis-eval`.

The problem Praxis addresses is that VLA rollout scores often compare whole
private pipelines rather than one declared design choice. Objective, tokenizer,
backbone, preprocessing, normalization, serving, simulator dependencies, and
benchmark adapters can all change at once. Praxis makes the intended variables
explicit by treating a robot policy as a cell in a design matrix:

```text
policy = objective + action representation + VLM backbone
```

One runnable cell is:

```text
bar_oat_pali = BAR objective + OAT codec + PaliGemma2 backbone
```

`bar_oat_pali` is a compact quickstart path, not a preferred method.

## What Is Included

| Surface | Implemented choices |
| --- | --- |
| Policy objectives | BAR, FM, KI |
| Action representations | Binning, FAST, OAT, BOAT, QueST, ActionCodec |
| VLM backbones | PaliGemma2, Qwen3-VL, SmolVLM |
| Runtime surfaces | training, validation, checkpoint export/load, serving, `praxis-eval` adapter |

Useful comparisons look like:

```text
bar_oat_pali  vs  bar_oat_qwen      # backbone comparison
bar_oat_pali  vs  bar_fast_pali     # action-representation comparison
bar_oat_pali  vs  ki_oat_pali       # objective comparison
```

## Documentation

The maintained documentation is the source of truth for setup, configuration,
training, evaluation, and extension:

https://chaoqi-liu.com/praxis-vla

Local entry points:

- [Introduction](docs/introduction.md): problem statement and repo model.
- [Foundations](docs/foundations/overview.md): policy cells, `PolicyIO`,
  evaluation boundary, and source layout.
- [Design Axes](docs/axes/overview.md): objectives, action representations,
  backbones, and comparison design.
- [Workflows](docs/workflows/overview.md): train codecs, train policies,
  evaluate, and serve.
- [Extension](docs/extension/overview.md): add codecs, backbones, objectives,
  and public cells.
- [Reference](docs/reference/cli.md): CLI, config, artifacts, factories, and
  docs maintenance.

## Install From Source

Praxis is released as a source research repository, not as a PyPI-first
package. Use a checkout and `uv`.

```bash
git clone https://github.com/Chaoqi-LIU/praxis-vla.git
cd praxis-vla
uv sync --group dev
uv run python -c "import praxis; print('praxis import OK')"
uv run python -m pytest tests/ -x
```

Install only the optional extras needed for the workflow:

```bash
uv sync --group dev --extra pali --extra serving --extra logging --extra libero
```

Common extras include `pali`, `qwen`, `smol`, `serving`, `logging`, `libero`,
`metaworld`, `mshab`, `robocasa`, `robomimic`, `simpler`, and `slurm`.

## Command Surface

```bash
uv run praxis-train --help
uv run praxis-eval --help
uv run praxis-serve --help
uv run praxis-eval-monitor --help
```

Training and benchmark rollout are GPU/simulator workloads. Use the
configuration docs before running them on shared infrastructure.

## Evaluation Boundary

Closed-loop benchmark execution is delegated to
[`praxis-eval`](https://github.com/Chaoqi-LIU/praxis-eval).

Praxis owns policy construction, codec training, checkpoint loading,
preprocessing, serving, and `PolicyEvalAdapter`. `praxis-eval` owns benchmark
setup, rollout execution, metrics, artifacts, simulator dependencies, and
benchmark-specific observation/action contracts.

```text
Praxis checkpoint -> PolicyEvalAdapter -> praxis-eval rollout -> metrics/media
```

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

Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, project
boundaries, checks, and license-header guidance.

## License

Unless otherwise noted in file headers, Praxis is licensed under Apache-2.0. See
[LICENSE](LICENSE) and [NOTICE](NOTICE).

Some source files retain third-party non-commercial license headers. Those files
must be used according to their own notices.
