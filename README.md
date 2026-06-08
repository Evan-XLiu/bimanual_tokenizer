# Praxis

[Website](https://ordered-action-tokenization.github.io/) | [arXiv](https://arxiv.org/abs/2602.04215) | [Docs](https://chaoqi-liu.com/praxis-vla)| [BibTeX](#citation)

Praxis is a research codebase for training and evaluating
vision-language-action (VLA) manipulation policies. This repository contains
the policy-side implementation: action tokenizers, policy objectives, VLM backbones, training, checkpoint export/load, serving, and the adapter boundary
used by [`praxis-eval`](https://chaoqi-liu.com/praxis-eval/).

## Table of Contents

- [Usage](#usage)
  - [Installation](#installation)
  - [Getting Started](#getting-started)
  - [Command Surface](#command-surface)
- [Policy Cells](#policy-cells)
- [Evaluation Boundary](#evaluation-boundary)
- [External Artifacts](#external-artifacts)
- [Development](#development)
- [License](#license)
- [Citation](#citation)

## Usage

### Installation

Praxis is released as a source research repository, not as a PyPI-first package.
Use a checkout and `uv`:

```bash
git clone https://github.com/Chaoqi-LIU/praxis-vla.git
cd praxis-vla
uv sync --group dev
uv run python -c "import praxis; print('praxis import OK')"
```

Install only the optional extras needed for your workflow. For the quickstart:

```bash
uv sync --group dev --extra pali --extra serving --extra logging --extra libero
```

Common extras include `pali`, `qwen`, `smol`, `serving`, `logging`, `libero`,
`metaworld`, `mshab`, `robocasa`, `robomimic`, `simpler`, and `slurm`.

### Getting Started

The maintained documentation is the source of truth for setup, configuration,
training, evaluation, and extension:

https://chaoqi-liu.com/praxis-vla

For a compact end-to-end smoke path, see
[BAR OAT Pali Path](docs/workflows/bar-oat-pali.md). It trains an OAT codec,
trains a BAR policy with a PaliGemma2 backbone, and evaluates the resulting
checkpoint through `praxis-eval`.

Inspect the composed Hydra config before launching real training jobs:

```bash
uv run praxis-train policy=bar_oat_pali --cfg job
```

Training and simulator evaluation are GPU/simulator workloads. On shared
clusters, run them inside an interactive or batch compute allocation.

### Command Surface

```bash
uv run praxis-train --help
uv run praxis-eval --help
uv run praxis-serve --help
uv run praxis-eval-monitor --help
```

Typical forms:

```bash
uv run praxis-train --config-name train_codecs codecs@codec=tokens/oat
uv run praxis-train policy=bar_oat_pali policy.action_tokenizer.pretrained_path=/path/to/codec
uv run praxis-eval policy=bar_oat_pali policy.pretrained_path=/path/to/policy env.type=libero env.task=libero
```

## Policy Cells

Praxis treats a robot policy as a cell in a design matrix:

```text
policy = objective + action representation + VLM backbone
```

One runnable cell is:

```text
bar_oat_pali = BAR objective + OAT codec + PaliGemma2 backbone
```

`bar_oat_pali` is the compact quickstart example, not a claimed best method.
Useful comparisons change one coordinate at a time:

```text
bar_oat_pali  vs  bar_oat_qwen      # backbone comparison
bar_oat_pali  vs  bar_fast_pali     # action-representation comparison
bar_oat_pali  vs  ki_oat_pali       # objective comparison
```

| Axis | Implemented choices |
| --- | --- |
| Policy objectives | BAR, FM, KI |
| Action representations | Binning, FAST, OAT, BOAT, QueST, ActionCodec |
| VLM backbones | PaliGemma2, Qwen3-VL, SmolVLM |
| Runtime surfaces | training, validation, checkpoint export/load, serving, `praxis-eval` adapter |

## Evaluation Boundary

Closed-loop benchmark execution is delegated to `praxis-eval`.

Praxis owns policy construction, codec training, checkpoint loading,
preprocessing, serving, and `PolicyEvalAdapter`. `praxis-eval` owns benchmark
setup, rollout execution, metrics, artifacts, simulator dependencies, and
benchmark-specific observation/action contracts.

```text
Praxis checkpoint -> PolicyEvalAdapter -> praxis-eval rollout -> metrics/media
```

## External Artifacts

External Hugging Face datasets, model checkpoints, simulator assets, benchmark
packages, and any released weights are governed by their upstream model cards,
dataset cards, licenses, and access terms. This repository's Apache-2.0 license
covers the Praxis source code, not those external artifacts.

## Development

Local documentation entry points:

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

Run checks from the repo root:

```bash
uv run python -m pytest tests/ -x
uv run python -m pre_commit run --all-files
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for project boundaries, development
setup, and license-header guidance.

## License

Praxis source code is licensed under Apache-2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).

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
