# Praxis

[![Docs](https://github.com/Chaoqi-LIU/praxis/actions/workflows/docs.yml/badge.svg)](https://github.com/Chaoqi-LIU/praxis/actions/workflows/docs.yml)
[![Tests](https://github.com/Chaoqi-LIU/praxis/actions/workflows/test.yml/badge.svg)](https://github.com/Chaoqi-LIU/praxis/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Praxis is a source research codebase for controlled comparison of
vision-language-action policies. It treats a robot policy as a cell in a design
matrix:

```text
policy = method + action representation + VLM backbone
```

One quickstart example is:

```text
bar_oat_pali = BAR objective + OAT codec + PaliGemma2 backbone
```

The point of the codebase is not to hide this matrix behind one recipe. Praxis
keeps the axes explicit so researchers can change one component at a time and
compare the result under the same training, checkpointing, serving, and
evaluation surface. `bar_oat_pali` is used in the quickstart because it is a
compact runnable example, not because it is a preferred method.

## What Is Included

| Axis | Implemented choices |
| --- | --- |
| Policy objectives | BAR, FM, KI |
| Action representations | Binning, FAST, OAT, BOAT, QueST, ActionCodec |
| VLM backbones | PaliGemma2, Qwen3-VL, SmolVLM |
| Runtime surfaces | training, validation, checkpoint export/load, serving, `praxis-eval` adapter |

Useful comparisons look like:

```text
bar_oat_pali  vs  bar_oat_qwen      # backbone comparison
bar_oat_pali  vs  bar_fast_pali     # action-codec comparison
bar_oat_pali  vs  ki_oat_pali       # policy-objective comparison
```

## Documentation

The maintained documentation is the source of truth for setup, configuration,
training, evaluation, and extension:

https://chaoqi-liu.com/praxis

Good entry points:

- [Introduction](docs/introduction.md): project model and reading path.
- [Quickstart](docs/quickstart.md): minimal BAR OAT Pali training/eval path.
- [Comparison Axes](docs/concepts/comparison-axes.md): how to run controlled
  method, codec, and backbone comparisons.
- [PolicyIO Contract](docs/concepts/policy-io.md): dataset/checkpoint/eval I/O
  contract.
- [Adding Codecs](docs/development/adding-codecs.md), [Adding Backbones](docs/development/adding-backbones.md),
  and [Adding Policy Variants](docs/development/adding-policy-variants.md):
  extension guides.

## Install From Source

Praxis is released as a source research repository, not as a PyPI package. Use a
checkout and `uv`.

```bash
git clone https://github.com/Chaoqi-LIU/praxis.git
cd praxis
uv sync --group dev
uv run python -c "import praxis; print('praxis import OK')"
uv run python -m pytest tests/ -x
```

Install only the optional extras needed for the workflow:

```bash
uv sync --group dev --extra pali --extra serving --extra logging --extra libero
```

Common extras include `pali`, `qwen`, `smol`, `serving`, `logging`, `libero`,
`metaworld`, `mshab`, `robocasa`, `robomimic`, and `simpler`.

## Command Surface

The installable entrypoints are:

```bash
uv run praxis-train --help
uv run praxis-eval --help
uv run praxis-serve --help
uv run praxis-eval-monitor --help
```

Training and benchmark rollout are GPU/simulator workloads. Use the
configuration docs before running them on shared infrastructure.

## Repository Map

| Path | Responsibility |
| --- | --- |
| `src/praxis/codecs/` | action codecs, tokenizers, latent autoencoders, codec factory |
| `src/praxis/policies/` | BAR, FM, KI policies, backbone adapters, policy catalog/factory |
| `src/praxis/training/` | trainers, checkpointing, distributed runtime, metrics, inline eval |
| `src/praxis/configs/` | Hydra configs for policies, codecs, datasets, trainers, recipes |
| `src/praxis/serving/` | policy server runtime and remote inference integration |
| `scripts/tokenizer/` | tokenizer diagnostics and reconstruction analysis utilities |
| `docs/` | VitePress documentation source |

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
