# Installation

Praxis is distributed as a source research repository. Use a checkout and `uv`;
do not expect a PyPI-first workflow.

## Base Environment

Recommended:

- Linux for real training and simulator evaluation;
- Python 3.10 through 3.13;
- CUDA-capable GPU for training;
- `uv` for dependency management.

Install the development environment:

```bash
uv sync --group dev
```

Verify the package:

```bash
uv run python -c "import praxis; print('praxis import OK')"
```

## Extras

Install the extras for the surfaces you need. For the quickstart:

```bash
uv sync --group dev --extra pali --extra serving --extra logging --extra libero
```

For broader local development:

```bash
uv sync --group dev --extra pali --extra qwen --extra smol --extra serving --extra logging --extra libero
```

`uv sync` makes the environment match the selected extras. A later sync with
fewer extras can remove packages installed by an earlier command.

| Extra | Purpose |
| --- | --- |
| `pali` | PaliGemma / PaliGemma2 dependencies. |
| `qwen` | Qwen3-VL utilities. |
| `smol` | SmolVLM preprocessing dependencies. |
| `serving` | gRPC serving dependencies. |
| `logging` | Weights & Biases logging. |
| `libero`, `robocasa`, `robomimic`, `metaworld`, `simpler`, `mshab` | Benchmark extras forwarded to `praxis-eval`. |

## Compute Nodes

Training and simulator evaluation are GPU workloads. On shared clusters, use
Hydra inspection on login nodes:

```bash
uv run praxis-train policy=bar_oat_pali --cfg job
```

Run real jobs in an interactive or batch compute allocation.

## Developer Checks

Run Python checks from the repo root:

```bash
uv run python -m pytest tests/ -x
uv run python -m basedpyright src/praxis scripts
uv run python -m pre_commit run --all-files
```

Run docs checks:

```bash
uv run python docs/scripts/check_snippets.py
```

Build the VitePress site:

```bash
cd docs
npm ci --no-audit --no-fund
npm run docs:build
```

External datasets, checkpoints, simulator assets, and model weights are governed
by their upstream cards, licenses, and access terms rather than by this
repository's Apache-2.0 license.

The docs package currently requires a Node version supported by Vite. If the
system `node` is broken or too old, use a working Node 20 or newer environment.
