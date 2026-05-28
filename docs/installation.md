# Installation

Praxis is a Python package with optional extras for serving, backbone-specific dependencies, logging, and benchmark runtimes.

## Base Environment

Recommended:

- Linux
- Python 3.10 through 3.13
- `uv`
- CUDA-capable GPU for real training

Install the development environment:

```bash
uv sync --group dev
```

Install all extras you need in one `uv sync` command. For example, a common
LIBERO plus PaliGemma serving environment is:

```bash
uv sync --group dev --extra pali --extra serving --extra logging --extra libero
```

For a broader local development environment, include the other backbone extras
in the same command:

```bash
uv sync --group dev --extra pali --extra qwen --extra smol --extra serving --extra logging --extra libero
```

`uv sync` makes the environment match the selected extras, so do not run a
second sync with fewer extras unless you intend to remove the earlier ones.

`praxis-eval` owns simulator-specific packages and setup commands. Praxis depends on `praxis-eval`, but it should not grow benchmark environment wrappers inside `src/praxis`.

## Compute Nodes

Training and simulator evaluation are GPU jobs. Use `--cfg job` on a login node
to inspect Hydra composition, and run real jobs inside an interactive or batch
compute allocation.

## Optional Extras

| Extra | Purpose |
| --- | --- |
| `pali` | PaliGemma / PaliGemma2 LoRA dependencies. |
| `qwen` | Qwen3-VL utilities and LoRA dependencies. |
| `smol` | SmolVLM preprocessing dependencies and LoRA dependencies. |
| `serving` | gRPC server dependencies. |
| `logging` | Weights & Biases logging. |
| `libero`, `robocasa`, `robomimic`, `metaworld`, `simpler`, `mshab` | Benchmark extras forwarded to `praxis-eval`. |

## Developer Checks

Run Python checks from the repo root:

```bash
uv run python -m pytest tests/ -x
uv run python -m basedpyright src/praxis scripts
uv run python -m pre_commit run --all-files
```

Run the cheap docs content check from the repo root:

```bash
uv run python docs/scripts/check_snippets.py
```

Build the VitePress site from `docs/` when Node/npm is working:

```bash
cd docs
npm install
npm run docs:build
```

The docs use VitePress and MathJax for equations. On clusters where the system
`npm` hangs, run the VitePress build in a Node environment where `npm --version`
returns promptly.
