# Praxis

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

Common extras include `pali`, `qwen`, `smol`, `molmo`, `serving`, `logging`, `libero`,
`metaworld`, `mshab`, `robocasa`, `robomimic`, `simpler`, and `slurm`.

### Hopper / BiCAT Molmo2-ER Environment

The current training environment uses Python 3.10.21, PyTorch 2.7.1,
torchvision 0.22.1, Transformers 4.57.6, LeRobot 0.4.4, datasets 4.1.1,
NumPy 2.2.5, PEFT 0.18.1, PyAV 15.1.0, and wandb 0.24.2. The commands below
create a matching Python environment in a fresh checkout:

```bash
git clone https://github.com/Evan-XLiu/bimanual_tokenizer.git
cd bimanual_tokenizer
git switch real_world
uv python install 3.10.21
uv sync --python 3.10.21 --group dev --extra molmo --extra pali --extra logging
uv pip install --python .venv/bin/python \
  'torch==2.7.1' 'torchvision==0.22.1' 'transformers==4.57.6' \
  'lerobot==0.4.4' 'datasets==4.1.1' 'numpy==2.2.5' \
  'peft==0.18.1' 'av==15.1.0' 'wandb==0.24.2'
```

These pins record the installed core versions, rather than a complete lock of
every transitive dependency. A later `uv sync` can replace these overrides;
the cluster scripts invoke `.venv/bin/python` directly.

Hopper jobs use the cluster-provided Singularity image for the CUDA runtime,
together with the repository's `.venv`. Run the following check inside a GPU
compute allocation:

```bash
module load singularity/1.3.1
singularity exec --nv --bind /scratch:/scratch \
  /app1/common/singularity-img/hopper/pytorch/pytorch_2.7.0_cuda_12.8.sif \
  .venv/bin/python -c "import torch, transformers, praxis; print(torch.__version__, torch.version.cuda, transformers.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The image name contains PyTorch 2.7.0; the Python environment currently supplies
PyTorch 2.7.1. The image path and PBS project are specific to this cluster.

Prepare the model cache from a machine with network access before launching
offline jobs:

```bash
export HF_HOME="$PWD/.cache/huggingface"
export HF_MODULES_CACHE="$HF_HOME/modules"
.venv/bin/hf download allenai/Molmo2-ER --local-dir .cache/models/Molmo2-ER
```

For PaliGemma, authenticate with `.venv/bin/hf auth login` using an account with
access to the model, then cache it with
`.venv/bin/hf download google/paligemma2-3b-mix-224`. Molmo loads its native
processor/model code with `trust_remote_code=True`.

Edit [the BiCAT training config](artifacts/bicat_box/ki_molmo2er_fft.yaml) for
your filesystem:

- `policy.vlm_model_name`: local Molmo2-ER model directory.
- `dataset.root`: LeRobot dataset directory containing `meta/info.json`, data,
  and videos. Its default is `${BICAT_DATA_ROOT}/box`, with a local fallback.
- `policy.action_tokenizer.pretrained_path`: trained BTT codec's
  `pretrained_model` directory, including its weights and normalization state.
- `trainer.wandb_project`: logging destination; authenticate with
  `.venv/bin/wandb login`, or set `trainer.use_wandb: false`.

The launcher sets `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`,
`OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. It also creates a per-job
temporary directory under `.t/`. Model and dataset files must already be local.

Submit from the repository root:

```bash
qsub artifacts/bicat_box/test_ki_molmo2er_memory.pbs
qsub artifacts/bicat_box/train_ki_molmo2er_fft.pbs
```

The training PBS script requests one H200, 12 CPUs, and 225 GB host memory.
It runs [run_ki_molmo2er.sh](artifacts/bicat_box/run_ki_molmo2er.sh) inside the
container and writes training outputs under
`outputs/BiCAT_box_molmo2ER_train/<PBS_JOBID>/`. The commands above submit jobs;
use the memory test to check the chosen batch size before a full training run.

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
@misc{liu2026praxis,
      title={Praxis: A Controlled Laboratory for Vision-Language-Action Policy Research},
      author={Chaoqi Liu},
      year={2026},
      url={https://ordered-action-tokenization.github.io/praxis},
}
```

## License

Praxis source code is released under the Apache-2.0 license. See
[LICENSE](LICENSE) and [NOTICE](NOTICE).
