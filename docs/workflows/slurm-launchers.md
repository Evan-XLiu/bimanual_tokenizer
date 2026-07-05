# Slurm Launchers

Praxis includes a small Submitit-based launcher layer for research clusters.
The public launcher surface is `scripts/infra/`: thin cluster entrypoints call a
shared implementation under `scripts/infra/launch/`.

```text
scripts/infra/<cluster>_launch.py
  -> scripts/infra/launch/runner.py
  -> scripts/infra/launch/stages.py
  -> praxis-train / praxis-eval / praxis-eval-monitor
```

This is meant for source checkouts. It does not replace reading the composed
Hydra config, and it does not own benchmark runtime behavior.

## Install

Install the Slurm extra:

```bash
uv sync --group dev --extra slurm
```

Cluster launchers require an account environment variable instead of a
hardcoded account string:

| Launcher | Account variables |
| --- | --- |
| `scripts/infra/fasrc_launch.py` | `PRAXIS_FASRC_ACCOUNT`, `PRAXIS_FASRC_KEMPNER_ACCOUNT` |
| `scripts/infra/delta_launch.py` | `PRAXIS_DELTA_ACCOUNT`, `PRAXIS_DELTAAI_ACCOUNT` |
| `scripts/infra/anvil_launch.py` | `PRAXIS_ANVIL_ACCOUNT` |
| `scripts/infra/psc_launch.py` | `PRAXIS_PSC_ACCOUNT` |

Example:

```bash
export PRAXIS_FASRC_ACCOUNT=your_slurm_account
export PRAXIS_FASRC_KEMPNER_ACCOUNT=your_kempner_account
```

## Dry Run First

Use `--dry-run` to inspect generated commands and Slurm parameters before
submitting:

```bash
uv run python scripts/infra/fasrc_launch.py \
  train-tokenizer \
  --codec oat \
  --dataset libero \
  --partition kempner \
  --dry-run
```

## Train A Codec

```bash
uv run python scripts/infra/fasrc_launch.py \
  train-tokenizer \
  --codec oat \
  --dataset libero \
  --partition kempner
```

For deterministic bootstrap codecs:

```bash
uv run python scripts/infra/fasrc_launch.py \
  train-tokenizer \
  --codec fast \
  --dataset libero \
  --skip-fit \
  --partition kempner \
  --dry-run
```

## Train A Policy

Codec-backed methods such as BAR and KI require an action tokenizer checkpoint:

```bash
CODEC_CKPT=outputs/train_codecs/example/checkpoints/latest/pretrained_model

uv run python scripts/infra/fasrc_launch.py \
  train-policy \
  --method bar \
  --codec oat \
  --backbone pali \
  --dataset libero \
  --tokenizer-ckpt "${CODEC_CKPT}" \
  --partition kempner_h100
```

Pass extra Hydra overrides after `--override`:

```bash
CODEC_CKPT=outputs/train_codecs/example/checkpoints/latest/pretrained_model

uv run python scripts/infra/fasrc_launch.py \
  train-policy \
  --method bar \
  --codec oat \
  --backbone pali \
  --dataset libero \
  --tokenizer-ckpt "${CODEC_CKPT}" \
  --override trainer.max_steps=1000 trainer.use_wandb=false \
  --dry-run
```

## Evaluate

Submit one checkpoint evaluation job:

```bash
POLICY_CKPT=outputs/train_policy/example/checkpoints/latest/pretrained_model

uv run python scripts/infra/fasrc_launch.py \
  eval \
  --policy-ckpt "${POLICY_CKPT}" \
  --eval-output-dir outputs/eval/libero_bar_oat_pali \
  --partition kempner
```

Forward evaluation-specific Hydra overrides with repeated `--eval-override`
flags:

```bash
POLICY_CKPT=outputs/train_policy/example/checkpoints/latest/pretrained_model

uv run python scripts/infra/fasrc_launch.py \
  eval \
  --policy-ckpt "${POLICY_CKPT}" \
  --eval-output-dir outputs/eval/libero_bar_oat_pali \
  --eval-override env.type=libero env.task=libero \
  --eval-override eval.num_eval_per_task=10 eval.num_parallel_env=10 \
  --dry-run
```

## Monitor Checkpoints

`eval-monitor` watches a training output directory and launches evaluations for
eligible checkpoints:

```bash
TRAIN_DIR=outputs/train_policy/example
TRAIN_JOB_ID=123456

uv run python scripts/infra/fasrc_launch.py \
  eval-monitor \
  --train-dir "${TRAIN_DIR}" \
  --train-job-id "${TRAIN_JOB_ID}" \
  --partition kempner
```

Useful controls:

```text
--stale-timeout-sec <seconds>   # stop if the trainer appears idle
--eval-timeout-sec <seconds>    # kill one evaluator subprocess after a timeout
--num-eval-per-task <n>         # episodes per task
--num-parallel-env <n>          # parallel environments per eval
```

## Runtime And Source Checkout

By default, submitted jobs use `uv run` from the current source checkout. For a
specialized environment, pass a Python binary:

```bash
CODEC_CKPT=outputs/train_codecs/example/checkpoints/latest/pretrained_model

uv run python scripts/infra/fasrc_launch.py \
  train-policy \
  --method bar \
  --codec oat \
  --backbone pali \
  --dataset libero \
  --tokenizer-ckpt "${CODEC_CKPT}" \
  --python /path/to/env/bin/python \
  --dry-run
```

To run a specific git ref through an ephemeral worktree:

```bash
CODEC_CKPT=outputs/train_codecs/example/checkpoints/latest/pretrained_model

uv run python scripts/infra/fasrc_launch.py \
  train-policy \
  --method bar \
  --codec oat \
  --backbone pali \
  --dataset libero \
  --tokenizer-ckpt "${CODEC_CKPT}" \
  --git-ref main \
  --dry-run
```

## Boundary

The launchers own cluster submission mechanics: Slurm partitions, accounts,
resources, source checkout handling, and command construction. They do not own
policy semantics, benchmark rollout behavior, simulator wrappers, or dataset
conversion.

Use [`praxis-eval`](https://github.com/Chaoqi-LIU/praxis-eval) for benchmark
setup and rollout contracts.
