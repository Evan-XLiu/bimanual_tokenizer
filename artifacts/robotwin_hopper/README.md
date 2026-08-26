# RoboTwin evaluation on NUS Hopper

This replaces the repository's former site-specific launch artifacts with one
portable PBS Pro job for NUS Hopper. The job runs one Praxis policy
server and one RoboTwin simulator worker on a single allocated H100 by default.

Before submission, provide paths to an existing RoboTwin checkout, its Python
environment, and an exported Praxis checkpoint:

```bash
cd /scratch/xuanweiliu/bimanual_tokenizer

export PRAXIS_HOPPER_PROJECT=CFP01-CF-XXX
export ROBOTWIN_ROOT=/scratch/xuanweiliu/RoboTwin
export ROBOTWIN_PYTHON=/scratch/xuanweiliu/robotwin-env/bin/python
export PRAXIS_CHECKPOINT=/scratch/xuanweiliu/checkpoints/robotwin/pretrained_model

qsub -P "$PRAXIS_HOPPER_PROJECT" \
  -v ROBOTWIN_ROOT,ROBOTWIN_PYTHON,PRAXIS_CHECKPOINT \
  artifacts/robotwin_hopper/job.pbs
```

`PRAXIS_CHECKPOINT` must contain `config.json` and `model.safetensors`.
`ROBOTWIN_PYTHON` must contain RoboTwin's simulator dependencies and
`praxis-remote`. The policy server uses this repository's `.venv`.

Optional environment variables are `ROBOTWIN_RUN_ID`, `ROBOTWIN_OUTPUT_DIR`,
`EPISODES_PER_TASK`, `MAX_SEED_ATTEMPTS`, `POLICY_PORT`, and
`HOPPER_SINGULARITY_IMAGE`. Results default to
`/scratch/xuanweiliu/bimanual_tokenizer/outputs/robotwin2_clean/<run-id>`.

Use `hpc project` after `amgr login` to find the project value. Hopper selects
the queue from the requested resources, so the PBS file intentionally does not
name a queue.
