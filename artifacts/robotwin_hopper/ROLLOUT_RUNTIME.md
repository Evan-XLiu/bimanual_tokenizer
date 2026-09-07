# RoboTwin Rollout Runtime

## The `libX11.so.6` failure

`run_eval.sh` uses the host RoboTwin interpreter:

```text
/scratch/xuanweiliu/conda_envs/robotwin/bin/python
```

SAPIEN, imported by RoboTwin, dynamically links against the host
`libX11.so.6`.  The Hopper PyTorch Singularity image does not expose that
library.  `--nv` only exposes NVIDIA libraries, and binding `/scratch` does
not bind `/lib64`, so this fails before the first episode:

```text
ImportError: libX11.so.6: cannot open shared object file
```

## Correct rollout command

For RoboTwin rollout PBS jobs, export `ROBOTWIN_ROOT`, `ROBOTWIN_PYTHON`, and
`PRAXIS_CHECKPOINT`, then run directly on the allocated Hopper node:

```bash
bash artifacts/robotwin_hopper/run_eval.sh
```

Do **not** wrap this command in `singularity exec`.  The successful jobs
`601937` and `599336` use this direct-host path; the failed job `603074`
wrapped it in Singularity and stopped at SAPIEN import.

Quick preflight:

```bash
"$ROBOTWIN_PYTHON" -c 'import sapien; print(sapien.__file__)'
```

Training jobs may continue to use Singularity; this rule is specifically for
the RoboTwin rollout/evaluator path.
