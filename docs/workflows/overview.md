# Workflows Overview

Praxis workflows follow the artifact chain:

```text
dataset metadata
  -> PolicyIO and normalization
  -> optional codec checkpoint
  -> policy checkpoint
  -> local or remote evaluation
```

## Main Workflows

| Workflow | Use it when |
| --- | --- |
| [Train A Codec](/workflows/train-codec) | training BAR or KI with a token action representation |
| [Train A Policy](/workflows/train-policy) | producing a policy checkpoint from a public cell |
| [Evaluate](/workflows/evaluate) | running local or remote rollout through `praxis-eval` |
| [Serve](/workflows/serve) | isolating policy runtime from simulator runtime |
| [Slurm Launchers](/workflows/slurm-launchers) | submitting tokenizer, policy, eval, and monitor jobs on research clusters |
| [BAR OAT Pali Path](/workflows/bar-oat-pali) | checking the full quickstart-style path |

## Rules Of Thumb

- BAR and KI need `policy.action_tokenizer.pretrained_path`.
- FM does not use a token codec.
- Use `--cfg job` to inspect Hydra composition before launching heavy jobs.
- Use launcher `--dry-run` before submitting Slurm jobs.
- Keep benchmark rollout details in `praxis-eval`; keep policy construction in
  `praxis-vla`.
