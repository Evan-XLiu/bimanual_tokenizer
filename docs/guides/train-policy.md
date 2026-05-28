# Train A Policy

Policy training uses:

```bash
uv run praxis-train --cfg job
```

`--cfg job` prints the composed Hydra job config. Drop that flag only when you
are on a compute node and ready to train.

The base policy-training template composes the quickstart example:

```text
policy=bar_oat_pali
dataset=libero
trainer=policy
train_recipe/policy@_global_=bar
```

## BAR Or KI

BAR and KI require an action-tokenizer checkpoint:

```bash
CODEC_CKPT="$(find outputs/train_codecs -path '*/checkpoints/latest/pretrained_model' -type d | sort | tail -n 1)"
test -n "${CODEC_CKPT}"

uv run praxis-train \
  policy=bar_oat_pali \
  policy.action_tokenizer.pretrained_path="${CODEC_CKPT}" \
  dataset.name=libero dataset.repo_id=lerobot/libero \
  device=cuda
```

For KI, select the KI train recipe:

```bash
CODEC_CKPT="$(find outputs/train_codecs -path '*/checkpoints/latest/pretrained_model' -type d | sort | tail -n 1)"
test -n "${CODEC_CKPT}"

uv run praxis-train \
  policy=ki_oat_pali \
  train_recipe/policy@_global_=ki \
  policy.action_tokenizer.pretrained_path="${CODEC_CKPT}" \
  dataset.name=libero dataset.repo_id=lerobot/libero \
  device=cuda
```

## FM

FM does not use an action tokenizer:

```bash
uv run praxis-train \
  policy=fm_pali \
  train_recipe/policy@_global_=fm \
  dataset.name=libero dataset.repo_id=lerobot/libero \
  device=cuda
```

## Output Structure

Policy training writes:

```text
outputs/train_policy/<date>/<run_name>/
  train_config.yaml
  train_metrics.jsonl
  policy_io.json
  checkpoints/
    latest/
      pretrained_model/
    best/
    step_<N>/
  eval/
```

Exact checkpoint retention depends on trainer config.

## Train Recipes

Policy configs should describe model semantics. Train recipes own optimizer scale and scheduler defaults:

```text
src/praxis/configs/train_recipe/policy/bar.yaml
src/praxis/configs/train_recipe/policy/fm.yaml
src/praxis/configs/train_recipe/policy/ki.yaml
```

When comparing methods, be explicit about whether you are using method-native recipes or a controlled shared optimizer ablation.
