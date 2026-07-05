# Configuration

Praxis uses Hydra. Public policy names, codec choices, datasets, train recipes,
and evaluation modes are selected through config composition.

## Policy Training

```bash
uv run praxis-train \
  policy=bar_oat_pali \
  policy.action_tokenizer.pretrained_path=/path/to/codec \
  dataset.name=libero dataset.repo_id=lerobot/libero \
  device=cuda
```

Policy aliases live under:

```text
src/praxis/configs/policy/
```

## Codec Training

```bash
uv run praxis-train --config-name train_codecs \
  codecs@codec=tokens/oat \
  dataset.name=libero dataset.repo_id=lerobot/libero
```

Codec configs live under:

```text
src/praxis/configs/codecs/
```

## Evaluation

```bash
uv run praxis-eval \
  policy=bar_oat_pali \
  policy.pretrained_path=/path/to/policy \
  env.type=libero env.task=libero \
  eval.mode=local_inproc
```

Evaluation config starts at:

```text
src/praxis/configs/eval_policy.yaml
```

## Recipes

Train recipes live under:

```text
src/praxis/configs/train_recipe/
```

Use `--cfg job` before launching expensive jobs.
