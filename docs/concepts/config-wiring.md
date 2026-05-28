# Configuration

Praxis uses Hydra composition. The active config root is:

```text
src/praxis/configs/
```

The public experiment axes are also config axes: policy method, action codec,
VLM backbone, dataset, trainer, and method-specific train recipe.

## Policy Training Composition

`praxis-train` loads `train_policy.yaml` when no config name is provided. That
template currently composes the quickstart example:

```yaml
defaults:
  - policy: bar_oat_pali
  - dataset: libero
  - trainer: policy
  - train_recipe/policy@_global_: bar
  - _self_
```

Inspecting composition is a login-node-safe operation:

```bash
uv run praxis-train policy=bar_oat_pali dataset.name=libero --cfg job
uv run praxis-train policy=ki_oat_pali train_recipe/policy@_global_=ki --cfg job
uv run praxis-train policy=fm_pali train_recipe/policy@_global_=fm --cfg job
```

BAR and KI require a codec checkpoint:

```text
policy.action_tokenizer.pretrained_path=/absolute/path/to/codec/pretrained_model
```

FM does not use `policy.action_tokenizer`.

## Codec Training Composition

Standalone codec training uses `train_codecs.yaml`:

```bash
uv run praxis-train --config-name train_codecs \
  codecs@codec=tokens/oat \
  dataset.name=libero dataset.repo_id=lerobot/libero \
  --cfg job
```

Codec configs use codec-owned fields:

```text
cfg.codec.kind
cfg.codec.name
```

Do not introduce a parallel `cfg.tokenizer` root for standalone codec training.

## Evaluation Composition

Local evaluation loads a policy checkpoint and wraps it with
`PolicyEvalAdapter`:

```bash
POLICY_CKPT="$(find outputs/train_policy -path '*/checkpoints/latest/pretrained_model' -type d | sort | tail -n 1)"
test -n "${POLICY_CKPT}"

uv run praxis-eval \
  policy=bar_oat_pali \
  policy.pretrained_path="${POLICY_CKPT}" \
  env.type=libero env.task=libero \
  eval.output_dir=outputs/eval/bar_oat_pali_smoke
```

Remote evaluation additionally requires:

```text
eval.mode=remote_grpc
server_address=host:port
```

## Train Recipes

Method-specific optimizer defaults live under:

```text
src/praxis/configs/train_recipe/policy/
```

The train recipe owns learning-rate scale, scheduler, warmup, and method-level
trainer defaults. Policy configs own policy semantics. Keeping these separate
prevents method comparisons from silently changing optimizer ownership.
