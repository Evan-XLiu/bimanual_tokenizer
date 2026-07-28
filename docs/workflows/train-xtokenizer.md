# Train X-Tokenizer

X-Tokenizer training is an offline artifact pipeline. Training reads a local
LeRobot cache, canonical Bridge action statistics, and a prepared Qwen
vision-language sidecar. It does not load or invoke Qwen.

Keep datasets, model snapshots, features, checkpoints, and generated caches
under the shared experiment root:

```bash
export EXP_ROOT=/mnt/robot-research-exp-0/user/hxs
export HF_HOME="${EXP_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export DATASET_ROOT="${EXP_ROOT}/datasets/bridge_single_view"
export STATS_ROOT="${EXP_ROOT}/xtokenizer/bridge_single_view"
export VL_SIDECAR="${EXP_ROOT}/xtokenizer/bridge_single_view/vl_features"
export HORIZON=32
export STATS="${STATS_ROOT}/stats_v3_h${HORIZON}.json"
export RUN_ROOT="${EXP_ROOT}/runs/xtokenizer_bridge/h${HORIZON}"
export ARTIFACT="${EXP_ROOT}/artifacts/xtokenizer_bridge/h${HORIZON}"
export TOKEN_CACHE="${EXP_ROOT}/caches/bridge_xtokenizer/h${HORIZON}"
```

## Prerequisites

Use immutable 40-character Hub commit SHAs for the dataset and Qwen model.
Update `src/praxis/configs/dataset/xtokenizer_bridge.yaml` with those revisions
and the paths above. The local LeRobot cache must already represent the pinned
dataset revision and contain `.praxis-dataset-provenance.json`; preparation
fails closed when that provenance is missing or different.

The configured action layout is model semantics, not display metadata.
`dof_dims` and `dof_order` describe the public canonical absolute 10-D layout.
The codec converts action chunks to one pre-chunk-observation-relative action
representation internally. The X-Tokenizer paper calls this anchored
representation a "delta action", but it is not an adjacent-frame control delta;
this repository calls it `observation_relative` to keep those meanings distinct.
Bridge's raw controller commands are reconstructed once for each complete
episode, starting from the controller target initialized by the first measured
state. Each clipped XYZ/RPY command updates the previous command target using
the measured EEF position as the controller pivot; rotations left-compose in
world axes, and clipped gripper commands remain absolute targets. Action chunks
slice this one shared absolute target trajectory, then the codec expresses every
pose relative to the observation at the chunk start.

## 1. Prepare Canonical Action Statistics

Scan the raw 7-D Bridge state and action columns, recover the public absolute
10-D chunks, then compute separate robust q0.1%/q99.9% MinMax ranges for the
internal observation-relative actions and absolute observation anchors. Build
each supported statistics horizon independently:

```bash
for horizon in 8 16 32 64; do
  uv run python scripts/tokenizer/prepare_bridge_xtokenizer_stats.py \
    --dataset-config src/praxis/configs/dataset/xtokenizer_bridge.yaml \
    --stats-horizon "${horizon}" \
    --output "${STATS_ROOT}/stats_v3_h${horizon}.json"
done
```

The v3 statistics contract is the first version bound to episode-wide Bridge
controller-target reconstruction. Do not reuse `stats_v2_h*.json`, an
X-Tokenizer checkpoint trained with those files, or an offline token cache
generated from such a checkpoint. Rebuild statistics first, then retrain,
export, and regenerate the token cache in that order.

These four artifacts are recomputed from the dataset, not copied, renamed, or
reused across horizons. The horizon controls which endpoint samples enter the
quantile population, so an artifact generated for h8 is not valid for h16,
h32, or h64. The embedded `normalization.stats_horizon` is authoritative; a
horizon-qualified filename makes the contract visible but does not replace
runtime validation.

The artifact binds the dataset repository, immutable revision, robot type,
statistics horizon, ordered public action layout, both representation contracts,
normalization mode, quantile range, clamp behavior, both statistic blocks, and a
content fingerprint.
Training validates this contract before opening the dataset or constructing the
model and optimizer.

Every stage must preserve this five-way equality:

```text
codec.action_horizon
  == dataset.action_horizon
  == statistics normalization.stats_horizon
  == exported checkpoint/artifact manifest.action_horizon
  == cache manifest.action_horizon
```

Never use a codec-only horizon override. Select the matching dataset horizon and
independently generated statistics artifact in the same composition.

## 2. Prepare Pinned Qwen Features

Run Qwen extraction as a separate preparation job. Substitute the same
immutable revisions configured in the dataset YAML:

```bash
export DATASET_REVISION=DATASET_COMMIT_SHA
export QWEN_REVISION=QWEN_COMMIT_SHA

uv run python scripts/tokenizer/prepare_vl_features.py \
  --output-root "${VL_SIDECAR}" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-repo-id chaoqi-liu/bridge_single_view \
  --dataset-revision "${DATASET_REVISION}" \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --model-revision "${QWEN_REVISION}" \
  --extractor-revision praxis-qwen2.5-vl-paper-a3-v2 \
  --image-key observation.images.image \
  --stride 4 \
  --feature-dim 3584 \
  --device cuda
```

The sidecar is resumable and publishes one validated manifest only after all
episode shards are ready. The Qwen processor and model snapshots resolve under
`${HF_HUB_CACHE}`; `QWEN_REVISION` must be the exact immutable 40-character Hub
commit SHA recorded in the feature sidecar manifest. The manifest's
`generation_fingerprint` and complete semantic identity are checkpoint-owned
once training begins, while all large Qwen and Hugging Face artifacts remain
under `${EXP_ROOT}`.

## 3. Train

The maintained Hydra composition defaults to h32 for both codec and dataset and
resolves `dataset.xtokenizer_quantile_stats` to
`${STATS_ROOT}/stats_v3_h32.json`:

```bash
uv run praxis-train --config-name train_codecs \
  codec=latents/xtokenizer \
  dataset=xtokenizer_bridge \
  train_recipe/codecs=xtokenizer \
  output_dir="${RUN_ROOT}"
```

The paper recipe defaults the optimizer-step global batch to 256. Distributed
startup derives each rank's DataLoader batch from the configured recipe batch
divided by world size and gradient accumulation, and rejects topologies where
that division is not exact. Override `trainer.batch_size` only when deliberately
changing the optimizer-step batch; the recipe binds `global_batch_size` to it.
The contrastive objective gathers valid pairs from every rank before computing
InfoNCE, so its negative set does not change from global to rank-local under
DDP.

Set `output_dir="${RUN_ROOT}"` and any deliberate trainer overrides on the same
command. For example, inspect the resolved default h32 job before a long run:

```bash
uv run praxis-train --config-name train_codecs \
  codec=latents/xtokenizer \
  dataset=xtokenizer_bridge \
  train_recipe/codecs=xtokenizer \
  output_dir="${RUN_ROOT}" \
  --cfg job
```

For a deliberate non-default horizon, override the codec, dataset, and public
statistics key together and use horizon-qualified outputs. This h16 example
must use the h16 artifact generated by the loop above:

```bash
export HORIZON=16
export STATS="${STATS_ROOT}/stats_v3_h${HORIZON}.json"
export RUN_ROOT="${EXP_ROOT}/runs/xtokenizer_bridge/h${HORIZON}"
export ARTIFACT="${EXP_ROOT}/artifacts/xtokenizer_bridge/h${HORIZON}"

uv run praxis-train --config-name train_codecs \
  codec=latents/xtokenizer \
  dataset=xtokenizer_bridge \
  train_recipe/codecs=xtokenizer \
  codec.action_horizon="${HORIZON}" \
  dataset.action_horizon="${HORIZON}" \
  dataset.xtokenizer_quantile_stats="${STATS}" \
  output_dir="${RUN_ROOT}"
```

A completed run writes `${RUN_ROOT}/TRAINING_COMPLETE` and the resumable
checkpoint at `${RUN_ROOT}/checkpoints/latest`.

## 4. Resume

Resume from the checkpoint directory, not its nested `pretrained_model`
directory:

```bash
uv run praxis-train --config-name train_codecs \
  codec=latents/xtokenizer \
  dataset=xtokenizer_bridge \
  train_recipe/codecs=xtokenizer \
  codec.action_horizon="${HORIZON}" \
  dataset.action_horizon="${HORIZON}" \
  dataset.xtokenizer_quantile_stats="${STATS}" \
  output_dir="${RUN_ROOT}" \
  run_mode=resume \
  resume_checkpoint="${RUN_ROOT}/checkpoints/latest"
```

The resolved sidecar identity is stored in
`checkpoints/latest/pretrained_model/train_config.yaml`. Resume reopens the
configured sidecar and requires exact identity equality, including
`generation_fingerprint`, before model or optimizer construction. Regenerating
features in another directory or under the same directory with different
inputs is therefore rejected. Restore the original sidecar generation or start
a new run; do not edit the saved identity.

## 5. Export

Export the latest training checkpoint into the inference-only artifact consumed
by cache generation:

```bash
uv run python scripts/tokenizer/export_xtokenizer.py \
  "${RUN_ROOT}/checkpoints/latest" \
  "${ARTIFACT}"
```

Export verifies that the checkpoint model layout exactly matches the training
dataset layout, removes training-private state, records normalization and
source digests, and reloads the staged artifact before publishing it. For
`checkpoints/latest` or its `pretrained_model` child, an adjacent
`training_state/ema.pt` is loaded strictly and applied before export. The source
digest binds both the pretrained model directory and EMA bytes, and both are
rechecked after loading. Candidate checkpoints without an adjacent EMA state
retain the direct model-weight export behavior.

## 6. Generate The Maintained H32 Offline Token Cache

Pass the exported artifact directory to the existing Bridge cache job:

```bash
export HORIZON=32
export ARTIFACT="${EXP_ROOT}/artifacts/xtokenizer_bridge/h${HORIZON}"
export TOKEN_CACHE="${EXP_ROOT}/caches/bridge_xtokenizer/h${HORIZON}"

uv run python scripts/tokenizer/generate_bridge_xtokenizer_cache.py \
  --dataset-root "${DATASET_ROOT}" \
  --checkpoint "${ARTIFACT}" \
  --output-root "${TOKEN_CACHE}" \
  --device cuda
```

The native Bridge cache generator is currently maintained at h32. Use the h32
checkpoint/artifact and h32-qualified output path shown above; generating
statistics for h8, h16, or h64 does not make non-h32 native cache generation
available. Extending that generator to other horizons requires a separate
code and validation change.

The generator stages the complete bundle, validates both split caches, and then
publishes the requested output directory atomically. The resulting h32 split
caches are authenticated offline artifacts, but this committed branch does not
contain a `ki_xtokenizer_qwen` policy/configuration path wired to consume them.
That downstream policy integration requires a separately reviewed change that
provides the train and validation cache paths, a trusted inference-artifact
digest, the expected split identity, and offline token-target plumbing. Such a
consumer must use the cached targets rather than run X-Tokenizer or Qwen online.

The cache script still accepts a legacy external X-Tokenizer checkpoint file.
That compatibility branch imports the separately installed external
`xtokenizer` package and uses its legacy statistics format and fixed checkpoint
contract. It is not the native Praxis training/export path. Use the exported
artifact directory above for checkpoints produced by this repository.
