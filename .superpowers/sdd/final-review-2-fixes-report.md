# Final Review 2 Fixes Report

## Status

`DONE`

Implementation commit:
`8283ce6cb2204e6020be8426da2992b821525e4e`

The report is committed separately after the implementation SHA is known. Its
own commit cannot be recorded inside the same immutable commit without a
self-referential amend loop.

## Scope And Preservation

Starting branch: `codex/ki-xtokenizer-paper-ar-review`

Starting reviewed HEAD: `8b6e428f445cf071bc4200ea409662c840b23f46`

Only the eleven allowed implementation, documentation, and directly covering
test files were included in the implementation commit. The dirty downstream
policy/KI, evaluation, serving, launcher, and offline-target work was neither
edited for this review wave nor staged.

Preservation hashes after the implementation commit:

```text
git diff | sha256sum
fd8f2e96246b5d17bfad4076b5b8f956e146daf97893bad527e58b8fadc54ae0  -

git ls-files --others --exclude-standard -z | sha256sum
13eff7345f1a6414029600512da5636a10ea4d5aecaffab80b4ecccb0cbe9d41  -
```

Both values exactly match the pre-wave preservation contract.

## Implemented Fixes

1. Documentation no longer claims that this committed branch contains a wired
   `ki_xtokenizer_qwen` cache consumer. The runbook and design require a
   separately reviewed integration to provide train/validation cache paths, a
   trusted artifact digest, split identity, and offline token-target plumbing.

2. Latest-checkpoint export detects `training_state/ema.pt` from either the
   checkpoint root or its `pretrained_model` child, rejects a symlinked EMA
   file, loads it with `weights_only=True`, validates exact parameter names via
   the existing `EMAModel`, and applies the shadow before artifact creation.
   The source digest binds both the pretrained directory and EMA bytes and is
   recomputed after model/EMA/metadata load. No-EMA exports retain the prior
   directory digest and raw stored weights.

3. Native cache manifests now require exact action and observation-state source
   digests. One shared public helper serializes a finite rank-2 little-endian
   float32 matrix with both shape dimensions, C-order bytes, and a domain
   separator. Generation hashes the split rows; native consumer binding hashes
   the live LeRobot `hf_dataset` columns before exposing any token row.

4. The paper recipe resolves a default optimizer-step global batch of 256 from
   `trainer.batch_size`, preserving deliberate small-batch smoke overrides.
   Symmetric global and local InfoNCE gather variable valid-pair counts from all
   ranks in stable rank order using autograd-capable distributed gathers.
   Single-process, uneven-count, and all-zero behavior is preserved.

5. Cache generation writes normalization/statistics, both split caches, and the
   summary into a unique sibling staging directory. Both split caches are
   opened and validated before the existing staging helper atomically publishes
   the bundle. Failures remove only invocation-owned staging/backup paths;
   pre-existing or absent requested outputs remain untouched and retryable.

## TDD Evidence

The work resumed after requirements 2-4 had completed their RED/GREEN cycles.
The retained handoff contained their GREEN results but not the exact earlier RED
commands or stdout. Those missing lines are not reconstructed here.

Retained GREEN evidence from the earlier session:

```text
EMA export/digest/symlink focused regressions: 3 passed
tests/src/praxis/data/test_action_token_cache.py: 43 passed
native writer/reader and source mutation focus: 3 passed
tests/src/praxis/codecs/latents/xtokenizer/test_objectives.py: 13 passed
focused Hydra global-batch composition: 1 passed
```

Atomic publication RED:

```text
uv run pytest -q \
  tests/scripts/tokenizer/test_export_xtokenizer.py::test_native_cache_failure_cleans_staging_and_retry_succeeds

1 failed
AssertionError: assert not True
The requested /cache directory remained after the injected train-split failure.
```

Atomic publication GREEN:

```text
uv run pytest -q \
  tests/scripts/tokenizer/test_export_xtokenizer.py::test_native_cache_failure_cleans_staging_and_retry_succeeds

1 passed in 0.61s
```

Adjacent atomic/provenance GREEN:

```text
uv run pytest -q tests/scripts/tokenizer/test_export_xtokenizer.py \
  -k 'native_cache_cli_is_self_contained or \
      native_cache_fails_closed_if_normalization_provenance_changes or \
      native_cache_refuses_to_overwrite_existing_output or \
      native_cache_failure_cleans_staging_and_retry_succeeds'

4 passed, 52 deselected in 0.67s
```

Global-batch override compatibility RED, added after the full smoke exposed an
eight-row fixture receiving a 256-row per-rank batch:

```text
uv run pytest -q \
  tests/src/praxis/configs/test_xtokenizer_train_config.py::test_xtokenizer_train_config_accepts_horizon_matched_overrides

4 failed
Expected global_batch_size=2; all four cases resolved 256.
```

Global-batch default/override GREEN:

```text
uv run pytest -q \
  tests/src/praxis/configs/test_xtokenizer_train_config.py::test_xtokenizer_train_config_accepts_horizon_matched_overrides \
  tests/src/praxis/configs/test_xtokenizer_train_config.py::test_xtokenizer_train_config_composes_with_documented_cli

5 passed in 0.80s
```

End-to-end smoke GREEN after the recipe fix:

```text
uv run pytest -q tests/scripts/test_train_xtokenizer_smoke.py

1 passed in 27.78s
```

## Verification

All directly covering files:

```text
uv run pytest -q \
  tests/scripts/tokenizer/test_export_xtokenizer.py \
  tests/src/praxis/data/test_action_token_cache.py \
  tests/src/praxis/codecs/latents/xtokenizer/test_objectives.py \
  tests/src/praxis/configs/test_xtokenizer_train_config.py

122 passed in 13.64s
```

Complete committed X-Tokenizer changed-test set (14 files):

```text
uv run pytest -q \
  tests/scripts/test_train_xtokenizer_smoke.py \
  tests/scripts/tokenizer/test_export_xtokenizer.py \
  tests/scripts/tokenizer/test_prepare_bridge_xtokenizer_stats.py \
  tests/src/praxis/codecs/latents/xtokenizer/test_codec.py \
  tests/src/praxis/codecs/latents/xtokenizer/test_config.py \
  tests/src/praxis/codecs/latents/xtokenizer/test_eval_metrics.py \
  tests/src/praxis/codecs/latents/xtokenizer/test_modeling.py \
  tests/src/praxis/codecs/latents/xtokenizer/test_objectives.py \
  tests/src/praxis/codecs/latents/xtokenizer/test_quantizer.py \
  tests/src/praxis/configs/test_xtokenizer_train_config.py \
  tests/src/praxis/data/test_action_token_cache.py \
  tests/src/praxis/data/test_bridge_xtokenizer.py \
  tests/src/praxis/data/test_xtokenizer_canonical_stats.py \
  tests/src/praxis/training/test_codecs_trainer_xtokenizer_bindings.py

349 passed in 56.68s
```

The first full-set run was `348 passed, 1 failed`: the existing smoke command
overrode `trainer.batch_size=2` but the initial independent literal
`global_batch_size=256` took precedence and yielded an empty drop-last loader.
The TDD compatibility cycle above fixed the recipe at the configuration source.

Static and documentation gates:

```text
uv run ruff check .
All checks passed!

uv run ruff format --check .
453 files already formatted

uv run python -m basedpyright
0 errors, 0 warnings, 0 notes

uv run python docs/scripts/check_snippets.py
Compiled 5 Python fenced code blocks.
Checked 70 Bash fenced code blocks.
Checked Markdown frontmatter scalar quoting.
Checked forbidden Markdown patterns.
Checked policy catalog / codec registry source: 37 11

git diff --check
exit 0
```

After an explicit type narrowing for Torch's functional autograd gather, the
directly affected test and gates were rerun:

```text
uv run pytest -q tests/src/praxis/codecs/latents/xtokenizer/test_objectives.py
13 passed in 4.62s

uv run python -m basedpyright
0 errors, 0 warnings, 0 notes

uv run ruff check src/praxis/codecs/latents/xtokenizer/objectives.py
All checks passed!

uv run ruff format --check src/praxis/codecs/latents/xtokenizer/objectives.py
1 file already formatted
```

## Changed Files

- `docs/workflows/train-xtokenizer.md`
- `docs/superpowers/specs/2026-07-24-xtokenizer-quantile-normalization-design.md`
- `scripts/tokenizer/generate_bridge_xtokenizer_cache.py`
- `src/praxis/codecs/latents/xtokenizer/inference.py`
- `src/praxis/codecs/latents/xtokenizer/objectives.py`
- `src/praxis/configs/train_recipe/codecs/xtokenizer.yaml`
- `src/praxis/data/action_token_cache.py`
- `tests/scripts/tokenizer/test_export_xtokenizer.py`
- `tests/src/praxis/codecs/latents/xtokenizer/test_objectives.py`
- `tests/src/praxis/configs/test_xtokenizer_train_config.py`
- `tests/src/praxis/data/test_action_token_cache.py`

## Self-Review And Simplifications

- Reused `EMAModel` instead of duplicating shadow validation/application.
- Reused `publish_staged_directory` and `remove_path`; no dependency was added.
- Kept one canonical matrix hash helper shared by writer and consumer.
- Kept native token arrays sealed until all artifact, dataset, split, row, source
  value, sidecar, and horizon checks succeed.
- Preserved the prior raw no-EMA directory digest exactly.
- Preserved single-process InfoNCE without a collective and used one bounded
  padding/gather helper for variable distributed counts.
- Scanned the staged production diff for debug statements, unresolved markers,
  and common secret patterns; no findings.
- Staged every implementation path explicitly. No broad add, reset, restore,
  checkout, clean, or commit-all command was used.

## Residual Risks

- Distributed semantics were exercised with two CPU/gloo ranks, uneven valid
  counts, exact ordering/loss, and gradient flow. Multi-node NCCL execution was
  not available in this environment.
- Cache publication used a real tiny native writer/reader CLI round trip and
  injected failure/retry. A full-scale Bridge dataset cache generation was not
  run.
- Source digest post-checks cover the concurrent-mutation contract in code, but
  this wave does not add a separately injected concurrent EMA mutation test.
- The downstream KI policy consumer remains intentionally uncommitted and must
  satisfy the documented cache path, trusted digest, split, and offline-target
  contract in a separate review.
