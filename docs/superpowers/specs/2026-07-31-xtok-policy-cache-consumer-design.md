# X-Tokenizer Policy Cache Consumer Design

## Goal

Make native format-v2 X-Tokenizer caches directly consumable by Qwen/KI policy
training, and make the cache generator emit the authenticated canonical
normalization identity required by the native reader.

## Design

The cache generator keeps tokenizer inference unchanged. When it writes the
native `normalization.json` sidecar, it copies the q001/q999 bounds from
`training_data_identity.canonical_stats_identity.normalization` and checks that
their float32 materialization is identical to the runtime manifest values. The
sidecar therefore satisfies exact JSON identity checks without changing encoded
token IDs.

`PolicyTrainer` passes the logical split (`train` or `val`), the trainer's
deterministic split seed and validation ratio, and an explicitly configured
artifact SHA256 to `CachedActionTokenDataset`. The native binding remains
fail-closed when the artifact digest is absent or mismatched; legacy caches
retain their existing behavior.

The submitted CloudML task is documented separately as a small, reviewable
configuration record. It references immutable external datasets, cache roots,
checkpoints, and output directories rather than copying artifacts into git.

## Verification

- A generator unit test proves canonical q001/q999 bounds are used and rejects
  a canonicalization that would alter float32 runtime values.
- A policy trainer unit test proves train and validation native cache wrappers
  receive their trusted split and artifact-binding arguments.
- Focused cache/trainer test modules and formatting checks run before staging.

## Commit Boundaries

1. Consumer and generator fixes with focused regression tests.
2. Reproducible CloudML submitted-task configuration and resume instructions.
