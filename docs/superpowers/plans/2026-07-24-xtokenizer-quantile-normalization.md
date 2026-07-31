# X-Tokenizer Quantile Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train, export, and cache X-Tokenizer tokens using independent `q001`/`q999` MinMax normalizers for observation-relative actions and absolute observation anchors.

**Architecture:** Keep dataset and public codec I/O in canonical absolute action space. Introduce a versioned fixed-horizon statistics artifact, transform actions inside `XTokenizerCodec`, persist both normalizer roles, and route native cache generation through the same absolute-action encode path.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, PyArrow, Hydra/OmegaConf, pytest.

## Global Constraints

- Default Bridge statistics horizon is exactly 32.
- Quantiles are exactly 0.001 and 0.999 and normalized values are clamped to `[-1, 1]`.
- Action and anchor statistics are separate and persisted together.
- Position and rotation are observation-relative; gripper remains absolute.
- No new dependency is added.
- Existing mean/std artifacts and caches are never overwritten.

---

### Task 1: Versioned Dual-Role Statistics Artifact

**Files:**
- Modify: `src/praxis/data/xtokenizer_canonical_stats.py`
- Test: `tests/src/praxis/data/test_xtokenizer_canonical_stats.py`

**Interfaces:**
- Produces: `XTokenizerQuantileStatistics.from_path(...)`, `write_xtokenizer_quantile_statistics(...)`, `predict_normalization_stats(action_key)`, and `observation_normalization_stats(observation_key)`.
- Persists: dataset identity, robot type, action layout, `stats_horizon`, quantile levels, clamp flag, two statistic blocks, and content fingerprint.

- [ ] **Step 1: Write failing round-trip and validation tests**

Add tests that write distinct predict/observation vectors, reload them, and
assert exact tensors and identity. Add rejection tests for missing roles,
`q001 >= q999`, changed horizon, swapped spaces, and changed fingerprints.

- [ ] **Step 2: Verify the tests fail for the missing v2 API**

Run:

```bash
uv run pytest tests/src/praxis/data/test_xtokenizer_canonical_stats.py -q
```

Expected: collection or assertion failure because the v2 artifact API does not exist.

- [ ] **Step 3: Implement the minimal immutable artifact API**

Use a frozen dataclass and atomic/idempotent publication matching the existing
artifact conventions. Require vectors of `action_dim`, finite values, and
strictly positive elementwise ranges.

- [ ] **Step 4: Verify the artifact tests pass**

Run the command from Step 2. Expected: all tests pass.

### Task 2: Fixed-Horizon Quantile Statistics Builder

**Files:**
- Modify: `scripts/tokenizer/prepare_bridge_xtokenizer_stats.py`
- Modify: `src/praxis/data/bridge_xtokenizer.py`
- Test: `tests/scripts/tokenizer/test_prepare_bridge_xtokenizer_stats.py`
- Test: `tests/src/praxis/data/test_bridge_xtokenizer.py`

**Interfaces:**
- Consumes: raw Bridge 7D state/action episodes and the configured canonical layout.
- Produces: one v2 artifact for one explicit positive `stats_horizon`.

- [ ] **Step 1: Write failing endpoint-population tests**

Construct an episode whose intermediate deltas differ from its horizon endpoint.
Assert predict quantiles use only `t -> t+horizon` endpoint representations,
observation quantiles use anchors, rotations use `R_endpoint @ R_anchor.T`, and
gripper is unchanged.

- [ ] **Step 2: Verify the tests fail under the current absolute-action builder**

```bash
uv run pytest tests/scripts/tokenizer/test_prepare_bridge_xtokenizer_stats.py tests/src/praxis/data/test_bridge_xtokenizer.py -q
```

Expected: failures showing a single absolute-action population and no horizon identity.

- [ ] **Step 3: Implement exact q001/q999 generation**

Add `--stats-horizon` with default 32. Collect endpoint and anchor populations
per episode, calculate `np.quantile(values, [0.001, 0.999], axis=0)`, reject
empty or degenerate channels, and publish the v2 artifact. Do not add TDigest.

- [ ] **Step 4: Verify both statistics and geometry tests pass**

Run the Step 2 command. Expected: all tests pass.

### Task 3: Codec-Specific Relative Preprocessing And Dual Normalizers

**Files:**
- Modify: `src/praxis/codecs/latents/xtokenizer/action_space.py`
- Modify: `src/praxis/codecs/latents/xtokenizer/codec.py`
- Modify: `src/praxis/training/codecs_trainer.py`
- Test: `tests/src/praxis/codecs/latents/xtokenizer/test_codec.py`
- Test: `tests/src/praxis/training/test_codecs_trainer_xtokenizer_bindings.py`

**Interfaces:**
- Consumes: absolute actions, absolute anchor, predict stats, observation stats.
- Produces: `set_xtokenizer_normalization_stats(...)`, normalized relative actions, normalized anchor, and absolute decoded actions.

- [ ] **Step 1: Write failing independent-normalizer and clamp tests**

Use deliberately different predict and observation ranges. Hook the encoder and
assert its first input is relative-action normalized with predict stats, its
anchor input uses observation stats, and both are clamped. Assert decoder output
is inverse-normalized and re-anchored.

- [ ] **Step 2: Verify the tests fail because both roles share one normalizer**

```bash
uv run pytest tests/src/praxis/codecs/latents/xtokenizer/test_codec.py tests/src/praxis/training/test_codecs_trainer_xtokenizer_bindings.py -q
```

Expected: role-specific normalization assertions fail.

- [ ] **Step 3: Implement the scoped X-Tokenizer normalizer lifecycle**

Keep generic `BaseCodec` behavior unchanged. Let `XTokenizerCodec` own separate
predict and observation `Normalize`/`Unnormalize` modules, explicit clamp, and
the existing differentiable SO(3) relative/absolute conversion. Bind the v2
artifact in `CodecsTrainer` and reject its absence for the paper-aligned recipe.

- [ ] **Step 4: Verify codec and trainer tests pass**

Run the Step 2 command. Expected: all tests pass.

### Task 4: Checkpoint, Inference Export, And Cache Consistency

**Files:**
- Modify: `src/praxis/codecs/latents/xtokenizer/inference.py`
- Modify: `src/praxis/codecs/latents/xtokenizer/cache.py`
- Modify: `scripts/tokenizer/export_xtokenizer.py`
- Modify: `scripts/tokenizer/generate_bridge_xtokenizer_cache.py`
- Test: `tests/scripts/tokenizer/test_export_xtokenizer.py`
- Test: `tests/src/praxis/data/test_action_token_cache.py`

**Interfaces:**
- Persists: both normalizer roles and v2 artifact identity.
- Guarantees: cache generation uses the public absolute-action codec path once.

- [ ] **Step 1: Write failing persistence and token-equivalence tests**

Assert save/load and export/load preserve both roles bitwise. Reject missing or
swapped roles. Generate a cache and assert each token row equals direct
`loaded_codec.encode(absolute_actions, anchor=absolute_anchor)`.

- [ ] **Step 2: Verify failures expose the one-role export and direct-encoder cache path**

```bash
uv run pytest tests/scripts/tokenizer/test_export_xtokenizer.py tests/src/praxis/data/test_action_token_cache.py -q
```

Expected: normalization schema and token-equivalence failures.

- [ ] **Step 3: Implement dual-role persistence and a single encode path**

Extend inference normalization validation and hashing to require both roles.
Remove cloned observation stats from the cache adapter. Route cache generation
through `codec.encode` with absolute actions and anchors, and record the v2
fingerprint plus horizon in each cache manifest.

- [ ] **Step 4: Verify persistence and cache tests pass**

Run the Step 2 command. Expected: all tests pass.

### Task 5: Configuration, Horizon Runbook, And Regression Verification

**Files:**
- Modify: `src/praxis/configs/dataset/xtokenizer_bridge.yaml`
- Modify: `docs/workflows/train-xtokenizer.md`
- Test: `tests/scripts/test_train_xtokenizer_smoke.py`
- Test: `tests/src/praxis/configs/test_xtokenizer_train_config.py`

**Interfaces:**
- Configures: `dataset.xtokenizer_quantile_stats` and explicit horizon identity.
- Documents: independent artifact generation for horizons 8, 16, 32, and 64.

- [ ] **Step 1: Write failing config composition and smoke assertions**

Assert the maintained recipe resolves a horizon-matched v2 artifact and rejects
model/statistics horizon mismatch before model construction.

- [ ] **Step 2: Verify the tests fail with the old single statistics path**

```bash
uv run pytest tests/src/praxis/configs/test_xtokenizer_train_config.py tests/scripts/test_train_xtokenizer_smoke.py -q
```

Expected: missing v2 path and horizon contract failures.

- [ ] **Step 3: Add configuration and exact build commands**

Document one output file per horizon, for example:

```bash
for horizon in 8 16 32 64; do
  uv run python scripts/tokenizer/prepare_bridge_xtokenizer_stats.py \
    --dataset-config src/praxis/configs/dataset/xtokenizer_bridge.yaml \
    --stats-horizon "${horizon}" \
    --output "${EXP_ROOT}/xtokenizer/bridge_single_view/stats_h${horizon}.json"
done
```

Document that `codec.action_horizon`, dataset sampling horizon, statistics
horizon, checkpoint manifest, and cache manifest must match exactly.

- [ ] **Step 4: Run focused and broad regression suites**

```bash
uv run pytest \
  tests/src/praxis/codecs/latents/xtokenizer \
  tests/src/praxis/data/test_bridge_xtokenizer.py \
  tests/src/praxis/data/test_xtokenizer_canonical_stats.py \
  tests/src/praxis/data/test_action_token_cache.py \
  tests/src/praxis/training/test_codecs_trainer_xtokenizer_bindings.py \
  tests/scripts/tokenizer/test_prepare_bridge_xtokenizer_stats.py \
  tests/scripts/tokenizer/test_export_xtokenizer.py \
  tests/scripts/test_train_xtokenizer_smoke.py -q
```

Expected: all tests pass with no warnings attributable to the change.
