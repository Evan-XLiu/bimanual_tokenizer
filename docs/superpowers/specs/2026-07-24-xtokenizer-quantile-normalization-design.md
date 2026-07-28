# X-Tokenizer Quantile Normalization Design

## Goal

Align the Praxis X-Tokenizer preprocessing contract with the released
X-Tokenizer implementation while keeping the public Praxis codec interface in
canonical absolute action space.

The model must receive observation-relative action channels normalized with
per-channel `q001`/`q999` statistics, while its observation anchor must use an
independent absolute-state normalizer. Existing mean/std artifacts and token
caches remain immutable and are not reused by the new contract.

## Official Contract

For one absolute action chunk `a[0:T]` and the proprioceptive observation `o`
immediately before that chunk:

- Cartesian position uses `a_position - o_position`.
- 6D rotation uses `R_relative = R_action @ R_observation.T`.
- Gripper and other identity channels remain absolute.
- The same observation anchors every frame in the chunk; this is not a
  frame-to-frame delta.

The action representation and observation anchor use independent robust
MinMax statistics:

```text
normalized = clamp((value - q001) * 2 / (q999 - q001) - 1, -1, 1)
```

Decoding inverts the MinMax transform and then restores absolute actions:

```text
position_absolute = position_relative + position_observation
R_absolute = R_relative @ R_observation
```

## Scope Decision

This change targets the released open-source helper contract with a fixed
statistics horizon. The maintained Bridge training configuration uses horizon
32, so its default statistics artifact is built with `stats_horizon=32`.

The paper also describes pretraining with horizons sampled from `[8, 64]`.
That is a separate variable-horizon model and sampling change. This design
supports building independent fixed-horizon statistics for 8, 16, 32, or 64,
but it never silently combines those distributions or permits a model to load
statistics built for a different horizon.

## Artifact Contract

Add a dedicated X-Tokenizer quantile statistics artifact alongside the
existing canonical mean/std artifact. The artifact contains:

```json
{
  "format_version": 2,
  "dataset_repo_id": "chaoqi-liu/bridge_single_view",
  "dataset_revision": "<40-character commit>",
  "robot_type": "WidowX",
  "action_layout": {
    "convention": "absolute",
    "dof_dims": {},
    "dof_order": []
  },
  "normalization": {
    "mode": "quantile_min_max",
    "quantile_low": 0.001,
    "quantile_high": 0.999,
    "clamp": true,
    "stats_horizon": 32,
    "predict_action": {
      "space": "observation_relative",
      "q001": [],
      "q999": []
    },
    "observation_state": {
      "space": "absolute",
      "q001": [],
      "q999": []
    }
  },
  "content_fingerprint": "<sha256>"
}
```

The fingerprint covers repository and revision identity, robot type, ordered
action layout, both statistic blocks, quantile levels, clamp behavior, and
statistics horizon. Loading fails closed on missing fields, non-finite values,
non-positive ranges, changed ordering, or horizon mismatch.

## Statistics Population

For each episode and each valid start `t`:

1. Convert the Bridge observation at `t` to the canonical absolute 10D anchor.
2. Accumulate raw Bridge controller increments through `t + horizon` to obtain
   the canonical absolute endpoint.
3. Convert that endpoint to the official observation-relative representation.
4. Add the endpoint representation to the `predict_action` population.
5. Add the canonical absolute anchor to the `observation_state` population.

Only endpoint deltas enter the predict population, matching the released
statistics helper. Intermediate chunk frames do not enter that population.
The implementation uses exact NumPy quantiles and adds no TDigest dependency.

## Runtime Data Flow

The dataset adapter continues to emit canonical absolute actions and a
canonical absolute `observation.state`. `XTokenizerCodec` owns the internal
conversion:

```text
absolute actions + absolute anchor
  -> observation-relative actions
  -> predict-action quantile MinMax + clamp

absolute anchor
  -> observation-state quantile MinMax + clamp

decoder output
  -> inverse predict-action MinMax
  -> compose with physical absolute anchor
  -> canonical absolute actions
```

The reconstruction loss is evaluated against normalized observation-relative
actions. Public `encode`, `decode`, `autoencode`, reconstruction metrics, cache
generation, and evaluation continue accepting or returning absolute actions.

## Persistence And Cache Identity

Checkpoints and inference exports persist both normalizer blocks. Loading must
require both roles and must not clone action statistics into the observation
role. Artifact hashes include the complete normalization payload. Exporting a
latest checkpoint root, or its `pretrained_model` child, applies the adjacent
strictly validated `training_state/ema.pt` shadow when present and binds both
the model directory and EMA bytes into the source digest. Exports without an
adjacent EMA state retain their stored model weights.

Native token-cache generation must call the codec's absolute-action encode
path exactly once. It must not pre-normalize absolute actions externally or
call the encoder directly. Cache manifests record the normalization artifact
fingerprint and `stats_horizon`; a cache generated under another horizon or
normalization contract is rejected.

## Compatibility

- Existing mean/std artifacts remain readable by existing non-X-Tokenizer
  paths.
- Existing X-Tokenizer checkpoints and caches retain their old identity and
  are not mutated.
- New paper-aligned checkpoints require the version-2 quantile artifact.
- KI policy continuous-action normalization is unchanged. This committed
  change produces authenticated offline token caches but does not provide a
  committed KI consumer. A separately reviewed downstream integration must
  provide train and validation cache paths, a trusted artifact digest, split
  identity, and offline token-target plumbing before a policy can consume them.

## Verification

Tests must prove:

- Position and non-identity rotation round-trip through relative conversion.
- Gripper remains an identity channel.
- Synthetic endpoint and anchor populations produce distinct `q001/q999`.
- Outliers clamp to `[-1, 1]`.
- Loading rejects swapped, missing, tampered, or wrong-horizon statistic roles.
- Checkpoint and inference export/load preserve both blocks exactly.
- Native cache tokens equal direct `codec.encode(absolute, anchor=absolute)`.
- Existing generic policy normalizers remain unchanged.
