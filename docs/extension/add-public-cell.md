# Add A Public Cell

Add a public cell when a supported objective, action representation, and
backbone combination should become part of the documented matrix.

## Required Changes

1. Add or update the family in `src/praxis/policies/catalog.py`.
2. Add the concrete class in the method `variants.py`.
3. Add a Hydra alias under `src/praxis/configs/policy/`.
4. Add tests proving the name resolves, composes, and constructs.

## Naming

Token-based cells:

```text
<method>_<codec>_<backbone>
```

Continuous FM-style cells:

```text
fm_<backbone>
```

## Checklist

- The cell has a clear research role.
- The config alias composes without hidden script logic.
- BAR/KI cells require a codec checkpoint.
- The factory path works for training, serving, and evaluation.
- The docs matrix is updated if the public count changes.
