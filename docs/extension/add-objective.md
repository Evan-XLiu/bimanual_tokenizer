# Add An Objective

Add an objective when the research idea changes the training loss, inference
procedure, or gradient path.

## Files

```text
src/praxis/policies/<method>/
src/praxis/configs/policy/<method>/
src/praxis/configs/train_recipe/policy/<method>.yaml
tests/src/praxis/policies/<method>/
```

Follow the existing families:

```text
src/praxis/policies/bar/
src/praxis/policies/fm/
src/praxis/policies/ki/
```

## Decide Whether The Objective Uses A Codec

If the objective predicts action tokens or uses token supervision, it needs a
codec coordinate and a pretrained tokenizer path. If the objective acts
directly in continuous action space, it should follow the FM-style null codec
coordinate.

## Public Matrix Wiring

Add a `PolicyFamily` in `src/praxis/policies/catalog.py`, then add concrete
variant classes and Hydra aliases.

## Tests

Add tests for:

- loss construction;
- inference action shape;
- factory construction;
- pretrained loading with `policy_io.json` and `norm_stats.pt`;
- catalog names and Hydra aliases.
