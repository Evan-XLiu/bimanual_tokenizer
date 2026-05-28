# Tests

Tests should follow code ownership first.

## Placement
- `src/praxis/...` tests live under `tests/src/praxis/...`
- top-level `scripts/...` tests live under `tests/scripts/...`
- only genuine cross-package workflows belong in `tests/integration/...`

## Naming
- prefer `test_<module>.py` for module-owned unit tests
- keep one file focused on one source module or one tight behavior surface
- use `_test_utils.py` only for small shared dummies and fixtures inside one package subtree

## Quality Bar
- test the narrowest honest abstraction boundary
- prefer assertions on observable behavior over helper implementation details
- add a regression test at the boundary where a bug actually escaped
- avoid generic dump buckets like `test_*helpers.py` when tests really target multiple modules
- avoid repeating the same schedule, decode, or config assertions across multiple backbones unless the family-specific plumbing differs

## Integration Scope
- use `tests/integration/...` when a test spans multiple packages or exercises a workflow end to end
- do not use `tests/integration/...` as a fallback for unclear ownership; decide the real owner first

## Current Hotspots
- No broad family-level token-generation policy dump buckets remain.

If a new top-level family suite starts mixing multiple source modules, split it back into module-owned files before it grows.
