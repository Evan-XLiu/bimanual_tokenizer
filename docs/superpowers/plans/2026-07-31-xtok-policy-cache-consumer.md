# X-Tokenizer Policy Cache Consumer Implementation Plan

> **For agentic workers:** This plan is executed inline in the current remote workspace; no CloudML submit or clone operation is part of the plan.

**Goal:** Make native X-Tokenizer caches usable by policy training and archive a reproducible CloudML launch/resume configuration.

**Architecture:** Keep native cache provenance checks strict. Canonicalize only the JSON normalization sidecar at generation time, and pass trusted identity values from dataset/trainer configuration into the existing binding boundary. Store CloudML parameters as static documentation/configuration under `scripts/cloudml-submitted/`.

**Tech Stack:** Python, PyTorch Dataset, OmegaConf, pytest, CloudML custom-train YAML/documentation.

## Global Constraints

- Do not weaken native cache provenance validation.
- Do not commit datasets, checkpoints, token arrays, outputs, secrets, or temporary helper scripts.
- Do not submit, clone, stop, or delete a CloudML task.
- Preserve the existing uncommitted `scripts/diagnostics/` directories.

### Task 1: Native cache consumer and generator

**Files:**
- Modify: `scripts/tokenizer/generate_bridge_xtokenizer_cache.py`
- Modify: `src/praxis/configs/dataset/bridge_single_view.yaml`
- Modify: `src/praxis/training/policy_trainer.py`
- Test: `tests/scripts/tokenizer/test_generate_bridge_xtokenizer_cache.py`
- Test: `tests/src/praxis/training/test_policy_trainer_hooks.py`

- [ ] Add failing tests for canonical normalization and trainer binding arguments.
- [ ] Run the focused tests and observe the expected failures.
- [ ] Add the minimal canonicalization helper and trainer/config plumbing.
- [ ] Run the focused tests and the existing cache regression module.
- [ ] Stage only Task 1 files for review.

### Task 2: Submitted CloudML configuration

**Files:**
- Create: `scripts/cloudml-submitted/20260730-xtok-policy-repro/README.md`
- Create: `scripts/cloudml-submitted/20260730-xtok-policy-repro/task.yaml`

- [ ] Record the stopped task's immutable launch parameters and external artifact identities.
- [ ] Document fresh-start and step-30000 resume commands without embedding runtime-generated helpers.
- [ ] Validate YAML/readability and stage only Task 2 files after Task 1 review staging is preserved.
