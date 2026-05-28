# Add A VLM Backbone

Add a backbone when you want a new pretrained VLM family to participate in the same policy-method and action-codec comparisons as Pali, Qwen, and Smol.

Do not add backbone-specific behavior to global policy core unless more than one backbone family needs it.

## Target Interface

A backbone must provide enough behavior for supported policy methods to:

- load the VLM and processor;
- build policy-facing VLM inputs from observations and instructions;
- expose prefix hidden-state or KV-cache geometry;
- declare prefix attention convention;
- freeze vision modules when configured;
- support LoRA and optimizer grouping where applicable.

## Files To Add

Use a short lowercase backbone key, for example `newvlm`.

Shared backbone modules:

```text
src/praxis/policies/backbones/newvlm/
  vlm.py
  prefix.py
```

Method-specific modules:

```text
src/praxis/policies/bar/backbones/newvlm/
  newvlm_base.py
  newvlm_config.py

src/praxis/policies/fm/backbones/newvlm/
  newvlm_base.py
  newvlm_config.py

src/praxis/policies/ki/backbones/newvlm/
  newvlm_base.py
  newvlm_config.py
```

Hydra fragments:

```text
src/praxis/configs/policy/bar/backbone/newvlm.yaml
src/praxis/configs/policy/fm/backbone/newvlm.yaml
src/praxis/configs/policy/ki/backbone/newvlm.yaml
```

Public aliases:

```text
src/praxis/configs/policy/bar_oat_newvlm.yaml
src/praxis/configs/policy/fm_newvlm.yaml
src/praxis/configs/policy/ki_oat_newvlm.yaml
```

Add the full supported matrix, not only OAT, when the backbone is ready for all relevant codecs.

## Registry Wiring

Update:

```text
src/praxis/policies/catalog.py
```

Add the new backbone to `POLICY_BACKBONES` and `_BACKBONE_CLASS_NAMES`.

Then update:

```text
src/praxis/policies/bar/variants.py
src/praxis/policies/fm/variants.py
src/praxis/policies/ki/variants.py
```

The variants file keeps public policy classes together while method and backbone modules own real behavior.

## Tests

At minimum, add or update tests that prove:

- the catalog includes the new names;
- `available_policies()` resolves the new policy classes;
- Hydra aliases compose;
- backbone prefix adapter shape/attention geometry is correct;
- one fake or tiny forward path works for each supported method.

Use Pali, Qwen, and Smol tests as references.

## Review Checklist

- The new backbone declares its own prefix attention convention.
- Method-specific behavior stays under `policies/{bar,fm,ki}/backbones/newvlm`.
- Shared code under `policies/backbones/newvlm` is truly shared by at least two methods.
- No benchmark env code is added to Praxis.
- No benchmark runtime code is added to the backbone implementation.
