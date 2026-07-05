# Add A Backbone

Add a backbone when the research idea changes the VLM family or observation
prefix construction.

## Files

Shared VLM-family code:

```text
src/praxis/policies/backbones/<name>/
```

Method-specific adapters:

```text
src/praxis/policies/bar/backbones/<name>/
src/praxis/policies/fm/backbones/<name>/
src/praxis/policies/ki/backbones/<name>/
```

Hydra fragments:

```text
src/praxis/configs/policy/{bar,fm,ki}/backbone/<name>.yaml
```

## What The Backbone Owns

The backbone module owns:

- model and processor loading;
- image preprocessing;
- prompt and task-text formatting;
- state-prefix handling;
- attention convention;
- hidden-state shape;
- cache behavior;
- fine-tuning and LoRA target choices.

## Public Cells

Add the backbone to the catalog only after construction works through the
factory and at least one method-specific adapter exists.

## Tests

Add tests for:

- preprocessing shape and key handling;
- method-specific prefix output shape;
- policy factory resolution;
- at least one public cell smoke construction path.
