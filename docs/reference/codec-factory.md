# Codec Factory

The codec factory lives at:

```text
src/praxis/codecs/factory.py
```

It owns:

- `make_codec(...)`;
- `register_codec(...)`;
- `available_codecs(...)`;
- pretrained codec loading;
- runtime-compatible config overrides;
- relative codec path resolution from policy checkpoints.

## Registry Key

Each codec is registered by:

```text
(kind, name)
```

Examples:

```text
("tokens", "oat")
("tokens", "fast")
("latents", "oat")
```

## Config Contract

Every codec config must include:

```yaml
kind: tokens
name: oat
```

Optional pretrained loading uses:

```yaml
pretrained_path: outputs/train_codecs/.../pretrained_model
```

## Available Codecs

```python
from praxis.codecs.factory import available_codecs

print(available_codecs())
print(available_codecs(kind="tokens"))
```

## Checkpoint Sidecars

`BaseCodec.save_pretrained(...)` writes:

```text
policy_io.json
norm_stats.pt
```

Those sidecars are part of the codec interface. A tokenizer checkpoint without action normalization stats is not valid for BAR/KI policy construction.
