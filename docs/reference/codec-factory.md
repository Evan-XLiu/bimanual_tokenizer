# Codec Factory

The codec factory constructs action representations from config or checkpoints.

## Registry Keys

Codec registry keys are `(kind, name)` pairs:

```text
tokens/oat
tokens/fast
tokens/bin
latents/oat
```

## Main Functions

| Function | Role |
| --- | --- |
| `available_codecs(kind=None)` | list registered codecs |
| `make_codec(codec_cfg)` | construct from config or checkpoint |
| `register_codec(name, import_path, kind=...)` | add an external codec |
| `codec_path_base(path)` | resolve relative pretrained paths against a base |

## Pretrained Loading

When `pretrained_path` is set, the factory loads the source codec and validates
runtime overrides before transplanting state.

Relative paths are resolved in this order:

1. current working directory;
2. active `codec_path_base`;
3. raw string as a Hub id.

## Source

| Responsibility | File |
| --- | --- |
| registry and construction | `src/praxis/codecs/factory.py` |
| shared codec lifecycle | `src/praxis/codecs/core/` |
| token codecs | `src/praxis/codecs/tokens/` |
| latent codecs | `src/praxis/codecs/latents/` |
