# Add An Action Codec

Add a codec when you want a new action representation to be compared against Binning, FAST, OAT, BOAT, QueST, and ActionCodec.

## Choose The Codec Kind

Praxis uses explicit codec kinds:

```text
tokens
latents
```

Use `tokens` when the policy will consume discrete token ids.

Use `latents` when the codec exposes continuous latent actions.

Most VLM action-token policies need a `tokens` codec.

## Token Codec Interface

A token codec must implement `BaseTokenizer`:

```python
class MyTokenizer(BaseTokenizer):
    def tokenize(self, actions):
        ...

    def detokenize(self, tokens, **kwargs):
        ...

    @property
    def codebook_size(self):
        ...
```

Also consider:

```python
@property
def max_code_seq_len(self):
    ...

@property
def variable_length_codes(self):
    ...

@property
def supports_prefix_decode(self):
    ...

@property
def block_schedule(self):
    ...
```

Those capabilities tell policies how the representation can be generated.

## Files To Add

For a deterministic token codec:

```text
src/praxis/codecs/tokens/mycodec/
  __init__.py
  config.py
  tokenizer.py
```

For a learned token codec backed by a latent autoencoder:

```text
src/praxis/codecs/latents/mycodec/
  __init__.py
  config.py
  autoencoder.py
  modeling.py

src/praxis/codecs/tokens/mycodec/
  __init__.py
  config.py
  tokenizer.py
```

## Factory Wiring

Register the codec in:

```text
src/praxis/codecs/factory.py
```

The registry key is:

```text
(kind, name)
```

For example:

```python
("tokens", "oat")
("latents", "oat")
```

## Config Wiring

Add standalone codec training config:

```text
src/praxis/configs/codecs/tokens/mycodec.yaml
```

If the codec should be available to BAR or KI policies, add policy fragments:

```text
src/praxis/configs/policy/bar/codec/mycodec.yaml
src/praxis/configs/policy/ki/codec/mycodec.yaml
```

Then add public policy aliases such as:

```text
src/praxis/configs/policy/bar_mycodec_pali.yaml
src/praxis/configs/policy/ki_mycodec_pali.yaml
```

## Policy Matrix Wiring

Update:

```text
src/praxis/policies/catalog.py
src/praxis/policies/bar/variants.py
src/praxis/policies/ki/variants.py
```

In `catalog.py`, update both the supported codec tuple for the relevant methods and `_CODEC_CLASS_PREFIXES`, because the catalog generates class names from the codec key.

Only add the codec to KI if KI can train and infer with it under the documented
KI objective. BOAT is the current example of a codec that is BAR-only.

## Tests

Add tests for:

- tokenizer round trip shape;
- action dimension validation;
- `codebook_size`;
- `max_code_seq_len`;
- `block_schedule` if used;
- `supports_prefix_decode` if used;
- factory construction;
- pretrained save/load sidecars;
- Hydra composition;
- policy catalog resolution if attached to BAR/KI.

## Review Checklist

- Token-only methods stay under `codecs/tokens`.
- Shared token/latent lifecycle stays in `codecs/core` only if it is truly shared.
- Policy-side stop/decode behavior stays in policy modules, not in tokenizer lifecycle.
- Codec checkpoints save `policy_io.json` and `norm_stats.pt`.
- The codec can be compared without adding benchmark-specific code.
