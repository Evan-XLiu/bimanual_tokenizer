# Add A Codec

Add a codec when the research idea changes the action representation.

## Choose The Kind

| Kind | Use it when |
| --- | --- |
| `tokens` | the policy consumes discrete token ids |
| `latents` | the codec trains a learned latent autoencoder used by a token codec or diagnostic path |

Token-based BAR and KI policies need a `tokens` codec.

## Files

```text
src/praxis/codecs/tokens/<name>/
src/praxis/configs/codecs/tokens/<name>.yaml
tests/src/praxis/codecs/tokens/<name>/
```

If the codec uses a learned latent autoencoder, also add:

```text
src/praxis/codecs/latents/<name>/
src/praxis/configs/codecs/latents/<name>.yaml
tests/src/praxis/codecs/latents/<name>/
```

## Interface

The policy-facing tokenizer should support:

```python
tokens = tokenizer.tokenize(actions)
actions = tokenizer.detokenize(tokens)
vocab_size = tokenizer.codebook_size
```

Declare capabilities such as:

```text
max_code_seq_len
variable_length_codes
supports_prefix_decode
block_schedule
```

## Factory Wiring

Register the codec in `src/praxis/codecs/factory.py`:

```text
("tokens", "<name>"): _ref("tokens.<name>.tokenizer", "<ClassName>")
```

## Policy Matrix Wiring

Only add the codec to `src/praxis/policies/catalog.py` when at least one policy
family can use it as a public cell. Add Hydra fragments under:

```text
src/praxis/configs/policy/{bar,ki}/codec/<name>.yaml
```

## Tests

Add tests for:

- tokenize/detokenize shape and dtype;
- save/load sidecars;
- factory construction;
- policy construction when the codec enters a public cell;
- invalid-decode behavior if arbitrary token samples are possible.
