# QueST Codec

QueST is a learned action-token codec with a temporal convolution / transformer autoencoder and finite scalar quantization.

Implementation:

```text
src/praxis/codecs/latents/quest/
src/praxis/codecs/tokens/quest/
```

## Math

QueST encodes normalized action chunks into a shorter latent sequence:

$$
y_{1:L} = f_{\mathrm{quest-enc}}(a_{1:H})
$$

Then finite scalar quantization produces tokens:

$$
z_i = q(y_i)
$$

The decoder reconstructs:

$$
\hat{a}_{1:H} = f_{\mathrm{quest-dec}}(q^{-1}(z_{1:L}))
$$

The current token-training objective is reconstruction:

$$
L_{\mathrm{QueST}}
= \| \hat{a}_{1:H} - a_{1:H} \|_2^2
$$

The tokenized QueST path does not add tokenizer-owned auxiliary loss terms.

## Structure

QueST validates horizon/downsampling compatibility. This matters because the temporal encoder reduces the action horizon before quantization.

The policy sees the resulting tokens as a fixed-length action sequence.

## Algorithm Sketch

QueST is useful to compare against OAT because both use FSQ but hide different
autoencoder structures behind the same token interface:

```python
# src/praxis/codecs/tokens/quest/tokenizer.py
latents = quest_autoencoder.encode(actions)       # temporal conv + transformer
tokens = finite_scalar_quantizer.codes(latents)

latents_hat = finite_scalar_quantizer.decode(tokens)
actions_hat = quest_autoencoder.decode(latents_hat)
```

The compatibility checks live in `src/praxis/codecs/latents/quest/`; they make
the horizon/downsampling assumptions fail early instead of producing malformed
token sequences.

## Properties

- learned temporal compression;
- finite scalar quantization;
- fixed-length token target for BAR/KI;
- useful comparison against OAT because both are learned FSQ tokenizers but use different autoencoder structure.

## Limitations

- requires codec training;
- horizon and downsampling must be compatible;
- no generic prefix decode capability in the current contract.

## Experimental Role

Use QueST when comparing learned tokenizers with different temporal latent
architectures. Relevant comparisons include:

```text
bar_oat_pali vs bar_quest_pali
ki_oat_pali vs ki_quest_pali
```

and inspect both reconstruction loss and closed-loop success.
