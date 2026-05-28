# ActionCodec / ACodec

ActionCodec, named `acodec` in configs, is a learned action-token codec using a Perceiver-style autoencoder and vector quantization.

Implementation:

```text
src/praxis/codecs/latents/acodec/
src/praxis/codecs/tokens/acodec/
```

## Math

The encoder maps actions to latent slots:

$$
y_{1:L} = f_{\mathrm{perc-enc}}(a_{1:H})
$$

Vector quantization maps each latent to a codebook entry:

$$
z_i = \arg\min_j \| y_i - e_j \|_2^2
$$

and uses the selected embeddings:

$$
\bar{y}_i = e_{z_i}
$$

The decoder reconstructs:

$$
\hat{a}_{1:H} = f_{\mathrm{perc-dec}}(\bar{y}_{1:L})
$$

Praxis uses EMA vector quantization for ACodec. The codec training loss is reconstruction plus the quantizer commitment term; codebook updates happen through EMA, so the source `codebook_loss` term is zero.

$$
L_{\mathrm{ACodec}}
= \| \hat{a} - a \|_2^2
 + \beta \| y - \mathrm{sg}(e_z) \|_2^2
$$

where `sg` means stop-gradient.

## Block Schedule

ACodec exposes one parallel block:

```text
block_schedule = (latent_horizon,)
```

This means BAR predicts the action-code sequence as one block rather than a timestep-aligned schedule.

## Algorithm Sketch

ACodec uses vector-quantized latent slots:

```python
# src/praxis/codecs/tokens/acodec/tokenizer.py
latents = perceiver_encoder(actions)
tokens, embeddings, commitment = vector_quantizer(latents)
actions_hat = perceiver_decoder(embeddings)

loss = reconstruction_loss(actions_hat, actions) + beta * commitment
```

The source code uses EMA codebook updates, so codebook movement is owned by the
quantizer state update rather than a nonzero `codebook_loss` term.

## Properties

- learned Perceiver-style latent compression;
- vector quantization with a codebook;
- fixed-length token target;
- useful contrast against FSQ-based OAT and QueST.

## Limitations

- requires codec training;
- VQ codebook dynamics can matter;
- no prefix decode in the current contract;
- reconstruction and rollout quality should be checked separately.

## Experimental Role

Use ACodec when the comparison question concerns vector-quantized learned action
latents rather than scalar-quantized latents. Relevant comparisons include:

```text
bar_oat_pali vs bar_acodec_pali
ki_oat_pali vs ki_acodec_pali
```
