# OAT Codec

OAT is a learned ordered action-token codec in Praxis.

Example policy:

```text
bar_oat_pali
```

Implementation:

```text
src/praxis/codecs/latents/oat/
src/praxis/codecs/tokens/oat/
```

## Representation

OAT learns an ordered latent representation of an action chunk, then quantizes each latent slot into a token. Its design target is the rate--distortion--modelability point where the token sequence is compact, totally decodable, and useful as a left-to-right prediction problem.

The high-level path is:

```text
actions -> OAT encoder -> ordered latents -> FSQ tokens -> OAT decoder -> actions
```

## Math

Let normalized actions be:

$$
a \in \mathbb{R}^{H \times D}
$$

The encoder writes the action chunk into $L$ learned register slots. Each register cross-attends to the whole action chunk, while register self-attention follows the configured schedule. For plain OAT, that schedule is unit causal:

$$
z_{1:L}=E_\phi(a,r_{1:L}),
\qquad
\mathrm{attend}(i,j)=1[j\leq i].
$$

Finite scalar quantization maps latents to token ids:

$$
T_i = q(z_i),
\qquad
\bar{z}_i = q^{-1}(T_i)
$$

The decoder reconstructs:

$$
\hat{a} = D_\theta(\bar{z}_{1:L})
$$

The token-training path uses reconstruction loss:

$$
L_{\mathrm{OAT}}
= \|D_\theta(q(E_\phi(a,r_{1:L})))-a\|_2^2.
$$

Latent autoencoder regularization belongs to the latent autoencoder path when configured; the current tokenized OAT path does not add tokenizer-owned auxiliary loss terms.

## Ordered Prefix Decode

OAT's key property is that latent slots are ordered. During training, nested dropout samples a prefix budget $K$, keeps the first $K$ tokens, and masks the suffix:

$$
\widetilde{T}_{1:L}
=
T_{1:K}
\oplus
\langle \mathrm{MASK}\rangle_{K+1:L}.
$$

The decoder must reconstruct the full action chunk from that partial budget:

$$
\hat{a}^{(K)}=D_\theta(\widetilde{T}_{1:L}).
$$

This trains the whole prefix rate--distortion curve:

$$
D(K)=\mathbb{E}\|a-\hat{a}^{(K)}\|_2^2.
$$

Short prefixes should capture coarse, high-impact motion; later tokens refine residual detail. Praxis exposes this through `decode_keep_k` and padded prefix detokenization.

## Algorithm Sketch

OAT is implemented as a latent autoencoder plus a token adapter:

```python
# src/praxis/codecs/tokens/oat/tokenizer.py
latents = oat_autoencoder.encode(normalized_actions)
tokens = finite_scalar_quantizer.codes(latents)

prefix_tokens = keep_first_k(tokens, decode_keep_k)
prefix_latents = finite_scalar_quantizer.decode(prefix_tokens)
actions_hat = oat_autoencoder.decode_prefix(prefix_latents, decode_keep_k)
```

The OAT latent model lives under `src/praxis/codecs/latents/oat/`; the token
adapter in `src/praxis/codecs/tokens/oat/` owns token ids, codebook size, and
prefix-decode capability declarations.

## Policy Interface

An OAT-backed BAR policy models:

$$
p_\theta(T_{1:L}\mid o)
=
\prod_{i=1}^{L}p_\theta(T_i\mid T_{<i},o).
$$

Plain OAT is AR-friendly for token-wise decoding. It is not by itself block-aware: grouping OAT tokens post hoc into larger BAR blocks can remove serial dependencies that the tokenizer was allowed to learn inside those groups.

## Properties

- learned temporal/action abstraction;
- fixed code horizon;
- prefix decode support when configured;
- learned ordered-token codec for BAR comparisons;
- uses finite scalar quantization rather than a large vector codebook.

## Limitations

- requires codec training;
- prefix decode depends on compatible latent dropout / masking configuration;
- vanilla OAT is token-wise ordered, not inherently blockwise;
- reconstruction quality and policy success can diverge, so evaluate both.

## Experimental Role

Use OAT when the experiment needs a learned ordered-token representation.
`bar_oat_pali` is a convenient first example because it pairs OAT with BAR and
PaliGemma2, but codec comparisons should be reported as explicit cells rather
than as deviations from a preferred recipe.
