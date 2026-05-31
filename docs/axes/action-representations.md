# Action Representations

The action-representation axis defines the policy's action language. For a
normalized action chunk $a_{1:H}\in\mathbb{R}^{H\times D}$, a token codec
defines:

$$
z_{1:L}=E_c(a_{1:H}),\qquad
\hat{a}_{1:H}=D_c(z_{1:L}).
$$

The codec is not only a reconstruction model. It defines token length,
vocabulary, prefix meaning, invalid-decode behavior, and the distribution a VLM
policy must model.

## Evaluation Criteria

| Criterion | Question |
| --- | --- |
| rate | how many symbols represent one action chunk? |
| distortion | how much control-relevant motion is lost after decode? |
| modelability | how hard is the induced token sequence for BAR or KI? |
| total decode | do arbitrary supported token arrays decode safely? |
| prefix semantics | do partial prefixes produce meaningful actions? |
| schedule compatibility | does the representation match token-wise, blockwise, or parallel decoding? |

Low reconstruction error is necessary but not sufficient. A codec can
reconstruct demonstrations well while producing a sequence distribution that is
hard for a policy to predict or unsafe to decode under model samples.

## Implemented Codecs

| Codec | Structure | Role |
| --- | --- | --- |
| Binning | scalar action buckets | transparent discretization baseline |
| FAST | DCT coefficients plus subword compression | deterministic temporal-compression baseline |
| OAT | ordered latent slots | learned ordered action-token representation |
| BOAT | ordered slots matched to a BAR block schedule | representation-policy coupling for blockwise BAR |
| QueST | temporal autoencoder latents | learned temporal-compression comparison |
| ActionCodec | Perceiver-style latent groups with vector quantization | learned VQ comparison against FSQ-style codecs |

## Codec Lifecycle

The shared lifecycle is:

```text
build codec from config
  -> attach PolicyIO
  -> attach normalization stats
  -> fit or train reconstruction objective
  -> save_pretrained(...)
  -> load from policy.action_tokenizer.pretrained_path
```

The codec artifact is the object a BAR or KI policy consumes. If the codec is
learned, the policy consumes the saved tokenizer, not a fresh untrained
instance.

## Codec-Specific Notes

| Codec | Notes for studies |
| --- | --- |
| Binning | Total and transparent, but sequence length can be long and prefix semantics are weak. |
| FAST | Compact deterministic spectral representation; variable-length behavior and decode assumptions should be recorded. |
| OAT | Ordered prefixes are trained to be meaningful, which makes it useful for budgeted or blockwise token prediction. |
| BOAT | Block-aware ordering makes it a BAR-specific representation-policy hypothesis. |
| QueST | Learned compact latents test reconstruction-oriented temporal compression as a policy language. |
| ActionCodec | VQ latents test a different quantization family and one-shot or low-depth token grouping. |

## Compatibility

| Method | Compatible action representations |
| --- | --- |
| BAR | Binning, FAST, OAT, BOAT, QueST, ActionCodec |
| KI | Binning, FAST, OAT, QueST, ActionCodec |
| FM | none |

## Implementation

| Responsibility | Files |
| --- | --- |
| codec factory | `src/praxis/codecs/factory.py` |
| shared codec lifecycle | `src/praxis/codecs/core/` |
| token codecs | `src/praxis/codecs/tokens/` |
| latent autoencoders | `src/praxis/codecs/latents/` |
| codec configs | `src/praxis/configs/codecs/` |

## Action-Representation Comparison Checklist

Before comparing codecs, declare:

| Contract | Example |
| --- | --- |
| objective | BAR or KI |
| backbone | PaliGemma2, Qwen3-VL, or SmolVLM |
| action scale | `norm_stats.pt` source |
| action horizon | codec and policy horizon |
| decode behavior | invalid samples, variable length, prefix decode |
| metrics | reconstruction MSE, validation CE, closed-loop success |
