# Codec Interface

Action codecs define the representation layer between continuous robot actions
and policy prediction targets. For a normalized action chunk
$a_{1:H}\in\mathbb{R}^{H\times D}$, a token codec provides:

$$
z_{1:L}=E(a_{1:H}),\qquad
\hat{a}_{1:H}=D(z_{1:L}),
$$

where each $z_i$ is a discrete token id. Continuous latent codecs use the same
checkpoint and normalization lifecycle, but token-based BAR and KI policies
consume token codecs.

## Evaluation Criteria

The codec axis changes the statistical problem presented to the policy. The
important criteria are:

| Criterion | Definition | Consequence for policies |
| --- | --- | --- |
| Rate | token length $L$ | controls serial depth, context length, and inference latency |
| Distortion | $\mathbb{E}\|a-D(E(a))\|_2^2$ | controls executable action fidelity |
| Modelability | difficulty of modeling $p_\theta(z_{1:L}\mid o)$ | controls token CE and rollout robustness |
| Total decode | whether arbitrary supported token arrays decode | controls behavior under imperfect policy outputs |
| Prefix semantics | whether partial prefixes produce meaningful actions | controls blockwise and budgeted decoding |

Low reconstruction error is necessary but not sufficient. A codec can
reconstruct demonstrations accurately while producing token sequences that are
long, invalid under samples, or poorly ordered for autoregressive prediction.

## Supported Codecs

| Codec | Representation | Quantizer | Primary experimental role |
| --- | --- | --- | --- |
| Binning | scalar action buckets | deterministic buckets | normalization and scalar-discretization baseline |
| FAST | DCT coefficients plus BPE | deterministic coefficient quantization and BPE | deterministic temporal-compression baseline |
| OAT | ordered latent slots | finite scalar quantization | learned ordered action-token codec |
| BOAT | OAT-style slots with block schedule | finite scalar quantization | BAR-aligned coarse-to-fine tokenization |
| QueST | temporal autoencoder latents | finite scalar quantization | learned temporal-compression comparison |
| ActionCodec | Perceiver latent slots | vector quantization | learned VQ comparison against FSQ codecs |

## Shared Interface

Token codecs inherit from `BaseTokenizer`. The policy-facing operations are:

```python
tokens = tokenizer.tokenize(actions)
actions = tokenizer.detokenize(tokens)
vocab_size = tokenizer.codebook_size
```

Codecs also declare capabilities used by policy methods:

```python
max_code_seq_len
variable_length_codes
supports_prefix_decode
block_schedule
```

The shared lifecycle owns:

- `PolicyIO` attachment;
- action normalization statistics;
- `save_pretrained(...)` and `from_pretrained(...)`;
- `policy_io.json` and `norm_stats.pt`;
- optimizer parameters for trainable codecs.

## Generic Learned-Codec Objective

OAT, BOAT, QueST, and ActionCodec use learned autoencoders. A common form is:

$$
y=f_{\mathrm{enc}}(a),\qquad
\bar{y}=q^{-1}(q(y)),\qquad
\hat{a}=f_{\mathrm{dec}}(\bar{y}).
$$

The reconstruction objective is:

$$
L_{\mathrm{reconst}}
= {1\over BHD}\sum_{b,h,d}
  (\hat{a}_{b,h,d}-a_{b,h,d})^2.
$$

Family-specific quantizer terms may be added. ActionCodec includes a VQ
commitment term; the current tokenized OAT, BOAT, and QueST paths use
reconstruction as the token-training loss.

## Policy Compatibility

| Method | Codec compatibility |
| --- | --- |
| BAR | Binning, FAST, OAT, BOAT, QueST, ActionCodec |
| KI | Binning, FAST, OAT, QueST, ActionCodec |
| FM | no action tokenizer |

BOAT is currently BAR-only because its block schedule is a representation-policy
coupling, not a generic tokenizer feature for all methods.

## Minimal Pseudocode

```python
policy_io = build_policy_io(dataset)
codec = make_codec(cfg.codec)
codec.set_policy_io(policy_io)
codec.set_norm_stats(action_norm_stats)

if codec.has_trainable_parameters:
    train_reconstruction(codec, dataset)
elif codec.requires_fit:
    codec.fit(dataset_actions)

codec.save_pretrained(output_dir / "pretrained_model")
```

The resulting `pretrained_model` path is the object referenced by
`policy.action_tokenizer.pretrained_path` in BAR and KI policies.
