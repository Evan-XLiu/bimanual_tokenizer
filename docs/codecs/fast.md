# FAST Codec

FAST is a deterministic action tokenizer based on temporal DCT coefficients plus BPE compression.

Implementation:

```text
src/praxis/codecs/tokens/fast/
src/praxis/policies/bar/codecs/fast/
```

## Math

Given normalized actions:

$$
a_{1:H,d}
$$

FAST applies a DCT-II along the time axis:

$$
\alpha_{k,d}
= w_k \sum_{t=1}^{H} a_{t,d}
  \cos\left({\pi \over H}(t - 1/2)k\right)
$$

with orthonormal scaling:

$$
w_0=\sqrt{1/H},
\qquad
w_k=\sqrt{2/H}\quad k>0.
$$

The coefficients are quantized:

$$
q_{k,d} = \operatorname{round}(s \alpha_{k,d})
$$

The quantized coefficients are shifted to nonnegative integer codes, converted to a character string, and encoded by a byte-level BPE tokenizer:

$$
z_{1:L} = \operatorname{BPE}(\operatorname{chars}(q))
$$

Detokenization reverses the path:

$$
z_{1:L}
\to \hat{q}
\to \hat{\alpha}
\to \hat{a}_{1:H}
$$

using inverse DCT.

## Variable Length

FAST emits variable-length token sequences. Praxis treats this as a real policy behavior:

- the codec declares `variable_length_codes=True`;
- BAR FAST has codec-owned decode/failure helpers;
- EOS and strict/lenient decode behavior are policy-side decisions, not generic tokenizer behavior.

## Algorithm Sketch

FAST has a deterministic signal-processing front end and a learned BPE string
codec:

```python
# src/praxis/codecs/tokens/fast/tokenizer.py
coeff = dct(actions, dim=time)
q = torch.round(scale * coeff).to(torch.int64)
chars = int_coefficients_to_chars(q)
tokens = bpe.encode(chars)

decoded_chars = bpe.decode(tokens)
q_hat = chars_to_int_coefficients(decoded_chars)
actions_hat = idct(q_hat.float() / scale, dim=time)
```

The variable-length BAR decode helpers live under
`src/praxis/policies/bar/codecs/fast/` because stop/failure policy is part of
the policy interface, not the tokenizer lifecycle.

## Properties

- temporal compression before tokenization;
- deterministic transform;
- BPE can shorten common coefficient patterns;
- useful baseline against learned tokenizers.

## Limitations

- BPE assets are checkpoint-owned;
- decode can fail if token sequences do not decode to the expected coefficient length;
- variable-length behavior makes policy stop rules important;
- DCT/BPE statistics may not transfer cleanly across action distributions.

## Experimental Role

Use FAST when the comparison needs deterministic temporal compression without a
neural action autoencoder. It is the relevant baseline for asking whether a
learned codec is necessary for a given dataset and policy method.
