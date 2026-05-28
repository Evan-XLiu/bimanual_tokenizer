# Binning Codec

Binning is the simplest action-token codec. It discretizes every scalar action value independently.

Implementation:

```text
src/praxis/codecs/tokens/bin/
```

## Math

Given a normalized action value $\tilde{a}$ clipped to $[a_{\min}, a_{\max}]$, define $K$ bin centers:

$$
c_k = a_{\min} + k {a_{\max} - a_{\min} \over K-1},
\qquad k=0,\ldots,K-1
$$

Tokenization chooses the bucket index:

$$
\tau_k = {c_k + c_{k+1} \over 2},
\qquad
z = \operatorname{bucketize}(\tilde{a}; \tau)
$$

Detokenization maps the token back to its center:

$$
\hat{a} = c_z
$$

For an action chunk $a_{1:H} \in R^{H \times D}$, Binning emits:

$$
L = H \cdot D
$$

tokens, flattened over time and action dimension.

## Block Schedule

Binning exposes:

```text
block_schedule = (action_dim,) * action_horizon
```

That means BAR can predict one timestep's action dimensions as a block.

## Algorithm Sketch

The implementation is intentionally close to the math:

```python
# src/praxis/codecs/tokens/bin/tokenizer.py
centers = torch.linspace(min_value, max_value, num_bins)
thresholds = (centers[:-1] + centers[1:]) / 2
tokens = torch.bucketize(actions.clamp(min_value, max_value), thresholds)
reconstructed = centers[tokens]
```

This makes Binning a useful first check for action normalization: if the
clamping rate is high or reconstruction is poor, the issue is visible before
any neural codec is involved.

## Properties

- deterministic;
- no neural codec training;
- transparent scalar quantization;
- useful sanity baseline;
- exposes action-scale problems quickly.

## Limitations

- token length grows as $H \cdot D$;
- every dimension is quantized independently;
- no learned temporal compression;
- reconstruction quality depends directly on bin count and normalization range.

## Experimental Role

Use Binning as a deterministic baseline. If a learned codec fails while Binning
works, inspect the learned codec training path and normalization sidecars. If
Binning also fails, inspect action normalization and policy optimization before
attributing the failure to the codec family.
