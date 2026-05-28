# BOAT Codec

BOAT is a BAR-focused OAT specialization with a coarse-to-fine block schedule.

Implementation:

```text
src/praxis/codecs/tokens/boat/
src/praxis/policies/bar/variants.py
src/praxis/policies/bar/core/
```

BOAT is currently supported by BAR, not KI.

## Representation

OAT gives ordered latent slots. BOAT changes the register dependency pattern so the tokenizer and BAR schedule agree. Earlier blocks remain causal context, but tokens inside the same generated block are learned as parallel latent slots:

```text
coarse action plan -> finer refinement -> full latent budget
```

## Math

Let OAT produce latent tokens:

$$
T_{1:L}
$$

BOAT requires $L$ to be a power of two:

$$
L = 2^m
$$

It derives a nondecreasing coarse-to-fine schedule:

$$
(g_1,g_2,\ldots,g_S)=(1,1,2,4,\ldots)
$$

with cumulative prefix boundaries:

$$
b_s = \sum_{j=1}^{s} g_j.
$$

Those boundaries are the only valid BOAT prefix decode budgets.

BAR can then decode at:

$$
T_{1:b_1}, T_{1:b_2}, \ldots, T_{1:L}
$$

and the decoder only accepts `decode_keep_k` values at those boundaries.

The register attention mask is block causal:

$$
\mathrm{attend}(i,j)=1[\ell(j)<\ell(i)] \lor 1[i=j],
$$

where $\ell(i)$ is the schedule level containing register $i$. A register can attend to earlier blocks and itself, but not to other registers in its own block. This is the representation change that makes the block a legitimate parallel prediction target.

## Blockwise Motivation

Post-hoc OAT plus BAR changes only the policy schedule. BOAT also changes the tokenizer training structure. That distinction matters because BAR assumes every token in $G_s$ can be predicted together from the realized prefix:

$$
p_\theta(G_s\mid o,T_{1:b_{s-1}}).
$$

If the tokenizer learned serial dependencies inside $G_s$, parallel prediction removes information the representation expected. BOAT discourages those within-block dependencies while preserving ordered prefix semantics across blocks.

## Algorithm Sketch

BOAT reuses the OAT tokenizer shape but changes the register schedule:

```python
# src/praxis/codecs/tokens/boat/config.py
block_schedule = (1, 1, 2, 4, ..., L / 2)
valid_prefixes = cumulative_sum(block_schedule)

# src/praxis/codecs/tokens/boat/tokenizer.py
latents = boat_autoencoder.encode(actions, register_schedule=block_schedule)
tokens = fsq.codes(latents)
assert decode_keep_k in valid_prefixes
actions_hat = boat_autoencoder.decode_prefix(tokens[:decode_keep_k])
```

BAR then uses the same `block_schedule` when it builds the decode stages. This
is the reason BOAT is a representation-policy pair rather than an OAT inference
variant.

## Properties

- retains OAT's learned ordered latents;
- makes action generation explicitly coarse-to-fine;
- aligns tokenizer register dependencies with BAR block prediction;
- exposes decode budget as a policy-side control;
- useful for studying early-exit or adaptive-token action prediction.

## Limitations

- BAR-specific in the current repo;
- requires power-of-two latent horizon;
- decode prefixes must land on schedule boundaries;
- introduces a tighter representation-policy coupling than plain OAT.

## Experimental Role

Use BOAT when the research question concerns budgeted or coarse-to-fine
action-token generation. Start from OAT, then test whether BOAT's schedule
improves token efficiency or closed-loop behavior under BAR.
