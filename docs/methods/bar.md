# BAR Objective

BAR is the Praxis policy objective that directly models action tokens. One
example policy cell is:

```text
bar_oat_pali
```

BAR is a blockwise autoregressive action-token policy. It converts continuous
action chunks into discrete tokens, trains the VLM with schedule-aware
action-token cross entropy, and detokenizes generated tokens back to continuous
actions.

## Data Model

Let a normalized action chunk be:

$$
a_{1:H} \in \mathbb{R}^{H \times D}
$$

An action codec encodes the chunk into tokens:

$$
T_{1:L} = \mathcal{T}(a_{1:H})
$$

The VLM sees:

```text
observation prefix + action-token prompt/suffix
```

and learns to predict action tokens.

## Block Schedule

BAR changes the unit of generation from one token to a schedule-defined block.
Let the token horizon be $L$ and let prefix boundaries be:

$$
0=b_0<b_1<\cdots<b_S=L.
$$

Stage $s$ appends:

$$
G_s=T_{b_{s-1}+1:b_s},
\qquad
g_s=b_s-b_{s-1}.
$$

The schedule controls the number of sequential policy calls. Token-wise AR is
$g_s=1$ for all stages, so it needs $L$ calls. One-shot decoding is $S=1$ and
$g_1=L$. A power-of-two schedule such as:

$$
(g_1,g_2,g_3,g_4,\ldots)=(1,1,2,4,\ldots)
$$

reduces sequential depth from $O(L)$ to $O(\log L)$ when the tokenizer is compatible with those blocks.

## Objective

For a unit token schedule, the objective is ordinary next-token cross entropy:

$$
L_{\mathrm{CE}}
= - \sum_{i=1}^{L} m_i \log p_\theta(T_i \mid o, T_{<i})
$$

where $m_i$ masks valid supervised action positions.

For a block schedule, Praxis uses the block analogue of shifted teacher forcing.
With $g_0=0$, the stage-$s$ input is:

$$
X^{(s)}
=
T_{1:b_{s-1}}
\oplus
\langle \mathrm{MASK} \rangle^{g_s-g_{s-1}}.
$$

The final $g_s$ prediction positions are supervised to produce the current block
$G_s$ in parallel:

$$
L_{\mathrm{BAR}}
= - \sum_{s=1}^{S}
    \log p_\theta(G_s \mid o, X^{(s)}).
$$

The code implements this with block-shift teacher-forcing inputs and schedule-aware attention masks. Fresh positions use mask embeddings, inherited positions contain realized prefix context, and no prediction position sees its current-block ground truth.

## Inference

BAR inference is:

```text
build VLM prefix
  -> generate action tokens
  -> validate decode stop rule
  -> detokenize action tokens
  -> return first n_action_steps from action queue
```

Mathematically:

$$
\hat{T}_{1:L} = \hat{G}_1 \oplus \hat{G}_2 \oplus \cdots \oplus \hat{G}_S,
\qquad
\hat{a}_{1:H} = \mathcal{T}^{-1}(\hat{T}_{1:L})
$$

Sampling controls such as temperature and top-k apply in token space, before detokenization.

## Algorithm Sketch

The implementation is schedule-aware, but the outer loop is compact:

```python
# src/praxis/policies/bar/core/vlm/base.py
prefix = build_vlm_prefix(observation)
tokens = []
for stage in decode_schedule(block_schedule):
    ids = build_prediction_input(prefix, tokens, stage)
    logits = vlm(ids, attention_mask=stage.mask)
    next_block = sample_or_argmax(logits[:, stage.prediction_positions])
    tokens.extend(next_block)
    if stop_rule.accepts(tokens):
        break
actions = action_tokenizer.detokenize(tokens)
```

During training, `src/praxis/policies/core/vlm/training.py` builds the matching
teacher-forcing inputs and labels. During inference,
`src/praxis/policies/bar/core/vlm/decoding.py` owns the cached/blockwise decode
loop.

## Experimental Role

BAR is the most direct method for comparing action codecs. If the VLM directly
predicts a codec's tokens, differences between OAT, FAST, Binning, BOAT, QueST,
and ActionCodec appear in:

- token prediction difficulty;
- reconstruction error;
- decode robustness;
- closed-loop success.

The main caveat is representation/schedule compatibility. A tokenizer can be
compact and totally decodable while still being poorly matched to blockwise
generation if tokens inside a proposed block were trained as a strict serial
chain.

## Implementation

| Source path | Role |
| --- | --- |
| `src/praxis/policies/bar/variants.py` | Concrete BAR method x codec x backbone classes. |
| `src/praxis/policies/bar/core/vlm/base.py` | BAR VLM policy lifecycle and action-token inference. |
| `src/praxis/policies/bar/core/vlm/decoding.py` | Cached/blockwise token decode. |
| `src/praxis/policies/bar/core/decode_schedule.py` | Decode stage scheduling. |
| `src/praxis/policies/bar/core/stop.py` | Stop-mode validation and EOS/budget rules. |
| `src/praxis/policies/core/vlm/training.py` | Shared action-token CE path used by BAR and KI. |

## Supported BAR Matrix

BAR supports:

```text
codecs:    oat, acodec, boat, fast, bin, quest
backbones: pali, qwen, smol
```

Examples:

```text
bar_oat_pali
bar_fast_qwen
bar_boat_smol
bar_quest_pali
```

## Configuration

BAR public aliases live in:

```text
src/praxis/configs/policy/bar_<codec>_<backbone>.yaml
```

BAR method fragments live in:

```text
src/praxis/configs/policy/bar/backbone/
src/praxis/configs/policy/bar/codec/
```

The training recipe is:

```text
src/praxis/configs/train_recipe/policy/bar.yaml
```
