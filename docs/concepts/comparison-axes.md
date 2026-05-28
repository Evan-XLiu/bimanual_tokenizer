# Comparison Axes

Praxis is organized around controlled ablations of VLA policy design. A policy
cell is indexed by:

$$
(m,c,b)\in\mathcal{M}\times\mathcal{C}\times\mathcal{B},
$$

where $\mathcal{M}$ is the set of policy objectives, $\mathcal{C}$ is the set
of action codecs, and $\mathcal{B}$ is the set of VLM backbones.

One example cell used by the quickstart is:

```text
bar_oat_pali = BAR objective + OAT codec + PaliGemma2 backbone
```

## Axis 1: Policy Objective

The policy objective defines both training supervision and inference.

| Method | Representation | Training objective | Inference |
| --- | --- | --- | --- |
| BAR | discrete action tokens | next-token or blockwise token CE | generate tokens, then detokenize |
| FM | continuous actions | flow-matching velocity regression | integrate a continuous action flow |
| KI | tokens plus continuous expert | token CE plus insulated flow loss | condition a flow expert with VLM prefix states |

The objective axis asks whether the VLM should emit action symbols directly,
condition a continuous generator, or receive token supervision while a separate
expert produces continuous control.

## Axis 2: Action Representation

Let $a_{1:H}\in\mathbb{R}^{H\times D}$ be a normalized action chunk. A token
codec provides:

$$
z_{1:L}=E_c(a_{1:H}),\qquad
\hat{a}_{1:H}=D_c(z_{1:L}).
$$

The codec changes the target distribution learned by token-based methods:

| Quantity | Meaning |
| --- | --- |
| $L$ | token sequence length |
| $|V|$ | vocabulary size |
| $\mathbb{E}\|a-D_c(E_c(a))\|_2^2$ | reconstruction distortion |
| prefix semantics | whether partial token prefixes decode to useful actions |
| invalid-decode behavior | what happens to arbitrary policy-emitted token sequences |

A useful codec is not merely a low-error autoencoder. It must also expose a
sequence distribution that the policy can model and decode robustly.

## Axis 3: VLM Backbone

The backbone controls how images, state, and language become prefix states:

$$
h_b(o)=\mathrm{BackboneAdapter}_b(o).
$$

| Backbone | Comparison role |
| --- | --- |
| PaliGemma2 | compact VLM with a bidirectional prefix convention |
| Qwen3-VL | Qwen vision-language stack with causal prefix convention |
| SmolVLM | smaller VLM family and SmolVLA-style preprocessing |

The backbone axis asks how much the pretrained VLM prior matters when the action
representation and policy objective are fixed.

## Controlled Comparisons

A Praxis comparison should change one axis at a time:

```text
bar_oat_pali  vs  bar_oat_qwen      # backbone
bar_oat_pali  vs  bar_fast_pali     # codec
bar_oat_pali  vs  ki_oat_pali       # policy objective
```

The following quantities should remain fixed unless they are the subject of the
experiment:

- dataset and split;
- action key, dimension, horizon, and normalization;
- observation keys and image resolution;
- benchmark task selector;
- training recipe and optimizer family;
- evaluation budget and success metric.

This is the reason Praxis keeps the matrix explicit in
`src/praxis/policies/catalog.py`: public names should correspond to controlled
experimental cells rather than ad hoc combinations.
