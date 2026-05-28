# PolicyIO Contract

`PolicyIO` is the data contract shared by dataset loading, action-codec
training, policy training, checkpoint loading, serving, and evaluation. It
records the feature-level information needed to interpret tensors:

| Field class | Examples |
| --- | --- |
| Observation schema | image keys, state keys, image shapes, state dimensions |
| Action schema | action key, action dimension, action horizon |
| Temporal schema | observation horizon, action horizon, dataset fps |
| Normalization schema | per-feature normalization mode |

The numerical normalization tensors are stored in `norm_stats.pt`; the structural
schema is stored in `policy_io.json`.

## Normalized Action Space

Let $a\in\mathbb{R}^{H\times D}$ be an action chunk from the dataset. Praxis
policies and codecs operate on normalized actions $\tilde{a}$.

For min/max normalization:

$$
\tilde{a}
=2\cdot {a-a_{\min}\over a_{\max}-a_{\min}}-1.
$$

For mean/std normalization:

$$
\tilde{a}={a-\mu\over\sigma}.
$$

The inverse map is part of the checkpoint semantics. A token sequence or
continuous action prediction is not meaningful without the normalization
statistics used to train the corresponding component.

## Checkpoint Sidecars

Praxis checkpoint artifacts are expected to include:

```text
policy_io.json
norm_stats.pt
```

Codec checkpoints require the same sidecars because a codec is a learned or
fitted representation of normalized action chunks. Token-based BAR and KI
policies therefore require:

```text
policy.action_tokenizer.pretrained_path=/absolute/path/to/codec/pretrained_model
```

This is not optional bookkeeping. It prevents a policy from decoding action
tokens under a different action scale, horizon, or feature schema from the codec
that produced them.

## Failure Modes Captured By The Contract

Many VLA failures present as model failures while actually being contract
failures. `PolicyIO` is the first object to inspect for:

| Symptom | Contract field to check |
| --- | --- |
| Incorrect action dimension | action feature shape |
| Policy expects state but receives none | observation state schema |
| Image preprocessing mismatch | image key and spatial shape |
| Tokenizer trained on a different action horizon | action horizon |
| Large action clipping rate | normalization mode and `norm_stats.pt` |

## Evaluation Adapter

`PolicyEvalAdapter` is the Praxis-side bridge to `praxis-eval`. It converts
evaluator observations into a Praxis batch, calls
`BasePolicy.predict_action(...)`, and returns numpy actions in the evaluator
convention.

The adapter is intentionally thin. Benchmark-specific task selection,
environment construction, action validation, metrics, and media artifacts remain
in `praxis-eval`; policy construction and checkpoint interpretation remain in
Praxis.
