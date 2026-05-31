# Comparison Design

A Praxis study should start as a sentence:

```text
This study changes <axis> from <cell A> to <cell B>
while holding <contracts> fixed.
```

If the sentence is vague, the experiment is probably mixing variables.

## Examples

```text
bar_oat_pali  vs  bar_oat_qwen      # backbone
bar_oat_pali  vs  bar_fast_pali     # action representation
bar_oat_pali  vs  ki_oat_pali       # objective
fm_pali       vs  fm_qwen           # backbone without tokenizer
```

## Contracts To Declare

| Contract | Why it matters |
| --- | --- |
| dataset and split | changes demonstration distribution |
| action key, dimension, horizon | changes the target control problem |
| normalization stats | changes action scale and token meaning |
| observation keys | changes policy context |
| codec checkpoint | changes the action language |
| train recipe and budget | changes optimization evidence |
| benchmark task selector | changes rollout distribution |
| success metric | changes the reported outcome |
| local or remote mode | changes runtime isolation and state management |

## What Praxis Does Not Solve

Praxis does not choose random seeds, budgets, hyperparameter sweeps, task
splits, or success metrics for you. It gives those choices a place to be
declared, inspected, and reused.

## Minimum Study Record

For an internal experiment log or paper table, record:

| Field | Example |
| --- | --- |
| cells | `bar_oat_pali`, `bar_fast_pali` |
| changed coordinate | action representation |
| fixed contracts | dataset, action horizon, backbone, train recipe, evaluator |
| artifact paths | codec checkpoint, policy checkpoint, eval output dir |
| metric | task success, validation CE, reconstruction MSE |
| notes | invalid-decode handling, remote/local mode, seed policy |
