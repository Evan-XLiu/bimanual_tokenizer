# Foundations Overview

Praxis turns a VLA policy implementation into an inspectable experimental
object. The important separation is:

```text
research variables  !=  surrounding contracts
```

A study changes a declared variable, then holds the relevant contracts fixed
long enough for the result to mean something.

## The Four Principles

| Principle | What it prevents | Repo mechanism |
| --- | --- | --- |
| Open coordinates | private one-off pipelines with hidden differences | public policy cells in `src/praxis/policies/catalog.py` |
| Shared contracts | feature, horizon, normalization, and artifact assumptions hidden in scripts | `PolicyIO`, `policy_io.json`, `norm_stats.pt`, codec sidecars |
| Axis-local ownership | objective, codec, and backbone details leaking across unrelated code | modules that own their axis-specific behavior |
| Evaluation boundary | simulator setup and metrics becoming policy variables | `PolicyEvalAdapter` plus `praxis-eval` benchmark ownership |

## The Main Interfaces

| Interface | Why it matters |
| --- | --- |
| Policy cell name | states the objective, action representation, and backbone being studied |
| `PolicyIO` | records policy-facing observations, actions, horizons, and normalization |
| Codec checkpoint | fixes the action-token language used by BAR and KI |
| Policy checkpoint | carries weights plus the sidecars required for loading and evaluation |
| `PolicyEvalAdapter` | exposes Praxis policies to `praxis-eval` without moving benchmark logic into policy code |

## Reading Order

1. [Policy Cells](/foundations/policy-cells) for the comparison unit.
2. [PolicyIO](/foundations/policy-io) for data and checkpoint contracts.
3. [Evaluation Boundary](/foundations/evaluation-boundary) for rollout
   ownership.
4. [Source Layout](/foundations/source-layout) for where the modules live.
