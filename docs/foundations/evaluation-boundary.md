# Evaluation Boundary

Evaluation is a boundary because benchmark rollout is part of the experiment.
A policy is interpreted through the observations it receives, the action
convention it must satisfy, the simulator runtime, task selection, and metrics.

## Responsibility Split

Praxis owns:

- policy construction and checkpoint loading;
- action codecs and policy inference;
- local adapter and remote serving;
- `PolicyEvalAdapter`.

`praxis-eval` owns:

- benchmark drivers;
- simulator dependencies;
- task selection;
- rollout;
- action validation;
- metrics and artifacts.

## Local Evaluation

Local evaluation keeps policy and simulator in one environment:

```text
Praxis checkpoint
  -> make_policy(...)
  -> PolicyEvalAdapter
  -> praxis_eval.evaluate(...)
  -> results and media
```

Use local evaluation when policy dependencies and simulator dependencies are
compatible.

## Remote Evaluation

Remote evaluation isolates the policy runtime:

```text
praxis-serve
  -> praxis-remote transport
  -> praxis-eval RemotePolicy
  -> benchmark rollout
```

Use this mode when the policy needs a different Python environment, machine, or
accelerator than the simulator.

## Adapter Contract

The policy-facing contract is intentionally narrow:

```text
reset episode state
observation payload in
batched action array out
```

Benchmark-specific semantics stay in `praxis-eval`; policy-specific loading
and inference stay in `praxis-vla`.
