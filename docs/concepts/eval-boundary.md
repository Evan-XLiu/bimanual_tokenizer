# System Boundary

Praxis and `praxis-eval` are separate packages. The split is part of the
experimental design: policy implementation should not depend on benchmark
runtime internals, and benchmark rollout should not depend on method-specific
policy classes.

## Responsibility Split

Praxis owns:

- policy classes and factories;
- action codecs and codec checkpoints;
- training loops and train recipes;
- checkpoint format and sidecars;
- local and remote serving;
- `PolicyEvalAdapter`.

`praxis-eval` owns:

- benchmark environment wrappers;
- task selectors;
- rollout loops;
- policy-facing observation and action contracts at the benchmark boundary;
- metrics, result artifacts, and media;
- simulator setup and verification helpers.

## Local Evaluation

Local evaluation follows:

```text
Praxis checkpoint
  -> praxis.policies.factory.make_policy(...)
  -> PolicyEvalAdapter
  -> praxis_eval.evaluate(...)
  -> results and media artifacts
```

Only `PolicyEvalAdapter` must satisfy the evaluator policy interface. The
underlying policy remains a Praxis object.

## Remote Evaluation

Remote evaluation isolates the simulator runtime from the policy runtime:

```text
praxis-serve
  -> praxis-remote transport
  -> praxis-eval RemotePolicy
  -> benchmark rollout
```

Use the remote path when policy inference and simulator execution belong on
different machines, different environments, or different accelerators.

## Boundary Contract

At the package boundary, the contract is:

```text
observation payload in
action array out
```

Praxis decides how a policy is constructed and how a checkpoint is interpreted.
`praxis-eval` decides how observations are produced, how actions are validated,
and how rollout metrics are computed. This separation keeps comparison across
backbones, codecs, and methods from being confounded by simulator integration
code.
