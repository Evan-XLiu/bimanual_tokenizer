# Serve A Policy

`praxis-serve` loads a Praxis checkpoint and exposes it over gRPC through `praxis-remote`.

Serving a real VLM checkpoint is a compute-node job. Use `--cfg job` on a login
node to inspect the server config:

```bash
uv run praxis-serve \
  policy=bar_oat_pali \
  policy.pretrained_path=/tmp/policy \
  host=0.0.0.0 \
  port=50051 \
  --cfg job
```

## Start Server

```bash
POLICY_CKPT="$(find outputs/train_policy -path '*/checkpoints/latest/pretrained_model' -type d | sort | tail -n 1)"
test -n "${POLICY_CKPT}"

uv run praxis-serve \
  policy=bar_oat_pali \
  policy.pretrained_path="${POLICY_CKPT}" \
  host=0.0.0.0 \
  port=50051
```

Start the server inside an interactive or batch compute allocation.

## Evaluate Remotely

```bash
uv run praxis-eval \
  env.type=libero env.task=libero \
  eval.mode=remote_grpc \
  server_address=127.0.0.1:50051 \
  eval.output_dir=outputs/eval/remote_bar_oat_pali
```

## When To Use Serving

Use serving when:

- the simulator runtime and policy runtime should be isolated;
- the evaluator runs on another machine;
- you want one policy process to handle multiple evaluator calls;
- an external benchmark uses `praxis-remote`.

## Security Note

The default gRPC path is intended for trusted networks. Do not expose it as a public unauthenticated endpoint.
