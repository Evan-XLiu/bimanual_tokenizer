# Existing-checkpoint VLA latency benchmark

Prepared 2026-09-28. This is a protocol, not measured GPU results. No retraining is required.

## Paper references checked

- OAT: Ordered Action Tokenization, arXiv:2602.04215v1, Section V-C1 / Table II. Reports action-token count and policy inference latency in milliseconds on one NVIDIA A100. The prefix variants change how many tokens are generated. Source: https://arxiv.org/html/2602.04215v1
- M2Tok: Multi-head Multi-codebook Discrete Action Tokenization for Vision-Language-Action Models, arXiv:2609.18259v1, Section 4.5 / Table 7. Reports VLA latency and inference frequency on one RTX 4090 in RoboTwin, including a separate vLLM configuration. Source: https://arxiv.org/html/2609.18259v1
- The retrieved sections do not fully specify all timing boundaries, warm-up counts, batch sizes, or synchronization details. The protocol below defines our own reproducible settings; it does not claim an exact reproduction of their harnesses. Do not compare absolute milliseconds across their hardware and ours.

## Primary benchmark

Measure a fresh full VLA action-chunk prediction:

    raw in-memory RGB observations + language + joint state
      -> preprocessing / device transfer
      -> vision and language/state prefix computation
      -> action-token generation with the native schedule and within-request KV cache
      -> tokenizer detokenization and action unnormalization
      -> ready-to-consume CPU float action chunk [1,32,14]

The outer wall-clock interval includes all these steps. Exclude dataset/video disk reads, checkpoint loading, process startup, RPC/network transport, simulator rendering/stepping and video recording. Report these separately if measuring deployment.

Use each checkpoint's training-time native schedule; no schedule changes:

| Model | Tokens/chunk | Native schedule | Generation blocks |
|---|---:|---|---:|
| BTT | 16 | [8,8] | 2 |
| OAT | 16 | [1] x 16 | 16 |
| QueST | 16 | [1] x 16 | 16 |
| Acodec | 16 | [16] | 1 |

Block count is explanatory metadata, not a measured speedup. Vision/prefix cost, batch shape, decoder architecture and implementation overhead also contribute.

## Suggested measurement settings

- Same dedicated GPU, CPU host, precision, backbone, attention backend, library versions and engine configuration across models. Record actual GPU name and checkpoint path.
- Batch size 1; three identical observation camera views and the same preprocessing resolution; action horizon 32 and action dimension 14.
- Model eval mode and inference_mode; greedy decoding with temperature=0 and top_k=1.
- Use the same fixed observation manifest for every model. Start with 200 observations drawn reproducibly from the existing Task-1 815-chunk manifest; use all 815 if affordable. Preserve language lengths and actual task distribution.
- Warm up for 30 full chunk predictions, excluded from statistics. Measure three repetitions of the selected manifest; rotate model order across repetitions. Warm-up is a suggested setting, not a copied paper protocol.
- Call torch.cuda.synchronize(device) immediately before and after the outer perf_counter interval. Do not time asynchronous GPU launches without waiting for completion.
- Call predict_action_chunk, with freshly constructed observations and a fresh per-request KV cache. Preserve within-request KV caching. Never count an action-queue pop as a new inference.
- Measure primary end-to-end latency with intermediate profiling disabled. In a separate diagnostic pass, enable internal stage timing; stage barriers can change total runtime.
- Log per-request latency, model, repetition, episode/frame ID, prompt length, output token count and output shape. Save mean, SD, median, p95 and optional episode-cluster-bootstrap mean CI. Do not discard failures or slow valid requests silently.

## Metrics and units

- Primary: mean full-chunk latency in ms, median and p95.
- Sequential batch-1 chunk frequency = number of completed chunk predictions / sum of measured seconds, equivalent to 1000 / mean latency_ms.
- Raw token emission rate = total generated tokens / sum of measured seconds; secondary because different token interfaces need not encode equivalent information.
- Do not report 32 / latency_seconds as the feedback/control frequency. It counts predicted action waypoints, not new observations or completed policy replans. The current deployment executes at most n_action_steps=16 from a 32-step prediction before replanning; any amortized action rate must be labeled explicitly.
- Speedup relative to OAT = mean OAT latency / mean method latency, under the same hardware/software/input protocol.
- Peak CUDA allocated memory can be reported with the baseline model footprint and measured incremental peak identified.

## Secondary tokenizer benchmark

Measure encode and decode separately at batch size 1, on the same [1,32,14] raw action chunks and their token IDs. Fix input/output residency explicitly; device-resident synchronized timings are appropriate for this microbenchmark.

Tokenizer encode is used to construct training/evaluation labels; it is not part of normal observation-to-action VLA inference. Tokenizer decode is part of VLA inference. A small encode+decode benchmark does not replace the full VLA latency benchmark.

## Existing code support

- src/praxis/policies/bar/core/vlm/decoding.py: run_blockwise_decode supports ROLLOUT_PROFILE=1 and records model_decode_seconds and detokenize_unnormalize_seconds. The model_decode interval includes observation/prefix handling performed by run_blockwise_token_decode; it is not exclusively post-prefill token decoding.
- src/praxis/policies/core/policy.py: queued prediction distinguishes new_inference from cached action consumption. Use predict_action_chunk directly for offline measurements.
- scripts/robotwin/policy/praxis_ar/deploy_policy.py: profiling separates RPC, observation preprocessing, environment stepping and other deployment costs.
- artifacts/robotwin_hopper/run_eval.sh: when ROLLOUT_PROFILE=1, initializes ROLLOUT_PROFILE_SERVER_TRACE. These traces can supplement the offline benchmark without rerunning the full 8-task suite.

## Suggested presentation

| Method | Tokens | Blocks | Latency mean (ms) | p50 | p95 | Chunks/s | Speedup vs OAT |
|---|---:|---:|---:|---:|---:|---:|---:|
| BTT | 16 | 2 | pending | pending | pending | pending | pending |
| OAT | 16 | 16 | pending | pending | pending | pending | 1x |
| QueST | 16 | 16 | pending | pending | pending | pending | pending |
| Acodec | 16 | 1 | pending | pending | pending | pending | pending |

Pair latency with the matched offline action/FK errors or existing rollout success rates. This supports an inference-efficiency/quality trade-off claim, primarily Q5; it does not on its own establish Q2 token learnability.
