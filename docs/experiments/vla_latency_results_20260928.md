# Existing-checkpoint VLA latency — 2026-09-28

Completed PBS job 636642.hopper-m-02 on NVIDIA H200 (hopper-20). Full VLA offline inference, not closed-loop rollout. These results describe the checkpoints listed below; newly trained checkpoints are not automatically substituted.

## Protocol

Batch size 1; PaliGemma2 3B mix224, bfloat16 VLM, native codec dtype; PyTorch 2.7.1+cu126, SDPA and native KV cache, no vLLM or compile. Same 200 cached observations for all five models, three rounds with rotated model order, 30 warmup requests per model per round. Each output has 16 action tokens and shape [1,32,14]. CPU input transfer, model preprocessing/vision/prefill/generation, action decoding and float32 CPU output are timed with CUDA synchronization. Disk/video I/O, checkpoint loading, RPC and simulation are excluded. Every request generates a fresh chunk.

Mean ± 95% CI uses an episode-cluster bootstrap normal approximation (5,000 draws, seed 42). This interval characterizes the sampled episodes within this run, not uncertainty across hardware or independent jobs. Chunks/s is 1000 divided by mean milliseconds, not environment control frequency.

## Full VLA results

| Method | Decode blocks | Mean ± 95% CI (ms) | p50 (ms) | p95 (ms) | Chunks/s | Speedup vs OAT single |
|---|---:|---:|---:|---:|---:|---:|
| BTT | 2 | 56.83 ± 0.11 | 57.04 | 58.85 | 17.60 | 5.80× |
| Acodec | 1 | 38.53 ± 1.44 | 36.69 | 38.04 | 25.96 | 8.56× |
| OAT_single | 16 | 329.66 ± 0.47 | 329.37 | 337.33 | 3.03 | 1.00× |
| OAT_pow2 | 5 | 115.98 ± 0.29 | 115.79 | 120.27 | 8.62 | 2.84× |
| QueST | 16 | 328.98 ± 0.30 | 329.37 | 336.68 | 3.04 | 1.00× |

## Tokenizer-only timings

Device-resident inputs, 200 measurements per method after warmup. Encode is not part of normal observation-to-action VLA inference; decode is. These measurements exclude CPU transfers and are distinct from full VLA timing.

| Method | Encode mean (ms) | Decode mean (ms) |
|---|---:|---:|
| BTT | 2.360 | 2.663 |
| Acodec | 2.202 | 1.932 |
| OAT_single | 4.235 | 2.920 |
| OAT_pow2 | 4.182 | 2.903 |
| QueST | 2.475 | 2.541 |

## Validation and limitations

All 3,000 primary measurements have the same sample identities across methods and rounds; all 1,000 codec timing pairs are present. Recomputed means, p95, confidence intervals and speedups match saved summaries. Input-cache and executed-script SHA256 checks pass. The runner checks finite [1,32,14] outputs before recording each primary measurement. No slow measurements were removed.

OAT_pow2 uses the separately trained BOAT checkpoint with schedule [1,1,2,4,8]. It is not a decoding-only change to the OAT_single checkpoint. BTT uses [8,8], Acodec [16], and OAT_single/QueST [1] × 16. Speed differences reflect these complete systems and schedules, not only standalone tokenizer decoding.

Acodec has a long latency tail: maximum 243.21 ms; 7 of 600 measurements exceed 100 ms. Its round means are 37.20, 35.15 and 43.22 ms. All observations are retained; the cause is not established by the timing log. Median and p95 should accompany its mean. The episode bootstrap does not capture all round-to-round drift.

## Checkpoints

- BTT: `/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_bar_btt_pali_robotwin_new_30k/622038.hopper-m-02/train/checkpoints/latest/pretrained_model`
- Acodec: `/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_bar_acodec_pali_robotwin_new_step300000_30k/630823.hopper-m-02/train/checkpoints/best/step_18000/pretrained_model`
- OAT_single: `/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_bar_oat_pali_fft_new_2500/601387.hopper-m-02/train/checkpoints/best/step_13400/pretrained_model`
- OAT_pow2: `/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_bar_boat_pali_robotwin_new_30k/616455.hopper-m-02/train/checkpoints/best/step_30000/pretrained_model`
- QueST: `/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_bar_quest_pali_robotwin_new_30k/613537.hopper-m-02/train/checkpoints/best/step_26000/pretrained_model`

## Artifacts

Raw measurements: `outputs/vla_latency/20260928_native5/measurements.jsonl`. Full manifest, hardware record, CSV/JSON summaries and executed script snapshot are in the same directory.

Round means and tail diagnostics:

```json
{
  "BTT": {
    "count": 600,
    "max_ms": 61.663717,
    "over_100_ms": 0,
    "round_means_ms": {
      "0": 55.70188688499999,
      "1": 57.398351765,
      "2": 57.37997164
    }
  },
  "Acodec": {
    "count": 600,
    "max_ms": 243.214513,
    "over_100_ms": 7,
    "round_means_ms": {
      "0": 37.202337755,
      "1": 35.15437549,
      "2": 43.219219515000006
    }
  },
  "OAT_single": {
    "count": 600,
    "max_ms": 344.319896,
    "over_100_ms": 600,
    "round_means_ms": {
      "0": 329.611132385,
      "1": 327.63104555,
      "2": 331.75143625
    }
  },
  "OAT_pow2": {
    "count": 600,
    "max_ms": 127.694625,
    "over_100_ms": 600,
    "round_means_ms": {
      "0": 114.58774068000001,
      "1": 117.03057698999999,
      "2": 116.31000481999999
    }
  },
  "QueST": {
    "count": 600,
    "max_ms": 342.231482,
    "over_100_ms": 600,
    "round_means_ms": {
      "0": 323.43726119,
      "1": 330.985277115,
      "2": 332.52175076500004
    }
  }
}
```
