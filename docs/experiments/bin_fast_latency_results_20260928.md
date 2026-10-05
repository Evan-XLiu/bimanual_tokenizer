# BIN and FAST checkpoints and latency — 2026-09-28

PBS job 636884.hopper-m-02 completed: 1,200 full VLA requests; 600 per method; same 200 observations as five-method benchmark; 3 rounds; 30 warmup requests per model per round. H200 on hopper-18, batch size 1, PyTorch eager / SDPA / native KV cache, bfloat16 VLM. Offline CPU-observation-to-CPU-action timing; excludes model loading, disk I/O, RPC and simulation.

## Checkpoints

### BIN

VLA: /scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_train/594020.hopper-m-02/train/checkpoints/best/step_17000/pretrained_model

Tokenizer: /scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_smoke/bin_robotwin_bootstrap/checkpoints/latest/pretrained_model

### FAST

VLA: /scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_bar_fast_pali_fft/600669.hopper-m-02/train/checkpoints/best/step_5200/pretrained_model

Tokenizer: /scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_fast_q01_q99_noclip_v1024_filtered/598729.hopper-m-02/train/checkpoints/latest/pretrained_model

## Results

| Method | Latency / s (mean ± 95% CI) | p95 / s | Predicted tokens mean [min,max] | VLM calls mean | GT tokens mean [min,max] |
|---|---:|---:|---:|---:|---:|
| BIN | 0.62690 ± 0.00122 | 0.65862 | 448.000 [448,448] | 32.000 | 448.00 [448,448] |
| FAST | 0.65988 ± 0.02718 | 1.02270 | 32.015 [14,66] | 33.015 | 24.39 [7,57] |

BIN: fixed 32 × 14 = 448 tokens; schedule [14] × 32; 32 VLM calls. FAST: all 600 predictions terminate on EOS, none exhaust the budget. Action-token counts exclude EOS and padding. Calls include the EOS step and are derived from native decoding semantics, not a forward hook. GT lengths encode the same 200 action chunks, unpadded and untruncated.

## Limitations

Both located checkpoints train on local/robotwin2_depth_merged_clean; FAST additionally filters training episodes. Test observations come from robotwin_new_demo_clean to control inputs across methods. This is not a matched-training-data comparison. FAST predicted lengths and latency may depend on this distribution difference.

Earlier five methods used hopper-20; these used hopper-18, both H200. Cross-job speed ratios are descriptive, not same-device paired measurements. BIN round means: {'0': 610.555032775, '1': 622.582155525, '2': 647.5701516350001} ms. FAST round means: {'0': 647.12046521, '1': 648.029053215, '2': 684.488262565} ms. Episode bootstrap CI (5,000 draws, normal approximation) does not capture all hardware/run drift.

## Verification

Validated sample identities, 600 unique round/sample pairs per method, means, p95, token distribution means, EOS accounting, confidence intervals, input hash and executed script hash. Original results unchanged.

Artifacts: outputs/vla_latency/20260928_bin_fast/. Token statistics: token_counts.csv.
