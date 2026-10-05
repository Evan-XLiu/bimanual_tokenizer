# 七种 Tokenizer 完整 Latency 汇总

整理日期：2026-09-29；测量日期：2026-09-28。来自现有 benchmark，没有重新运行模型。所有时间单位均为 **ms / chunk**。

OAT^sing 对应原始记录 OAT_single；OAT^pow2 对应 OAT_pow2（独立训练的 BOAT checkpoint）。

## 1. 总览

Encoder / Decoder 指完整 codec.tokenize / codec.detokenize API，不仅是神经网络层。Encode + Decode 是两个均值的算术和，不是独立端到端测量。正常 VLA 推理不包含 tokenizer encode。

| Method | 预测 tokens | 生成 blocks / calls | Encode mean | Decode mean | Encode + Decode | VLA mean ± 95% CI | VLA p95 | Chunks/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BIN | 448 | 32 | 0.109 | 0.099 | 0.208 | 626.902 ± 1.220 | 658.615 | 1.595 |
| FAST | 32.015 | 33.015 | 0.667 | 0.397 | 1.064 | 659.879 ± 27.184 | 1022.704 | 1.515 |
| QueST | 16 | 16 | 2.475 | 2.541 | 5.016 | 328.981 ± 0.296 | 336.676 | 3.040 |
| OAT^sing | 16 | 16 | 4.235 | 2.920 | 7.155 | 329.665 ± 0.469 | 337.333 | 3.033 |
| OAT^pow2 | 16 | 5 | 4.182 | 2.903 | 7.085 | 115.976 ± 0.289 | 120.272 | 8.622 |
| Acodec | 16 | 1 | 2.202 | 1.932 | 4.135 | 38.525 ± 1.443 | 38.038 | 25.957 |
| BTT | 16 | 2 | 2.360 | 2.663 | 5.024 | 56.827 ± 0.114 | 58.853 | 17.597 |

FAST tokens / calls 为均值，tokens 不含 EOS / padding，calls 包含 EOS 步骤。Chunks/s = 1000 / VLA mean，不是机器人控制频率。

## 2. Tokenizer 完整编码与解码统计

每方法预热后测量 200 对；外层输入已放到目标设备，同步计时，外层 CPU→GPU 搬运不计入。API 内部实现自身的转换仍计入。解码输入为真实动作编码得到的 token，不是 VLA 预测 token。

### Encoder：codec.tokenize

| Method | N | Mean | Std | p50 | p95 | 1000 / mean（次/s） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BIN | 200 | 0.109 | 0.006 | 0.108 | 0.113 | 9195.652 |
| FAST | 200 | 0.667 | 0.014 | 0.666 | 0.691 | 1500.231 |
| QueST | 200 | 2.475 | 0.025 | 2.472 | 2.494 | 404.031 |
| OAT^sing | 200 | 4.235 | 0.045 | 4.235 | 4.277 | 236.110 |
| OAT^pow2 | 200 | 4.182 | 0.037 | 4.181 | 4.223 | 239.118 |
| Acodec | 200 | 2.202 | 0.025 | 2.200 | 2.236 | 454.039 |
| BTT | 200 | 2.360 | 0.019 | 2.360 | 2.380 | 423.640 |

### Decoder：codec.detokenize

| Method | N | Mean | Std | p50 | p95 | 1000 / mean（次/s） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BIN | 200 | 0.099 | 0.004 | 0.099 | 0.103 | 10068.182 |
| FAST | 200 | 0.397 | 0.009 | 0.397 | 0.413 | 2516.578 |
| QueST | 200 | 2.541 | 0.010 | 2.540 | 2.557 | 393.576 |
| OAT^sing | 200 | 2.920 | 0.024 | 2.924 | 2.948 | 342.509 |
| OAT^pow2 | 200 | 2.903 | 0.023 | 2.904 | 2.936 | 344.495 |
| Acodec | 200 | 1.932 | 0.018 | 1.930 | 1.959 | 517.490 |
| BTT | 200 | 2.663 | 0.013 | 2.664 | 2.683 | 375.470 |

## 3. 完整 VLA 推理统计

CPU observation → 预处理/传输 → vision/prefix → token 生成 → detokenize/反归一化 → CPU float32 action chunk。每次预测新 chunk；排除磁盘 I/O、checkpoint 加载、RPC 和仿真。

| Method | N | Mean ± 95% CI | Std | p50 | p95 | 相对 OAT^sing 加速比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BIN | 600 | 626.902 ± 1.220 | 17.546 | 622.876 | 658.615 | 0.526×（跨作业） |
| FAST | 600 | 659.879 ± 27.184 | 163.577 | 625.976 | 1022.704 | 0.500×（跨作业） |
| QueST | 600 | 328.981 ± 0.296 | 4.971 | 329.374 | 336.676 | 1.002× |
| OAT^sing | 600 | 329.665 ± 0.469 | 4.422 | 329.365 | 337.333 | 1.000× |
| OAT^pow2 | 600 | 115.976 ± 0.289 | 2.375 | 115.785 | 120.272 | 2.843× |
| Acodec | 600 | 38.525 ± 1.443 | 17.647 | 36.693 | 38.038 | 8.557× |
| BTT | 600 | 56.827 ± 0.114 | 1.407 | 57.036 | 58.853 | 5.801× |

## 4. VLA 分阶段 Profiling

独立诊断 pass，每方法 50 次。profile_model_ms 包括 observation/prefix 处理和 token 生成，并非纯自回归时间。profile_detokenize_ms 是 VLA 路径的 detokenize + 反归一化阶段。样本数和同步边界不同，阶段均值不能直接相加当作主 benchmark 的端到端均值。现有数据没有独立的 vision encoder / prefill 耗时，也不能由相减推算。

### 模型处理及 token 生成

| Method | N | Mean | Std | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BIN | 50 | 606.218 | 1.930 | 606.848 | 609.018 |
| FAST | 50 | 635.906 | 164.922 | 589.988 | 948.782 |
| QueST | 50 | 318.838 | 1.322 | 318.648 | 320.890 |
| OAT^sing | 50 | 326.763 | 3.215 | 326.778 | 331.272 |
| OAT^pow2 | 50 | 109.770 | 0.819 | 109.610 | 110.789 |
| Acodec | 50 | 33.281 | 0.596 | 33.202 | 34.045 |
| BTT | 50 | 50.644 | 0.468 | 50.760 | 51.277 |

### Detokenize + 反归一化

| Method | N | Mean | Std | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BIN | 50 | 0.492 | 0.017 | 0.489 | 0.516 |
| FAST | 50 | 0.713 | 0.034 | 0.712 | 0.746 |
| QueST | 50 | 3.285 | 0.061 | 3.272 | 3.353 |
| OAT^sing | 50 | 3.696 | 0.084 | 3.684 | 3.740 |
| OAT^pow2 | 50 | 3.599 | 0.042 | 3.605 | 3.659 |
| Acodec | 50 | 2.588 | 0.068 | 2.584 | 2.676 |
| BTT | 50 | 3.382 | 0.053 | 3.392 | 3.446 |

## 5. Token 长度与生成调度

| Method | 调度 | 预测 tokens mean [min,max] | GT tokens mean [min,max] | VLM calls mean [min,max] |
| --- | ---: | ---: | ---: | ---: |
| BIN | [14] × 32 | 448.000 [448, 448] | 448.000 [448, 448] | 32.000 [32, 32] |
| FAST | [1] × N；EOS 停止；预算 128 | 32.015 [14, 66] | 24.390 [7, 57] | 33.015 [15, 67] |
| QueST | [1] × 16 | 16（固定） | 16（固定） | 16（固定） |
| OAT^sing | [1] × 16 | 16（固定） | 16（固定） | 16（固定） |
| OAT^pow2 | [1, 1, 2, 4, 8] | 16（固定） | 16（固定） | 5（固定） |
| Acodec | [16] × 1 | 16（固定） | 16（固定） | 1（固定） |
| BTT | [8] × 2 | 16（固定） | 16（固定） | 2（固定） |

FAST：600/600 次在 EOS 停止，预算耗尽 0 次。GT 长度来自同一组 200 个动作 chunk，无 padding、无截断。VLM calls 按原生解码语义推导，不是 forward hook 实测。

## 6. 显存与各轮 VLA 均值

显存为完整 VLA 的 CUDA allocated memory，十进制 GB；不是 tokenizer 独占显存，也不是 nvidia-smi 总占用。增量为 peak − resident。

| Method | Resident GB | Peak GB | 增量 GB | Round 0 ms | Round 1 ms | Round 2 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BIN | 6.112 | 6.303 | 0.191 | 610.555 | 622.582 | 647.570 |
| FAST | 6.115 | 6.297 | 0.182 | 647.120 | 648.029 | 684.488 |
| QueST | 6.165 | 6.348 | 0.182 | 323.437 | 330.985 | 332.522 |
| OAT^sing | 6.163 | 6.345 | 0.182 | 329.611 | 327.631 | 331.751 |
| OAT^pow2 | 6.163 | 6.345 | 0.182 | 114.588 | 117.031 | 116.310 |
| Acodec | 6.168 | 6.351 | 0.184 | 37.202 | 35.154 | 43.219 |
| BTT | 6.149 | 6.333 | 0.184 | 55.702 | 57.398 | 57.380 |

## 7. 测量条件与比较边界

- NVIDIA H200，batch size 1；PaliGemma2 3B mix224；VLM bfloat16，codec 原生 dtype；PyTorch 2.7.1+cu126、SDPA、native KV cache、eager，无 vLLM / compile。
- 相同 200 条缓存 observations，3 轮，每轮每方法 30 次预热；完整 VLA 每方法 600 次，输出 [1,32,14]。
- BIN / FAST 在 hopper-18，PBS 636884.hopper-m-02；其他五种在 hopper-20，PBS 636642.hopper-m-02。跨作业加速比仅作描述，不是同卡配对结果。
- CI 为 episode-cluster bootstrap 正态近似，5,000 draws，seed 42；Std 为样本标准差。CI 不包含独立硬件/作业不确定性或全部轮间漂移。
- OAT^pow2 使用独立训练的 BOAT checkpoint，不是同一 OAT^sing checkpoint 只更改推理调度。
- BIN / FAST 的训练数据为 local/robotwin2_depth_merged_clean，FAST 额外过滤 episode；测试输入来自 robotwin_new_demo_clean。七方法不是严格匹配训练数据的比较。
- Acodec 有长尾：600 次中 7 次超过 100 ms，最大 243.215 ms，全部保留；因此 mean 可高于 p95。
- 本表仅代表测量日期的指定 checkpoint，不自动代表当前代码或新 checkpoint 的重测性能。

## 8. Checkpoint 溯源

### BIN

- Policy：bar_bin_pali
- VLA checkpoint：/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_train/594020.hopper-m-02/train/checkpoints/best/step_17000/pretrained_model
- Tokenizer checkpoint：/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_smoke/bin_robotwin_bootstrap/checkpoints/latest/pretrained_model

### FAST

- Policy：bar_fast_pali
- VLA checkpoint：/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_bar_fast_pali_fft/600669.hopper-m-02/train/checkpoints/best/step_5200/pretrained_model
- Tokenizer checkpoint：/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_fast_q01_q99_noclip_v1024_filtered/598729.hopper-m-02/train/checkpoints/latest/pretrained_model

### QueST

- Policy：bar_quest_pali
- VLA checkpoint：/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_bar_quest_pali_robotwin_new_30k/613537.hopper-m-02/train/checkpoints/best/step_26000/pretrained_model
- Tokenizer checkpoint：/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_quest_robotwin_new/613294.hopper-m-02/train/checkpoints/best/step_299000/pretrained_model

### OAT^sing

- Policy：bar_oat_pali
- VLA checkpoint：/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_bar_oat_pali_fft_new_2500/601387.hopper-m-02/train/checkpoints/best/step_13400/pretrained_model
- Tokenizer checkpoint：/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_oat_robotwin_new_2500/600693.hopper-m-02/train/checkpoints/best/step_296000/pretrained_model

### OAT^pow2

- Policy：bar_boat_pali
- VLA checkpoint：/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_bar_boat_pali_robotwin_new_30k/616455.hopper-m-02/train/checkpoints/best/step_30000/pretrained_model
- Tokenizer checkpoint：/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_boat_robotwin_new/616167.hopper-m-02/train/checkpoints/best/step_300000/pretrained_model

### Acodec

- Policy：bar_acodec_pali
- VLA checkpoint：/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_bar_acodec_pali_robotwin_new_step300000_30k/630823.hopper-m-02/train/checkpoints/best/step_18000/pretrained_model
- Tokenizer checkpoint：/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_acodec_robotwin_new/627717.hopper-m-02/train/checkpoints/recent/step_300000/pretrained_model

### BTT

- Policy：bar_btt_pali
- VLA checkpoint：/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_bar_btt_pali_robotwin_new_30k/622038.hopper-m-02/train/checkpoints/latest/pretrained_model
- Tokenizer checkpoint：/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_btt_robotwin_new/621340.hopper-m-02/train/checkpoints/best/step_297000/pretrained_model

## 9. 原始数据与核对

从 summary.json 的未舍入值生成，显示三位小数；已对照 measurements.jsonl 核对七方法全部五类时间指标的计数和均值（35 项），均一致。

### 20260928_native5

- [summary.json](../../outputs/vla_latency/20260928_native5/summary.json)
- [measurements.jsonl](../../outputs/vla_latency/20260928_native5/measurements.jsonl)
- [manifest.json](../../outputs/vla_latency/20260928_native5/manifest.json)
- [hardware.json](../../outputs/vla_latency/20260928_native5/hardware.json)
- [benchmark_vla_latency.executed.py](../../outputs/vla_latency/20260928_native5/benchmark_vla_latency.executed.py)
- [verification.json](../../outputs/vla_latency/20260928_native5/verification.json)

### 20260928_bin_fast

- [summary.json](../../outputs/vla_latency/20260928_bin_fast/summary.json)
- [measurements.jsonl](../../outputs/vla_latency/20260928_bin_fast/measurements.jsonl)
- [manifest.json](../../outputs/vla_latency/20260928_bin_fast/manifest.json)
- [hardware.json](../../outputs/vla_latency/20260928_bin_fast/hardware.json)
- [benchmark_vla_latency.executed.py](../../outputs/vla_latency/20260928_bin_fast/benchmark_vla_latency.executed.py)
- [verification.json](../../outputs/vla_latency/20260928_bin_fast/verification.json)

既有解读：[五方法报告](vla_latency_results_20260928.md)、[BIN/FAST 报告](bin_fast_latency_results_20260928.md)。

## 10. Tokenizer 参数量（2026-09-29 核对）

使用第 8 节相同 tokenizer checkpoint，通过 make_codec 加载后统计去重的 named_parameters() / numel()，并逐项核对 model.safetensors 中的参数名及 shape。仅统计 tokenizer，不包含 VLM、VLA action embedding 或 LM head。1 M = 1,000,000。

| Method | Encoder 参数 | Decoder 参数 | 其他 Parameter | 总参数 | 总参数（M） |
| --- | ---: | ---: | ---: | ---: | ---: |
| BIN | 0 | 0 | 0 | 0 | 0.000000 |
| FAST | 0 | 0 | 0 | 0 | 0.000000 |
| QueST | 5,269,252 | 6,325,518 | 0 | 11,594,770 | 11.594770 |
| OAT^sing | 6,297,348 | 4,720,654 | 0 | 11,018,002 | 11.018002 |
| OAT^pow2 | 6,297,348 | 4,720,654 | 0 | 11,018,002 | 11.018002 |
| Acodec | 4,875,008 | 4,870,670 | 0 | 9,745,678 | 9.745678 |
| BTT | 1,056,260 | 6,326,542 | 0 | 7,382,802 | 7.382802 |

统计口径：

- Encoder 包含输入投影、时序卷积、encoder 主干和 pre_quant；Decoder 包含 post_quant、decoder 主干及 action head。BTT 共享的双臂 encoder 和 pre_quant 只计一次。
- 不计固定位置编码、归一化统计、FSQ 隐式码本、BIN 分箱等 buffers。
- BIN / FAST 的神经网络 Parameter 数为 0；FAST 的 BPE vocabulary / merges 是拟合的离散状态，并不意味着 tokenizer 没有存储或学习得到的内容。
- Acodec 的 EMA VQ codebook 是 buffer：2,048 × 512 = 1,048,576 个浮点值，不在上述 9,745,678 个 Parameter 内。若把该学习得到的码本也计入模型规模，则为 10,794,254（10.794254 M）；EMA 辅助状态 embed_avg / cluster_size 不重复计入该数。
- 表中参数在独立加载 tokenizer 时均 requires_grad=True；进入 VLA 后可被 freeze_action_tokenizer 冻结，冻结不改变总参数量。
- OAT^sing 与 OAT^pow2 参数量相同，但 checkpoint 和生成调度不同；参数量相同不代表权重相同。

模块级原始拆分：

| Method | 顶层模块及参数数量 |
| --- | --- |
| BIN | 无 nn.Parameter |
| FAST | 无 nn.Parameter |
| QueST | action_proj=3,840; conv_block=525,824; encoder=4,738,560; decoder=6,320,640; pre_quant=1,028; post_quant=1,280; action_head=3,598 |
| OAT^sing | encoder=6,297,348; decoder=4,720,654 |
| OAT^pow2 | encoder=6,297,348; decoder=4,720,654 |
| Acodec | encoder=4,875,008; decoder=4,870,670 |
| BTT | encoder=1,055,232; pre_quant=1,028; decoder=6,326,542 |
