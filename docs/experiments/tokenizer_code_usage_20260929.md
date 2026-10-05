# Tokenizer 词表使用情况 — 2026-09-29

## 统一数据范围

原始五方法使用与 latency / 参数量报告一致的 tokenizer checkpoint；本次追加三个 tokenizer 300k 的 BTT 消融比较，详见文末。统计 robotwin_new_demo_clean 的 125 个验证 episodes 中，815 个完整、无 padding、不重叠的 32×14 动作块（stride=32），每模型共 13,040 个 action token。

这里统计 GT action 编码产生的 token ID 在完整 1920 / 2048 词表上的覆盖率，不是 VLA 预测词表使用率，也不是全训练集/整个训练过程的覆盖率。验证集参与过 checkpoint selection，不是独立 test。

## 总表

| Method | 词表大小 | 已使用 | 未出现 | 使用率 | 有效词表大小（perplexity） |
| --- | ---: | ---: | ---: | ---: | ---: |
| QueST | 1920 | 1757 | 163 | 91.51% | 1036.92 |
| OAT^sing | 1920 | 1832 | 88 | 95.42% | 1219.37 |
| OAT^pow2 | 1920 | 1860 | 60 | 96.88% | 1402.95 |
| Acodec | 2048 | 1854 | 194 | 90.53% | 1376.41 |
| BTT（原表，297k） | 1920 | 1670 | 250 | 86.98% | 782.27 |
| BTT（300k，匹配基线） | 1920 | 1677 | 243 | 87.34% | 783.24 |
| BTT w/o convention（300k） | 1920 | 1500 | 420 | 78.12% | 794.82 |
| BTT independent（300k） | 1920 | 1731 | 189 | 90.16% | 1030.39 |

已使用 = 至少出现一次的不同 token ID 数；未出现只表示本次样本中未出现，不能据此认定为 dead code。有效词表大小 = exp(Shannon entropy)，用于反映使用分布是否集中，并不是不同 token ID 的数量。

## 原表 BTT（297k）双臂

左臂使用 1395/1920；右臂使用 1405/1920；交集 1130，并集 1670。

双臂共享同一套 FSQ token ID 空间，分母为 1920；不能把左右臂已使用数量直接相加。

## 每个 token 位置的不同 ID 数

| Method | Positions 0–15 |
| --- | --- |
| QueST | 333, 407, 470, 433, 428, 471, 480, 442, 416, 432, 452, 452, 471, 443, 414, 353 |
| OAT^sing | 403, 604, 564, 595, 577, 603, 480, 389, 389, 421, 194, 209, 209, 285, 574, 592 |
| OAT^pow2 | 408, 587, 552, 573, 562, 557, 517, 386, 373, 390, 356, 492, 461, 508, 582, 587 |
| Acodec | 112, 160, 177, 512, 512, 500, 480, 481, 492, 481, 308, 349, 336, 330, 319, 274 |
| BTT（297k） | 253, 289, 324, 353, 392, 446, 432, 431, 265, 308, 336, 339, 388, 452, 453, 438 |

## 来源与复核

- QueST / OAT^sing / Acodec / BTT：复用 task1_fk_reconstruction/20260928_validation 中的 token_ids，并核对 checkpoint 路径。
- OAT^pow2：本次对完全相同的 shared_samples.npz 补算，CPU float32，eval + inference_mode，batch=32。
- 核对 shared_samples SHA256；五种输出都是 [815,16]，全部 token ID 在合法范围；每份直方图计数总和为 13,040。
- [统计 JSON](../../outputs/tokenizer_code_usage/20260929_validation815/summary.json) 包含 checkpoint、未使用 ID 列表、逐位置覆盖和数据 hash。
- [完整 token IDs](../../outputs/tokenizer_code_usage/20260929_validation815/token_ids.npz)。

逐 ID 频次表（包括出现次数为 0 的 ID）：

- [QueST](../../outputs/tokenizer_code_usage/20260929_validation815/QueST_counts.csv)
- [OAT^sing](../../outputs/tokenizer_code_usage/20260929_validation815/OAT_single_counts.csv)
- [OAT^pow2](../../outputs/tokenizer_code_usage/20260929_validation815/OAT_pow2_counts.csv)
- [Acodec](../../outputs/tokenizer_code_usage/20260929_validation815/Acodec_counts.csv)
- [BTT](../../outputs/tokenizer_code_usage/20260929_validation815/BTT_counts.csv)

## BTT 消融验证：共享参数是否降低词表覆盖率？

### 验证层级

本次是 **纯 tokenizer 验证**：GT action → tokenizer → token IDs / reconstructed actions。读取用户指定 VLA 的 latest checkpoint 配置与训练步数，使用其中 action_tokenizer.pretrained_path 指向的真实 tokenizer checkpoint。没有运行 VLM forward、VLA token 生成或环境 rollout。两个 VLA 均为 step 30,000、freeze_action_tokenizer=true，其 model.safetensors 不含 tokenizer/codec 键，tokenizer 由外部路径加载。

该实验直接回答词表覆盖、双臂 ID 复用和动作信息保留问题。若问题是 VLA 能否学会这些 token、能否提高任务成功率，则另需 VLA 生成评估/闭环 rollout；本报告未测量或推断这些指标。

### 对齐条件和消融差异

- 所有新比较使用 tokenizer step 300,000，从各自 VLA step 30,000 配置定位。原表 BTT 是 step 297,000，保留原结果，但不作为新消融的主基线。
- 相同 815 个动作块、125 个验证 episodes、每模型 13,040 个 token；CPU float32，eval + inference_mode，batch=32，threads=4。
- 三者训练配置核对了相同数据根目录、seed=42、val_ratio=0.05。单次训练结果不能代表多 seed 的因果结论。
- 标准 BTT：共享 arm encoder / self-attention / 双向 cross-attention 参数 / pre_quant；使用右臂 sign convention 和共享归一化范围。
- w/o convention：保持共享结构，右臂 sign 全为 +1；归一化范围也改为原始左右臂 pooled range。因此不是只改一个 sign 输入的推理实验。
- independent：保留右臂 convention，但 arm encoder、self-attention、双向 cross-attention、pre_quant 均改为左右独立；同时使用左右独立归一化范围。参数从 7,382,802 增至 8,439,062，不是严格只切换 weight sharing 的单变量对照。
- 三者仍使用同一个固定 FSQ 网格 [8,8,6,5] 和 1920 个 ID；independent 并未变成两套各 1920 的独立词表。

### 同步数主结果

| 模型 | 已使用 / 1920 | 覆盖率 | 左臂已使用 | 右臂已使用 | 交集 | 交集 / 并集 | 有效词表大小 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTT（300k，匹配基线） | 1677 | 87.34% | 1392 | 1403 | 1118 | 66.67% | 783.24 |
| BTT w/o convention（300k） | 1500 | 78.12% | 1174 | 1284 | 958 | 63.87% | 794.82 |
| BTT independent（300k） | 1731 | 90.16% | 1302 | 1185 | 756 | 43.67% | 1030.39 |

### 分布与对称性检查

JSD 是同一模型内左右臂 ID 频率分布的 Jensen–Shannon divergence（bit）；0 表示分布相同。归一化输入交换测试：先用各模型自己的 normalization 得到输入，再交换左右 7 维，检验输出是否恰好交换前后 8 个 token。这样隔离 normalization 差异，检查 encoder / projection 的结构对称性。交换后的输入不计入自然动作覆盖率。

| 模型 | 左右臂分布 JSD（bit） | 交换后 token 一致率 | 交换后整序列一致率 |
| --- | ---: | ---: | ---: |
| BTT（300k，匹配基线） | 0.151178 | 100.0000% | 100.00% |
| BTT w/o convention（300k） | 0.265696 | 100.0000% | 100.00% |
| BTT independent（300k） | 0.534860 | 0.0537% | 0.00% |

ID 重叠或低 JSD 不能单独证明动作语义相同，尤其 independent 的左右投影可形成不同编码。交换一致率检查结构等变性，不等于重建质量，也不表示 decoder 一定具有相同等变性。

### 原始动作重建质量

在原始动作坐标下比较，不比较不同 normalization 下的训练 loss。MSE/MAE 对同一 [815,32,14] 等权平均；14 维包含关节和归一化夹爪命令，不是末端位置/旋转误差。

| 模型 | Raw action MSE | 相对基线 MSE 变化 | Raw action MAE | 左臂 MSE | 右臂 MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| BTT（300k，匹配基线） | 0.000519642 | +0.00% | 0.012053653 | 0.000564840 | 0.000474445 |
| BTT w/o convention（300k） | 0.000571689 | +10.02% | 0.012649665 | 0.000562953 | 0.000580425 |
| BTT independent（300k） | 0.000551306 | +6.09% | 0.011770082 | 0.000569682 | 0.000532930 |

配对差值 = 消融 − 标准 BTT，正值表示消融 MSE 更高。按 episode 做 5,000 次 cluster bootstrap，seed=42，95% percentile CI；仅反映验证样本不确定性，不覆盖训练 seed 方差，未作多重比较校正。

| 比较 | MSE 差值 | 95% CI |
| --- | ---: | ---: |
| BTT_wo_convention_minus_BTT_300k | 0.000052047 | [0.000002851, 0.000116323] |
| BTT_independent_minus_BTT_300k | 0.000031663 | [-0.000015962, 0.000089614] |

### 对原假设的结论

1. **部分支持共享促进跨臂 ID 复用的解释。** independent 合并覆盖从 1677/1920 上升至 1731/1920（+54 ID，+2.8125 个百分点），但左右单臂覆盖均下降；交集从 1118 降至 756，Jaccard 从 66.67% 降至 43.67%。合并覆盖增加伴随两臂使用更加分离，而不是每条臂都使用更多 token。
2. **不能认定 convention 降低词表覆盖。** 去掉 convention 后覆盖降至 1500/1920（−177 ID，−9.21875 个百分点）；共享和 convention 是不同因素。覆盖下降而 perplexity 略升，说明出现过多少 ID 与使用是否均匀不同。
3. **更高覆盖不自动意味着更好重建。** independent 的 MSE 点估计高 6.09%，但配对 95% CI 跨 0，不能确认其 MSE 变差；MAE 反而略低。w/o convention 的 MSE 点估计高 10.02%，本次未校正 CI 不跨 0。
4. **因果解释有限。** independent 同时改变归一化和容量，且只有单 seed；结果与共享促进复用一致，但不能唯一归因于共享权重，也不能推断 VLA 成功率。

### Checkpoint 对应关系

#### BTT（300k，匹配基线）

- VLA（step 30,000；仅读配置定位 tokenizer）：`/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_bar_btt_pali_robotwin_new_step300000_30k/630822.hopper-m-02/train/checkpoints/latest/pretrained_model`
- Tokenizer（step 300,000；实际执行）：`/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_btt_robotwin_new/621340.hopper-m-02/train/checkpoints/best/step_300000/pretrained_model`

#### BTT w/o convention（300k）

- VLA（step 30,000；仅读配置定位 tokenizer）：`/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_bar_btt_wo_convention_pali_robotwin_new_step300000_30k/633464.hopper-m-02/train/checkpoints/latest/pretrained_model`
- Tokenizer（step 300,000；实际执行）：`/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_btt_wo_convention_robotwin_new/628094.hopper-m-02/train/checkpoints/best/step_300000/pretrained_model`

#### BTT independent（300k）

- VLA（step 30,000；仅读配置定位 tokenizer）：`/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_bar_btt_independent_pali_robotwin_new_step300000_30k/634432.hopper-m-02/train/checkpoints/latest/pretrained_model`
- Tokenizer（step 300,000；实际执行）：`/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_btt_independent_robotwin_new/633923.hopper-m-02/train/checkpoints/best/step_300000/pretrained_model`

### 新增逐位置覆盖

| 模型 | Positions 0–15 |
| --- | ---: |
| BTT（300k，匹配基线） | 259, 293, 326, 360, 401, 443, 433, 435, 273, 306, 335, 350, 398, 452, 458, 445 |
| BTT w/o convention（300k） | 240, 322, 369, 377, 393, 393, 386, 336, 236, 225, 302, 339, 415, 492, 464, 363 |
| BTT independent（300k） | 293, 368, 394, 406, 455, 482, 512, 481, 224, 297, 318, 342, 409, 382, 364, 334 |

### 复现与产物

仓库根目录执行：

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 .venv/bin/python scripts/tokenizer/evaluate_btt_usage_ablation.py
```

- [评估脚本](../../scripts/tokenizer/evaluate_btt_usage_ablation.py)
- [完整结果、checkpoint hash 和配置](../../outputs/tokenizer_code_usage/20260929_btt_ablation/summary.json)
- [验证记录](../../outputs/tokenizer_code_usage/20260929_btt_ablation/verification.json)

- BTT（300k，匹配基线）：[ID 频次](../../outputs/tokenizer_code_usage/20260929_btt_ablation/BTT_300k_counts.csv)、[逐 chunk 误差](../../outputs/tokenizer_code_usage/20260929_btt_ablation/BTT_300k_per_chunk.csv)、[token IDs 和重建动作](../../outputs/tokenizer_code_usage/20260929_btt_ablation/BTT_300k.npz)。
- BTT w/o convention（300k）：[ID 频次](../../outputs/tokenizer_code_usage/20260929_btt_ablation/BTT_wo_convention_counts.csv)、[逐 chunk 误差](../../outputs/tokenizer_code_usage/20260929_btt_ablation/BTT_wo_convention_per_chunk.csv)、[token IDs 和重建动作](../../outputs/tokenizer_code_usage/20260929_btt_ablation/BTT_wo_convention.npz)。
- BTT independent（300k）：[ID 频次](../../outputs/tokenizer_code_usage/20260929_btt_ablation/BTT_independent_counts.csv)、[逐 chunk 误差](../../outputs/tokenizer_code_usage/20260929_btt_ablation/BTT_independent_per_chunk.csv)、[token IDs 和重建动作](../../outputs/tokenizer_code_usage/20260929_btt_ablation/BTT_independent.npz)。

复核包含 checkpoint 步数、数据划分配置、样本 hash、token 范围/shape、重建有限值、保存后的直方图计数及重建 MSE。
