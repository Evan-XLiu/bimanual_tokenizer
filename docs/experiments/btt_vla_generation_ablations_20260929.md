# BTT VLA 生成方式消融设计

记录日期：2026-09-29。

**状态（2026-09-29 更新）：实验二已实现为独立的 `btt_oneblock` tokenizer / BAR policy 接口；原 BTT 保持不变。实验一、三仍为设计。实验二的检查和训练提交状态见第 14 节，本文尚无该实验的训练完成结果、latency 或 rollout 成功率。**

## 1. 目的与范围

在现有 BTT 基线上，分别验证三个问题：

1. 左右臂的 token 生成顺序是否重要？
2. 将两个 block 合并为一个 block，能否减少串行步骤并保持效果？
3. 预测右臂时，输入左臂 token 是否带来收益？

这三个实验属于 **VLA 层面的消融**。可以复用同一个 BTT tokenizer checkpoint，无需重新训练 tokenizer；改变的是 VLA 如何预测 16 个动作 token。由于训练目标排列、条件输入或 attention 结构发生变化，应分别训练对应 VLA，而不是仅在现有 VLA 推理时切换设置就作为正式消融结果。

已有的 BTT independent 改的是 tokenizer 内的参数共享；本文的三个实验改的是 VLA 生成方式，二者不是同一层面的消融。

## 2. 当前基线的实际行为

### 2.1 Convention 与 token 排列

标准 BTT 将右臂动作逐维乘以以下 sign，转换到与左臂一致的坐标约定，再进行归一化和编码：

```text
right_arm_sign = [-1, +1, +1, +1, -1, -1, +1]
```

这是统一坐标约定，不会交换左右臂身份。左右臂使用共享编码模块，但各自产生 8 个 token。序列排列为：

```text
[L0, L1, ..., L7, R0, R1, ..., R7]
block_schedule = [8, 8]
```

这里的 L、R 表示左右臂 latent token，而不是单个原始动作时间步。

### 2.2 训练与推理的区别

记 O 为图像、语言和状态等观测，L、R 为真实动作编码得到的左右臂 token。当前训练学习：

```text
根据 O 预测 L
根据 O 和真实 L 预测 R
```

训练采用 teacher forcing，将两个预测任务放在一次联合前向中计算：

```text
训练输入：O + [MASK × 8] + [真实 L × 8]
监督目标：    [真实 L × 8] + [真实 R × 8]
```

Block attention mask 防止预测左臂的位置偷看后面输入的真实 L。训练不是先生成预测左臂，再将预测左臂喂入右臂训练；一次联合前向也不代表取消了右臂对左臂的条件依赖。这里不把 gradient checkpointing 的反向重计算计作生成调用次数。

推理没有真实 L，因此必须先生成左臂：

```text
第 1 次 VLM 调用：O → 并行预测左臂 8 个 token，得到 L_hat
第 2 次 VLM 调用：O + L_hat → 并行预测右臂 8 个 token，得到 R_hat
[L_hat, R_hat] → tokenizer decoder → 双臂动作
```

第二次调用复用 KV cache。这里是两个 block、两次 VLM 调用，不是逐个 token 调用 16 次。

## 3. 实验矩阵

下列为实验标签；其中实验二已注册为 `btt_oneblock` / `bar_btt_oneblock_pali`，其余两个新实验的标签仍为建议命名。

| 实验 | 建议标签 | 生成方式 | 推理 VLM 调用次数 | 核心问题 |
| --- | --- | --- | ---: | --- |
| 当前基线 | BTT-LR | 左 8 → 右 8 | 2 | 参照组 |
| 反转顺序 | BTT-RL | 右 8 → 左 8 | 2 | 生成顺序是否重要？ |
| 合并一个 block | BTT-1Block | 左右 16 个一次预测 | 1 | 能否减少串行步骤并保持效果？ |
| 去掉跨臂 token 条件 | BTT-NoTokenCond | 保留两个 block，右臂不输入左臂 token | 2，作为控制 | 右臂是否受益于已知左臂 token？ |

三组共用同一个 tokenizer 的权重、convention、FSQ 词表、decoder、动作长度及总 token 数。反转顺序只改变 VLA 侧的排列接口，不改变原 tokenizer decoder 所要求的 [L, R] 输入语义。

## 4. 实验一：反转生成顺序，右 → 左

### 要验证的假设

当前左 → 右是否有顺序偏置？先生成哪条臂，会不会影响另一条臂的预测及双臂协调？

### 修改内容

保留 convention、共享编码器、词表和 decoder，仅将 VLA 的目标排列从 [L, R] 改为 [R, L]；block 大小仍为 [8, 8]。

```text
训练输入：O + [MASK × 8] + [真实 R × 8]
监督目标：    [真实 R × 8] + [真实 L × 8]

推理第 1 次：O → R_hat
推理第 2 次：O + R_hat → L_hat

生成结果 [R_hat, L_hat]
    → 恢复为 [L_hat, R_hat]
    → 原 tokenizer decoder
```

**反转生成顺序不等于反转 convention。** 右臂仍使用原来的 sign 变换；不交换机器人输出动作中的左右臂通道。

### 重点评估

- 左右臂分别的预测动作误差、夹爪误差。
- 双臂相对位姿误差。
- 任务成功率，并按任务观察顺序敏感性。
- 完整 VLA latency，验证同为两次调用时实际耗时是否接近。

若效果明显变化，说明生成顺序值得研究；变化也可能与不同任务的主从臂角色相关，不能只凭顺序差异就认定某条臂普遍更重要。

## 5. 实验二：合并成一个 block

### 要验证的假设

是否必须先产生一条臂的具体 token，才能预测另一条臂？一次并行生成全部 token 能否获得更好的速度与效果取舍？

### 修改内容

将生成调度由 [8, 8] 改为 [16]，维持原始 [L, R] 输出顺序：

```text
训练输入：O + [MASK × 16]
监督目标：    [L × 8, R × 8]

推理：一次 VLM 前向 → [L_hat, R_hat]
    → 原 tokenizer decoder
```

没有将已生成左臂 token 再输入右臂的过程。但当前 block attention 允许同一 block 内的预测位置交换隐藏信息，因此该实验不等于左右臂完全没有交互。

### 重点评估

- 完整 VLA latency：mean、p50、p95、chunks/s。
- VLA 生成动作的重建/预测误差及双臂相对位姿误差。
- 相同任务、相同 episode/seed 下的 rollout 成功率。

两次 VLM 调用减为一次，不代表端到端延迟必然减半；视觉前缀、序列长度、缓存、解码等成本需要通过实测体现。

## 6. 实验三：去掉右臂对左臂 token 的条件依赖

### 推荐的第一版定义

先做修改范围较小的对照：保留 [8, 8]、左右排列和原 attention mask，仅将第二个 block 的左臂 token 输入替换为 MASK。

| 场景 | 当前基线的第二块输入 | 消融的第二块输入 |
| --- | --- | --- |
| 训练 | 真实左臂 token × 8 | MASK × 8 |
| 推理 | 预测左臂 token × 8 | MASK × 8 |

```text
训练输入：O + [MASK × 8] + [MASK × 8]
监督目标：    [L × 8]    + [R × 8]
保留两个 block 的原 attention 结构

推理第 1 次：生成 L_hat
推理第 2 次：输入 MASK 而非 L_hat，生成 R_hat
```

右臂不接收真实或预测的左臂 token 内容，但仍可读取观测，以及第一块由观测和 MASK 计算出的缓存隐藏状态。该版本应称为 **去掉跨臂 token 条件**，不应称为完全独立双臂。

两次调用用于与当前基线保持生成流程的对照，不表示这种无 token 条件的模型在部署时必须分两次运行。改变执行方式是否保持等价，需要另行核验。

### 当前代码的关键注意事项

当前实现会将上一块 token 复制到下一块的输入位置。左臂 token 因此直接出现在第二块的输入 embedding 中。**只修改 attention mask、遮住前一块的位置，不能完成这个消融。** 训练的输入构造和推理的输入构造都必须同步取消这次复制。

### 重点评估

- 保留两次推理调用时，去掉左臂 token 是否影响右臂误差和双臂协调。
- 分别比较 teacher-forced 验证和实际生成验证，避免把真实左臂条件的收益与预测左臂条件的收益混为一谈。
- 任务成功率及左右臂分项误差。

可以增加条件干预诊断：固定观测，替换/打乱基线第二块输入的左臂 token，观察右臂输出变化；对 NoTokenCond，改变一个未被输入的左臂预测结果不应改变右臂输出。该诊断检查条件通路，不替代重训后的主结果。

### 可选后续：完全隔离双臂预测位置

如果还要禁止两个臂的预测位置通过隐藏状态交互，可进一步使用双臂隔离的 attention mask，保留各自对观测的访问。但这样同时改变了信息通路，不属于上述最小修改的 NoTokenCond 实验，宜单独命名、单独比较。

## 7. 单 block 与无 token 条件为何不是重复实验

二者均不输入另一条臂的真实/已预测 token，但隐藏信息的传播结构不同：

| 对比项 | BTT-1Block | BTT-NoTokenCond（推荐版） |
| --- | --- | --- |
| 调度 | [16] | [8, 8] |
| 输入动作位置 | 全部 MASK | 全部 MASK |
| Block 内交互 | 16 个位置处于同一 block | 每块 8 个位置 |
| 跨臂隐藏交互 | 按当前 mask 可双向交互 | 保留原左 → 右 block attention，右可读第一块隐藏状态 |
| 已生成左臂 token 作为右臂输入 | 无 | 无 |
| 主对照中的推理调用 | 1 | 2 |

因此基线 vs NoTokenCond 更直接检查左臂 token 内容的条件价值；基线 vs 1Block 检查整体并行化方案的速度与效果。1Block vs NoTokenCond 同时涉及 attention 结构和执行方式，不能把差异只归为调用次数。

## 8. 统一训练条件

建议三组复用标准 BTT tokenizer step 300,000：

```text
/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_btt_robotwin_new/621340.hopper-m-02/train/checkpoints/best/step_300000/pretrained_model
```

匹配基线的现有 VLA checkpoint 为：

```text
/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_bar_btt_pali_robotwin_new_step300000_30k/630822.hopper-m-02/train/checkpoints/latest/pretrained_model
```

正式消融建议固定：

- 相同 tokenizer 权重，冻结 tokenizer；相同 convention、FSQ 词表、动作输出几何和 decoder。
- 相同预训练 VLM 初始化、backbone、precision 和训练方式。
- 相同数据、train/validation episode 划分和采样设置。
- 相同 batch size、优化器、学习率计划和 30k VLA 训练预算。
- 一致的评估步数/选择规则，不为每组临时选择不同口径的 checkpoint。

新增三组应从与基线匹配的预训练 VLM 起点训练，不能一组从已完成的 BTT VLA 30k 继续微调、另一组从头训练却宣称预算相同。如果后续选择继续微调，则需要为所有比较组定义一致的继续训练预算和对照。

可先用同一个 seed 完成筛选；更强结论需要多个训练 seed。

## 9. 评估层级与建议顺序

### 9.1 实现一致性检查

训练前检查目标排列、还原排列、attention 可见性、训练/推理输入构造和解码接口。至少应确认：

- RL 输出还原后再进入原 decoder，左右臂不会串位。
- 同一批 GT 动作经过排列适配后，tokenizer 重建与原 tokenizer 一致。
- 训练输入没有泄露当前目标 block 的真实 token。
- NoTokenCond 的第二块输入不包含真实/预测的左臂 token；训练与推理保持同一规则。
- 实际生成调用数符合设计，KV cache 与 attention mask 一致。

### 9.2 离线 VLA 验证

在相同观测和 GT 动作上比较：

- Teacher-forced CE/预测准确率，区分第一/第二 block 及左右臂。
- 实际生成 token 的准确率和经 tokenizer decoder 得到的动作误差。
- 左右臂动作/夹爪误差、双臂相对位姿误差。

Teacher forcing 和自由生成的结果分开报告。真实 token 条件下的验证结果不能直接替代部署时使用预测 token 的表现。

### 9.3 完整 VLA 延迟

沿用既有 latency protocol，统一硬件、batch、观测、预热、同步和计时边界，报告 mean、p50、p95 和 chunks/s。不能用单独 tokenizer decoder latency 替代完整 VLA latency。

### 9.4 闭环任务评估

在相同任务、episode/seed 和执行设置下进行 rollout，报告任务成功率。任务表现用于判断生成设计能否提升实际控制效果，不能只靠离线 CE 或 token 准确率判断。

建议流程：实现一致性检查 → 三组 VLA 训练 → 同样本离线验证 → 延迟评估 → 相同设置的 rollout。

## 10. 哪些指标不会因这三个消融自动改变

由于 tokenizer 权重和输入动作固定，纯 tokenizer 的 GT 编码词表覆盖率、参数量及 GT tokenize → detokenize 重建能力应保持不变；RL 只重新排列 ID，不改变整体频次。

可能改变的是 VLA 预测 token 的使用分布、生成准确率、最终动作误差、推理延迟和任务成功率。需要明确区分 GT token 词表使用率与 VLA 预测 token 使用率。

## 11. 代码定位与工程边界

- [BTT tokenizer](../../src/praxis/codecs/tokens/btt/tokenizer.py)：[L,R] 序列化、[8,8] 调度、detokenize 的左右拆分。
- [BTT encoder](../../src/praxis/codecs/tokens/btt/encoder.py)：右臂 convention、归一化和共享编码路径。
- [VLA 训练](../../src/praxis/policies/core/vlm/training.py)：teacher-forced 输入准备及联合前向。
- [训练调度与 attention mask](../../src/praxis/policies/core/vlm/schedule.py)：上一块 token 复制及 block 可见性。
- [推理输入构造](../../src/praxis/policies/bar/core/vlm/decode_tokens.py)：build_prediction_inputs 将上一块预测复制到当前输入。
- [推理循环](../../src/praxis/policies/bar/core/vlm/decoding.py)：逐 block 调用、KV cache 和最终解码。
- [增量 attention mask](../../src/praxis/policies/bar/core/decode_schedule.py)：缓存推理阶段的可见性。
- [Policy config](../../src/praxis/policies/bar/core/vlm/config.py)：当前不允许直接配置 policy.block_schedule，调度来自 tokenizer 接口。

因此单 block 版本不能仅向现有配置添加 policy.block_schedule=[16]。实现时需设计明确的排列/调度适配接口，并保证保存后能从 checkpoint 恢复相同语义；可以保持 tokenizer 数值权重不变。不要直接覆盖当前 BTT 基线的默认行为。

现有同预算 launcher 可作训练条件参考：

- [标准 BTT 300k → VLA 30k PBS](../../artifacts/robotwin_hopper/train_bar_btt_pali_robotwin_new_step300000_30k.pbs)。
- [共用训练 launcher](../../artifacts/robotwin_hopper/train_bar_tokenizer_pali_robotwin_new_step300000_30k.sh)。

## 12. 相关记录与后续约定

- [Tokenizer latency 和参数量](tokenizer_latency_all_20260929.md)。
- [词表使用率与 tokenizer 结构消融](tokenizer_code_usage_20260929.md)。
- [VLA latency protocol](vla_latency_benchmark_protocol.md)。

本文件保存本次讨论，作为后续实现和复核入口。后续提到这三个生成消融时，以本文的定义区分 RL、1Block 和 NoTokenCond；如修改定义，应同步更新本文，不能用同一名称混用不同的 attention 或条件输入设计。

用户在本次明确要求将上述讨论完整写入 Markdown。此次操作已落地为本文；不代表已经授权选择其中某组替代其他组，也不代表任何训练已提交。

## 13. 实验二详解：同一 block 互相 attention 会不会造成训练/推理不一致？

补充核对日期：2026-09-29。

### 13.1 结论

按本文设计正确地在训练和推理中都使用 [16] 调度时，16 个预测位置均以 MASK 为输入，一次前向得到 16 个位置的 logits，然后一起选择输出 token。同块 attention 读取的是这些输入位置的隐藏向量，不是尚未生成的动作 token，也不是训练标签中的真实动作 token。因此不存在“训练看到了同块真实前序 token，推理却拿不到”的这类不一致。

这不等于无需核对实现：调度、输入构造、attention mask、标签对齐和位置处理仍必须一致。当前 BTT 默认仍是 [8,8]；此次核验是对现有通用函数传入 [16] 的逻辑检查，不代表单 block BTT 已训练完成。

### 13.2 两种 mask 不要混淆

| 名称 | 作用 | 单 block [16] 的含义 |
| --- | --- | --- |
| MASK token | 实际输入词表中的占位 token，有对应 embedding | 16 个待预测位置都填这个 token，不填真实动作 token |
| Attention mask | 决定每个 query 可以读取哪些 key/value 位置 | 允许这 16 个占位位置之间互相 attention，并允许读取观测前缀 |

输入位置填 MASK 并不意味着该位置被 attention 屏蔽。它只是表明这个位置没有输入真实目标值。源码中的 additive attention mask 使用 0 表示允许，浮点最小值表示屏蔽。

### 13.3 训练输入和真实标签是两份不同的数据

对于 [16]，build_prediction_block_input_ids 先把所有输入填成 MASK；因为这是第一块，没有上一块可以复制，所以它保持全 MASK。

```text
输入：观测 O + [M0, M1, ..., M15]
标签：         [L0, ..., L7, R0, ..., R7]
```

M0…M15 表示同一个 MASK token 在不同位置上的实例，不是 16 种不同 MASK ID。真实 token 被单独写入 labels，只在计算交叉熵时使用。第 i 个预测位置的 logits 对齐第 i 个目标，不是把同块真实目标整体作为输入再做普通 next-token shift。

因此训练时最后一个位置看到前面位置的隐藏向量，并不是看到了真实的 L0…R6。前面的真实答案并没有输入这一单 block。

### 13.4 块内是双向可见，而非仅后面看前面

build_attention_mask 的动作位置规则是：

```python
allow |= is_suffix_query & is_suffix_key & key_ids.le(query_ids)
```

key_ids / query_ids 是 block 编号，不是 token 在 block 内的位置编号。一个 block 的 16 个位置编号相同，因此前面可以看后面，后面也可以看前面。

以四个预测位置示意，1 代表允许：

```text
           K/V: M0 M1 M2 M3
Query M0        1  1  1  1
Query M1        1  1  1  1
Query M2        1  1  1  1
Query M3        1  1  1  1
```

实际 [16] 对应完整 16×16 可见性；padding 仍被屏蔽。prefix_attention 配置只改变观测前缀内部的可见性，不会把同一个动作 block 改成 token 级下三角 mask。

### 13.5 隐藏向量为什么在推理时也存在

未知的是最终动作 token ID，不是输入位置本身。推理开始时就构造全部 16 个 MASK 输入位置，它们有 embedding 和各自位置。当前 Pali adapter 通过 position_ids / RoPE 区分位置，随后各位置从观测中读取特征。

每层先由上一层全部位置的隐藏向量计算 Q、K、V，再进行 attention，产生下一层的隐藏向量：

```text
输入：16 个 MASK embedding + 观测与位置信息
  ↓ Transformer 第 1 层：全部位置一起计算
  ↓ Transformer 第 2 层：使用上一层全部位置的结果
  ↓ ...
输出：16 个隐藏向量 → 16 组 logits → 16 个 token ID
```

不存在先算当前层 M0 的最终答案、再让当前层 M1 读取该答案的顺序。读取的是上一层已经存在的连续表示。真实标签不参与这次前向的隐藏状态计算；梯度在 loss/backward 阶段更新模型权重是正常监督学习，不是同次前向泄露标签。

### 13.6 推理不是逐 token 的 generate 循环

run_blockwise_token_decode 按 generation_schedule 循环，而不是按每个 token 循环。[16]、keep_k=16 只有一轮：

1. prev_block_ids=None，build_prediction_inputs 产生 16 个 MASK。
2. Prefill 将观测前缀和这 16 个位置一起送入 VLM，使用同一套 block attention 构造规则。
3. 取最后 16 个位置的 logits，shape 为 [B,16,V]。
4. 将位置维展开，逐位置 argmax/采样，再恢复为 [B,16]。此处逐位置选择不包含 16 次 VLM 前向。
5. 一轮已经覆盖全部 16 个 token，退出生成循环并 detokenize。

同块内部不会在采样出某个 token 后立即将其喂回其他位置。如果选择独立随机采样，共享隐藏计算也不等于对采样结果做了显式的联合约束；本实验仍需用重建和 rollout 检查双臂协调质量。

### 13.7 与现有 [8,8] 的差异

| | 当前 [8,8] | 实验二 [16] |
| --- | --- | --- |
| 训练输入动作位置 | MASK × 8 + 真实左臂 × 8 | MASK × 16 |
| 推理输入动作位置 | 第一块 MASK；第二块输入预测左臂 | 全部 MASK，一次输入 |
| 右臂接收另一臂的具体 token | 有，训练为 GT、推理为预测值 | 无 |
| 同块隐藏交互 | 每个 8-token block 内可交互 | 全部 16 个位置可交互 |
| 同块输出 token 是否依次反馈 | 否 | 否 |

标准 [8,8] 的 teacher-forcing 与预测条件差异发生在 block 之间。单 block 去掉了该跨 block 的 GT/预测条件差异，但不能仅因此保证效果更好，因为它也去掉了右臂对已选择左臂 token 的显式条件化。

### 13.8 代码证据链

| 环节 | 文件与函数 | 本次定位行号 |
| --- | --- | --- |
| 训练输入 MASK / block shift | core/vlm/schedule.py — build_prediction_block_input_ids | 154–192 |
| 块内双向可见性 | core/vlm/schedule.py — build_attention_mask | 232–262 |
| 区分输入与目标 | core/vlm/training.py — prepare_action_token_ce_from_token_targets | 217–254 |
| 同位置标签赋值 | core/vlm/supervision.py — build_blockwise_labels_from_prediction_positions | 281–291 |
| Hidden 与标签分开计算 CE | core/vlm/training.py — compute_prepared_action_token_ce_loss_from_hidden | 429–474 |
| 训练层使用自定义 mask | backbones/pali/prefix.py — prepare_prefix / forward_prefix_layer | 88–116、139–151 |
| 推理 MASK 输入 | bar/core/vlm/decode_tokens.py — build_prediction_inputs | 56–74 |
| Prefill 传入 block mask | bar/core/vlm/decoding.py — build_vlm_prefill_inputs | 65–86 |
| 每块一次前向与所有位置采样 | bar/core/vlm/decoding.py — run_blockwise_token_decode | 290–338、370–376 |

上述路径相对于 src/praxis/policies/，对应可点击源码链接见第 11 节；行号记录于本次核对，未来源码修改可能移动。

### 13.9 已执行的辅助函数核验与边界

使用仓库实际的输入构造、标签、调度和 mask 函数，构造两条含不同左 padding 的合成序列，并额外覆盖训练尾部 processor EOS 场景。核验通过：

- [16] 训练输入全部是 MASK，且与推理首块输入完全一致。
- 替换所有真实目标 ID 后，输入不变，labels 改变。
- 训练标签按同位置对齐，不包含输入前缀。
- 训练/推理共同的观测及动作位置上，attention mask 完全相同。
- 16×16 共 256 条块内边全部允许，前后两个方向都允许。
- Padding key 被屏蔽；动作预测不能读取训练尾部 EOS。
- prefix_attention=bidirectional 和 causal 两种设置均通过以上检查。
- [16] 的生成调度仅一轮；[8,8] 的训练输入仍为 MASK × 8 + GT 左臂 × 8。

结果：[verification.json](../../outputs/tokenizer_code_usage/20260929_single_block_attention_check/verification.json)。复现：[verify.executed.py](../../outputs/tokenizer_code_usage/20260929_single_block_attention_check/verify.executed.py)。

这次没有运行完整 PaliGemma 权重的前向，也没有完成单 block VLA 训练。核验确认了输入/标签/mask 的机制，没有声称两个执行路径的全模型 logits 已做数值一致性测试；正式实现单 block 后仍应检查完整模型、位置处理和缓存路径。

## 14. 实验二实现与训练配置（2026-09-29）

### 14.1 新接口与原 BTT 的区别

独立名称为 **`btt_oneblock`**。实现继承原 `BTTTokenizer`，仅以新的配置身份覆盖 `block_schedule`；原 BTT 的 encoder、decoder、config 和训练脚本均未修改。工厂和 policy catalog 只新增注册项，不改变原名称的解析。

| 项目 | 标准 BTT | 新 `btt_oneblock` |
| --- | --- | --- |
| Tokenizer 参数量（当前标准配置） | 7,382,802 | 7,382,802 |
| 动作 chunk / 双臂维度 | 32 × 14 | 相同 |
| 右臂 convention | `[-1,1,1,1,-1,-1,1]` | 相同 |
| 双臂参数共享、FSQ、联合动作 decoder | 原实现 | 相同实现 |
| FSQ 容量 | 8 × 8 × 6 × 5 = 1920 | 相同 |
| 序列排列 | `[L0…L7,R0…R7]` | 相同 |
| VLA 生成调度 | `[8,8]` | `[16]` |
| VLA 训练输入的动作位置 | `MASK×8 + GT_L×8` | `MASK×16` |
| VLA 训练前向 | 一次联合前向 | 一次联合前向 |
| 完整 16-token 推理的 VLM 调用 | 两次、先左后右 | 一次、左右同时输出 |
| 同块 attention | 每 8 个位置内双向可见 | 全部 16 个位置双向可见 |
| 右臂输入已选择的左臂 token | 有 | 无；仍可交换预测位置的隐藏状态 |

**单 block 是 VLA 生成分块，不是把 tokenizer 的 6 层 decoder 改成 1 层，也不是把左右 latent 合并成 8 个 token。**

训练输入与标签分离：新接口中更换真实目标 token 不会改变 16 个 MASK 输入，只改变交叉熵标签；推理一次得到 `[B,16,V]` logits 再逐位置选择 ID，没有把同块某个采样 ID 再喂回其他位置。训练前向与推理调用数的区别见第 2、13 节。

纯 tokenizer 重建训练不使用这个生成调度。因此按相同配置重训 tokenizer 属于重新训练数值上相同的模型；实验效果需看后续 **VLA 重训和 rollout**。同权重、同输入时编码和重建相同，但独立训练的两个 run 不应未经核对就宣称权重逐位相同。若要严格固定 tokenizer 权重做消融，可另行做显式 checkpoint 身份转换；本次没有静默改写旧 checkpoint。

### 14.2 代码与选择接口

- [Tokenizer 实现](../../src/praxis/codecs/tokens/btt_oneblock/tokenizer.py)、[配置类型](../../src/praxis/codecs/tokens/btt_oneblock/config.py)。
- [Tokenizer Hydra 配置](../../src/praxis/configs/codecs/tokens/btt_oneblock.yaml)：直接继承标准 BTT 的数值配置；选择 `codecs@codec=tokens/btt_oneblock`。
- [BAR 变体注册](../../src/praxis/policies/bar/variants.py)：Pali、Qwen、Smol 均有对应新 policy。
- 主要实验选择 `policy=bar_btt_oneblock_pali`；另有 `bar_btt_oneblock_qwen`、`bar_btt_oneblock_smol`。
- 保存的 tokenizer `config.json` 中 `name=btt_oneblock`，加载后保持 `(16,)`；不接受把原 `name=btt` checkpoint 直接当作新接口加载，也不允许非严格权重加载。
- 不使用 `policy.block_schedule=[16]` override：现有 policy 不开放此配置，调度从新 tokenizer 属性读取。

### 14.3 训练参数与 W&B 的一致性

[配置准备脚本](../../scripts/tokenizer/prepare_btt_oneblock_training.py) 复制已经执行过的原 run 的 `resolved_config.yaml`，不重新拼当前默认值。逐字段恢复允许修改的身份字段后必须与原快照完全相等；同时写出原快照 SHA256 和 `.audit.json`。

Tokenizer 参照：`outputs/hopper_btt_robotwin_new/621340.hopper-m-02/resolved_config.yaml`。仅修改 `codec.name`、`run_name`、`output_dir`。

| Tokenizer 训练项 | 复用原设置 |
| --- | --- |
| 数据 | `local/robotwin_new_demo_clean`，原数据路径和采样配置 |
| Steps / batch / seed | 300,000 / 512 / 42 |
| Optimizer | LR 5e-5，betas (0.9,0.95)，eps 1e-8，weight decay 0 |
| Scheduler | constant，warmup 200 |
| Precision / EMA | bf16；EMA 0.995 |
| Validation / checkpoint | 每 1,000 / 5,000 step；原 5% 划分与 split seed |
| W&B | `use_wandb=true`，project=`tokenizer_pretrain`，entity=null |
| 新 run 名称 | `robotwin_new_btt_oneblock_tokenizer` |
| 新输出根目录 | `outputs/hopper_btt_oneblock_robotwin_new` |
| PBS 资源 | 与原 tokenizer 一致：1×H200，12 CPU，225GB，72h |

VLA 参照：`outputs/hopper_bar_btt_pali_robotwin_new_step300000_30k/630822.hopper-m-02/resolved_config.yaml`。仅修改 policy/tokenizer 名称、新 tokenizer checkpoint 路径、run 名和输出目录；保留原 PaliGemma2 3B 初始化、冻结 tokenizer、30,000 steps、batch 64、seed 42、bf16、optimizer/scheduler、数据和 W&B project=`praxis-vla`。W&B 默认 online，使用现有凭据，不在代码中写入密钥。

配置中仍为 `block_ce_weighting=block_mean`；等长的两个 8-token block 与一个 16-token block 对同一组逐位置 CE 都等价于 16 个位置的平均，不因合块额外放大 loss。模型 logits 会因条件输入与 attention 改变而变化。

### 14.4 训练入口

新入口均采用独立目录，并拒绝覆盖已有 run 的配置。`PREPARE_ONLY=1` 只生成及重读 Hydra 配置，不启动训练或 W&B。

```bash
# 从仓库根目录提交 tokenizer 重训
qsub artifacts/robotwin_hopper/train_btt_oneblock_robotwin_new.pbs

# 使用新的 btt_oneblock 训练产物提交 Pali VLA
qsub -v TOKENIZER_CHECKPOINT=/absolute/path/to/new/checkpoint/pretrained_model \
  artifacts/robotwin_hopper/train_bar_btt_oneblock_pali_robotwin_new_30k.pbs

# 只检查 tokenizer 配置与启动命令
PREPARE_ONLY=1 bash artifacts/robotwin_hopper/train_btt_oneblock_robotwin_new.sh
```

VLA launcher 要求新 checkpoint 的 `config.json`、`model.safetensors`、`norm_stats.pt`、`policy_io.json` 完整且身份为 `btt_oneblock`；不从旧 BTT VLA 的 30k checkpoint 继续微调。

### 14.5 已完成的验证与边界

- 原 BTT、codec factory、policy catalog/registry、训练和推理调度及新增接口的回归检查通过。
- 同随机初始化的标准/新 tokenizer：state dict、编码 ID、重建、反向梯度逐项一致；新接口全尺寸参数量为 7,382,802。
- 工厂构造、checkpoint 保存/加载、normalization 和 PolicyIO sidecar 保留新身份及 `[16]` 调度。
- 实际 BAR 推理循环搭配 dummy VLM：只有一次 forward，覆盖 16 个预测位置，输入全部 MASK，16×16 块内 attention 允许双向访问。
- 实际训练输入构造：16 个 MASK、同位置真实 labels；改变目标只改变 labels。
- Tokenizer 和 VLA 两种 launcher 均完成 PREPARE_ONLY 配置重读检查；VLA 检查使用临时小 tokenizer fixture，没有用它训练。
- Python lint、脚本语法和原 BTT 源码未修改检查通过。

检查记录：[verification.json](../../outputs/btt_oneblock_implementation_check_20260929/verification.json)。本轮接口测试没有运行完整 PaliGemma 3B 前向；尚无完整 VLA latency 或 rollout 实测结果，一次调用不代表完整耗时必然减半。

### 14.6 提交状态

已在 2026-09-29 提交两阶段训练，并核对 PBS 依赖关系：

| 阶段 | PBS job | 最近核对状态 | W&B project |
| --- | --- | --- | --- |
| Tokenizer 300k | `638302.hopper-m-02` | `Q`，排队；调度器显示可用内存资源不足 | **`tokenizer_pretrain`** |
| 单 block Pali VLA 30k | `638303.hopper-m-02` | `H`，等待 tokenizer 成功完成，系统依赖 hold | `praxis-vla` |

VLA 依赖为 `afterok:638302.hopper-m-02`，输入 checkpoint 固定为新 run 的最终 300k step：

```text
outputs/hopper_btt_oneblock_robotwin_new/638302.hopper-m-02/train/checkpoints/recent/step_300000/pretrained_model
```

使用 `recent/step_300000` 保证以固定训练预算的最终 step 比较，不依赖新 run 的 best 排名恰好包含该 step。两阶段输出分别位于 `outputs/hopper_btt_oneblock_robotwin_new/638302.hopper-m-02` 和 `outputs/hopper_bar_btt_oneblock_pali_robotwin_new_30k/638303.hopper-m-02`。

提交成功不等于训练完成；上述核对时两个训练尚未启动，W&B run 尚未创建。Tokenizer 始终沿用 **`tokenizer_pretrain`**，只有后续 VLA 沿用原 VLA 基线的 `praxis-vla`。
