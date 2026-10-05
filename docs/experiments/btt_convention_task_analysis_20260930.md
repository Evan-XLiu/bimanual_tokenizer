# BTT convention：麦克风交接与扔瓶子的反向结果

分析日期：2026-09-30。

**最明确的发现：标准 BTT 的 handover_mic 问题高度集中在左→右交接；右→左方向与 w/o convention 基本相当。put_bottles_dustbin 的交接恰好固定为右→左，并不是不包含交接的“对称任务”。** 这比“convention 对对称任务有利、对非对称任务有害”更符合现有证据。

本次读取实际 episode records、运行/训练配置、任务代码和 tokenizer/VLA 实现；抽查两模型同种子视频；重新汇总已保存的 GT tokenizer 重建，补算 FK；对全部 500 条麦克风评测视频首帧分类初始方向。没有重新训练或运行 VLA rollout。机制解释与已验证现象分开陈述。

## 1. 实际目录与完整结果

用户的 w/o 路径实际包含目录分隔符：outputs/robotwin_eval/rollout8task_100/BTT_w/o_convention。

每个表格单元是成功数/100，来自对应 comparison.json 并用 episode_records.jsonl 重算。

| 任务 | BTT | w/o convention | BTT − w/o | OAT | QueST | BOAT | Acodec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shake_bottle_horizontally | 100 | 93 | +7 | 93 | 96 | 93 | 98 |
| place_container_plate | 92 | 88 | +4 | 90 | 92 | 85 | 83 |
| stack_blocks_two | 66 | 66 | 0 | 47 | 43 | 49 | 34 |
| dump_bin_bigbin | 85 | 85 | 0 | 78 | 69 | 70 | 60 |
| stack_bowls_three | 81 | 64 | +17 | 50 | 55 | 54 | 54 |
| place_bread_skillet | 47 | 26 | +21 | 8 | 10 | 6 | 25 |
| put_bottles_dustbin | 49 | 24 | +25 | 20 | 20 | 20 | 37 |
| handover_mic | 71 | 95 | −24 | 94 | 90 | 77 | 72 |
| 八任务平均 | 73.875 | 67.625 | +6.25 | | | | |

标准 BTT 在五项更好、两项持平、一项更差。麦克风交接的异常值得定向诊断，不支持整体删除 convention。

## 2. 隐藏在总分中的交接方向差异

麦克风任务根据初始 x 位置选择 donor arm，对侧为 receiver。首帧画面左右与世界 x/任务手臂对应：机器人相机配置的 left 向量是世界 −x。

| 模型 / VLA checkpoint | 左→右 | 右→左 | 总成功率 |
| --- | ---: | ---: | ---: |
| BTT 15k | 23/50 = 46.0% | 50/50 = 100.0% | 73% |
| BTT 19k | 25/49 = 51.0% | 49/51 = 96.1% | 74% |
| BTT 23k | 21/47 = 44.7% | 52/53 = 98.1% | 73% |
| BTT 30k | 23/49 = 46.9% | 48/51 = 94.1% | 71% |
| w/o convention 30k | 45/47 = 95.7% | 50/53 = 94.3% | 95% |

这不是只有最终 30k checkpoint 才出现的下降。四个 BTT checkpoint 来自同一训练轨迹，不能当成四次独立训练复现，但都呈现相同方向性问题。

BTT 30k 的 29 次失败有 26 次来自左→右。两模型共有的 90 个场景种子进一步分解为：

| 相同场景 seed 子集 | 数量 | BTT 成功 | w/o 成功 |
| --- | ---: | ---: | ---: |
| 左→右 | 39 | 16 | 38 |
| 右→左 | 51 | 48 | 48 |

共同场景没有消除语言指令差异，因此这里只能称为“同场景 seed 比较”，不能称为完整受控配对实验。两个模型共同种子的初始画面侧别完全一致。

方向分类方法：逐视频读取第 0 帧，在 y=[65,190)、左 x=[35,125)、右 x=[215,305) 的区域计数灰度小于 130 的像素；要求目标侧 >100 像素、另一侧 <40 像素，否则拒绝分类。全部 500 视频通过；30k 两模型的目标侧至少 409 像素，另一侧均为 0。已生成首帧联系表供人工复核。

训练集方向也不是近乎单边：50 个麦克风 demonstrations 中，按第一个夹爪跨越 0.5 阈值闭合判定 donor，左 23、右 27；扣除既有验证 split 的四个 episodes 后，训练子集左 21、右 25。此阈值只是判定先抓取者的离线代理，不替代仿真成功判据。

## 3. 任务性质：两者都不等于“两个臂做相同动作”

### handover_mic

任务代码 envs/handover_mic.py：

- 初始麦克风在左/右两端，三种模型；近侧手先抓取（contact IDs 1、9…15）。
- donor 抬升、把同一个物体送到中央 pose [0, −0.05, 0.98, 0, 1, 0, 0]。
- receiver 从另一组接触点（0、2…8）接取；之后 donor 开夹爪、两臂分离。
- 成功要求 receiver 夹爪关闭、donor 夹爪打开、麦克风高于 0.92m 且位于接收侧，并且存在物体接触。

因此核心是不同角色之间的接触几何、持物稳定性和释放顺序。它需要互补动作，并不要求两臂 feature 相等。任务在两个方向上都出现，尤其能暴露固定左/右身份与 donor/receiver 角色之间的偏差。

抽查同种子 100001、100013、100039、100065、100100、100130：标准 BTT 都能先抓取并带到中央附近，之后出现停滞、交接不成或掉落；w/o 最终成功。有些 w/o 也需要更久，例如 100039 用 456 步、100100 用 413 步，因此不是“去掉 convention 后一开始就完全不同”。这些是选取的具体样例，不是 29 条失败的穷尽阶段标注。没有保存接触力/逐步 action trace，不能仅凭画面把所有失败精确归为夹爪早开或接收位姿偏差。

### put_bottles_dustbin

任务代码 envs/put_bottles_dustbin.py：

- 三个瓶子，垃圾桶固定在左侧 x=−0.45。
- 左边瓶子：左手抓取→搬到桶上→释放。
- 右边瓶子：右手抓取→中央交接→左手接住→右手释放/回位→左手扔进桶。
- 右手抓取相对上移 0.06m，左手接取相对下移 0.06m；角色与抓取高度有明确分工。
- 所有三个瓶子都进入目标区域才算整集成功。

**该任务的交接固定为右→左，恰好避开标准 BTT 在麦克风上的困难方向。** 所以 BTT 在此更强并不矛盾，也不能据此宣称“BTT 擅长所有交接”。

同 seed 100009、100049、100132 中，w/o 分别在早期抓取/搬运阶段把红瓶、透明瓶或黄瓶弄倒，后续未能清完；BTT 完成三瓶。这些样例说明瓶子差距不都发生在中央交接阶段。

成功 episode 的步数中位数：麦克风 BTT 208、w/o 209；扔瓶子 BTT 649、w/o 647.5。扔瓶子的流程明显更长，包含重复抓取、角色切换、放置和回位。每个阶段的偏差都可能通过物体位置改变和后续状态分布积累。不能把三个瓶子的成功概率假定独立，但“必须全完成”的评价确实使局部失误更容易成为整集失败。

## 4. 纠正对 convention 的数学理解

S=diag(−1,+1,+1,+1,−1,−1,+1)。标准 BTT 先把右臂转换为 S q_R，再让两臂通过同一个逐关节 affine normalization N 和共享编码器 F：

    h_L = F(N(q_L))
    h_R = F(N(S q_R))

w/o 则使用 S=I，同时使用重新 pooled 的 raw normalization 范围。两者仍然共享 encoder、self/cross attention、pre_quant 和 FSQ；不是把 w/o 改成独立双臂编码器。

若整个输入动作窗口满足 q_L=S q_R，则在 eval 的确定性路径中共享单臂 features 一致；对同一对完全相同输入，对称 cross-attention 也输出相同结果。只在某一时刻相等不够，因为 encoder 使用时间上下文；跨不同样本比较最终 coordinated feature 时，还要考虑另一条臂的上下文。

但有三点不能混淆：

1. S 可逆，S²=I；归一化也可逆且配置不 clamp，预处理本身没有抹掉信息。
2. token 仍按 [L0…L7,R0…R7] 保留左右位置；decoder 对每个时刻 concat(L_t,R_t)，不是求和/平均，所以同 ID 不等于“不知道是哪条手”。
3. 严格成立的是 encoder 对 canonical 双臂交换的等变关系，不是整个 VLA / decoder / 环境在交换后都具有对称性。decoder 的 concat→Linear、14 维 action head，以及固定左右顺序的 VLA 都没有被强制镜像等变。

自然 GT 数据里，同一时刻 L/R token 相等比例很低：麦克风 BTT 297k 为 4%，BTT 300k 和 w/o 为 6%；扔瓶子约 2%。因此没有证据支持“两只手都编码成同一个 feature，接收/递交身份彻底消失”的直接解释。

## 5. 离线重建没有解释闭环结果反转

复用完全相同的验证数据和 cached reconstructions，计算 raw action MSE 和 URDF FK。BTT 297k 是当前 rollout 实际 tokenizer；补列 BTT 300k 控制 tokenizer 训练 step。麦克风只有 4 个 validation episodes / 25 chunks；扔瓶子只有 3 个 / 56 chunks。以下是描述性点估计，不代表训练 seed 方差或大量独立样本。

| 任务 | tokenizer | Raw action MSE | 相对末端位置误差 cm | 相对末端旋转误差 deg |
| --- | --- | ---: | ---: | ---: |
| 麦克风 | BTT 297k | 0.0004150 | 1.1189 | 2.2806 |
| 麦克风 | BTT 300k | 0.0004095 | 1.0331 | 2.3447 |
| 麦克风 | w/o 300k | 0.0003589 | 1.0369 | 2.4315 |
| 扔瓶子 | BTT 297k | 0.0003307 | 1.1322 | 2.2191 |
| 扔瓶子 | BTT 300k | 0.0003470 | 1.0452 | 2.2088 |
| 扔瓶子 | w/o 300k | 0.0002897 | 1.0043 | 2.2337 |

尤其扔瓶子：w/o 的 GT 重建 MSE/相对位置误差点估计更低，rollout 却为 24% vs 49%。说明现有结果不能归结为“标准 BTT 还原动作更准”；需要考虑 VLA 生成的 token 是否正确、错误 token 配对的 decoder 行为，以及进入非专家状态后的反馈。

已有全验证集 815 chunks 的 matched-300k 结果则是标准 BTT MSE 0.0005196，w/o 0.0005717。全任务平均和任务局部趋势不同，也不能用全局 MSE 代替关键阶段诊断。

已有 matched-300k 词表统计：BTT 使用 1677/1920 IDs，w/o 使用 1500；左右臂 token 频率 JSD 分别 0.1512 / 0.2657 bit。支持 convention 使两臂编码分布更接近，但 JSD 不证明角色混淆，更不直接证明成功率因果。

## 6. 最值得检验的机制

最合理的当前工作假说是 **共享 canonical 表示与下游左右角色/生成顺序的相互作用**，而不是 sign 映射在数学上删除了手臂身份。

当前 VLA：训练预测 L 时只看观测 O，预测 R 时看 O 和真实 L；推理则先得到 L_hat，再以 O 和 L_hat 预测 R_hat。左→右交接时，接收方恰好是第二个 block 的右手。这个方向要求右手根据实际持物几何与释放时机作出稳定接取，可能更易受第一 block 预测误差、teacher forcing 条件差异或共享 token 语义带来的错误关联影响。

去掉 convention 后左右 token 分布更分离，可能给角色区分提供更显式的线索，也可能只是改变了不同动作的量化分区与 VLA 学习难度。现有数据没有测量 token 复制率、角色混淆矩阵或自由生成条件下的右臂误差，因此不能把“发生了复制 shortcut”写成结论。相同 LR 生成顺序在 w/o 仍然存在，所以顺序单独也不是充分解释，必须考虑它与表示/训练的交互。

扔瓶子则复用固定右→左交接，并多次使用左右单臂抓取/搬运。convention 可能帮助共享动作模式、降低多任务 VLA 的学习负担，收益在较长流程中表现出来；目前视频和成功率与这个解释一致，但尚未做能将其与其他训练差异隔离的因果实验。

### 补充：关节符号对称不等于 URDF 精确空间镜像

额外使用已验证的 RobotwinFK 做合成检查：保留记录的左臂 q_L，构造 q_R=S q_L，比较右末端位置与左末端关于世界 x=0 的反射。

- 全部 26080 waypoints：平均偏差 1.0041cm、P95 1.9163cm、最大 2.4318cm。
- 麦克风的 800 个合成 waypoints：平均 0.7922cm、最大 0.9713cm。
- 扔瓶子的 1792 个合成 waypoints：平均 0.8353cm。

这些数值是 **假设两臂严格满足符号关系时的机械几何镜像残差**，绝不是 rollout 中量到的接取偏差或 tokenizer 重建误差。URDF 的对应链使用相同侧向 offsets/小旋转，不是所有参数都做几何反射。此次合成 FK 又用原始 URDF 的 SAPIEN CPU 独立核验了 256 个 waypoints、512 臂，最大位置差 5.52e−7m，通过 1e−5m 阈值。

所以“同 canonical joint feature”应解释为共享关节动作约定，而非保证末端严格镜像。微小几何差别是精细接触任务可检验的因素；左右保留身份的 decoder 仍可学习补偿，现有证据不能说这个残差必然导致 24 个百分点差距。

## 7. 比较边界与下一步优先级

已核对两份 VLA 训练配置，只有 tokenizer 路径、run_name 和 output_dir 不同。均为相同 Pali 初始化、数据、seed=42、30k steps、batch=64、16 action steps。tokenizer 实际是 BTT 297k vs w/o 300k；不是同一模型推理时简单关 sign。w/o 的 sign 与 pooled normalization 范围同时改变，归一化 MSE 的物理关节权重也随之改变。

两次评测的 root seed=0、demo_clean、seen、各 100 集和任务 step limits 相同，但实际接受的场景种子不完全相同。麦克风共 90 个，扔瓶子共 89 个；共同种子的实际语言字符串分别 0/90、0/89 相同。

原因在当前源码中可解释：官方指令生成会调用 Python random.shuffle/random.choice，而环境 seed 只设置 numpy/torch，random.seed 被注释；外层固定 PCG64 只固定候选索引，不能固定候选文本。此外种子预检查的接受集合不同，按 accepted episode 消耗指令 RNG 也会错位。应固定并回放完整 (scene seed, instruction) 清单，而不是只设置相同 root seed。

两份运行记录 source commit/overlay 不同；对应 commit 间 BTT tokenizer、BAR core 和 core/vlm 文件的 git diff 无变化，overlay 没有直接替换 BTT encoder/decoder，但没有重跑全过程建立严格因果隔离。也只有单训练 seed。

建议依次做：

1. **固定实际场景和完整指令**，控制 tokenizer step=300k、VLA step=30k，按左右 donor 和物体型号分层评测。先确认方向差距在受控输入下依然存在。
2. **定位右臂生成误差**：在相同交接前观测上，分别比较 P(R|O,L_GT) 与 P(R|O,L_hat)，解码后测 receiver 末端相对持物点误差、夹爪事件时间、角色错误，并按 donor 方向分开。不仅比较 token CE。
3. **测试固定生成顺序的作用**：同 tokenizer 分别训练 LR、RL、oneblock 的 VLA；评测相同左右方向子集。若偏差随顺序反转，支持顺序交互；若始终右 receiver 困难，更应检查右手视觉、几何/表示与角色学习。不能只在旧 checkpoint 推理时交换 block。
4. **保留共享、加强身份或角色条件的受控消融**：可比较 arm embedding 或小型 arm-specific adapter。donor/receiver 角色若作为输入，必须由观测/语言可推断，不能把未来 GT 角色标签泄漏给策略。用多训练 seed 验证，不直接删除 convention。

## 8. 可复核产物

- ../../outputs/btt_convention_task_analysis/20260930/summary.json：全部任务、同 seed、局部重建/FK/token 统计。
- ../../outputs/btt_convention_task_analysis/20260930/rollout_summary.csv：六模型八任务。
- ../../outputs/btt_convention_task_analysis/20260930/direction_summary.json：五个 checkpoint 的方向统计。
- ../../outputs/btt_convention_task_analysis/20260930/mic_image_side.json：每集记录、分类与像素计数。
- ../../outputs/btt_convention_task_analysis/20260930/mic_training_donor.json：50 个示范的 donor 判定。
- ../../outputs/btt_convention_task_analysis/20260930/geometry_probe.json：合成镜像检查；synthetic_fk_verification/fk_verification.json：SAPIEN 核验。
- 同目录 handover_mic_seed*.jpg / put_bottles_dustbin_seed*.jpg：同场景 seed 的视频联系表。另有两模型全部初始画面联系表。
- analyze.py、direction_breakdown.py、geometry_probe.py、contact_sheets.py：本次统计和制图代码。

核心源码定位：src/praxis/codecs/tokens/btt/{config,encoder,tokenizer,decoder,modeling}.py；src/praxis/policies/bar/core/vlm/{decode_tokens,decoding}.py；相邻 RoboTwin checkout 的 envs/{handover_mic,put_bottles_dustbin}.py 和 description/utils/generate_episode_instructions.py。
