# G1 官方下半身策略与统一全身执行层

2026-10-13：已完成官方策略的离线集成、数值稳定性修复、官方Dex1模型适配、位置保持外环，以及统一全身控制器的仿真实现。**可靠位置反馈下通过180秒筛选，但仅关节/IMU推算的长期漂移尚未解决，不能上真机。** 原debug、arm_sdk与采集入口继续保留，默认仍为arm_sdk。

完整结果、失败案例和轨迹哈希见[本轮验证报告](../../artifacts/unitree_g1/whole_body_integration_20261013.json)。上一轮未经修复的结果保留在[初次报告](../../artifacts/unitree_g1/lower_body_validation_20261013.json)，不替换或隐藏。

## 源码与模型

- 官方[unitree_rl_gym固定提交](https://github.com/unitreerobotics/unitree_rl_gym/tree/276801e46c5d433564f24658bac64f254b7d2d4b)与[锁文件](../../artifacts/unitree_g1/lower_body.lock.json)：CPU TorchScript、50Hz、47维观测、12维腿部动作。模型、配置、身体模型及网格全部检查SHA256。
- 官方[unitree_ros固定提交](https://github.com/unitreerobotics/unitree_ros/tree/5994d4faef0a9cadd3287f8de0199a67eeb2a259)与[Dex1锁文件](../../artifacts/unitree_g1/dex1_model.lock.json)：取mode15的Dex1子树，单手质量0.364946kg。用户确认暂无实测数据，先用官方模型。
- 保留既有mode5/29自由度rev1.0身体，不换成mode15新版腰腕电机。原MJCF将0.17kg橡胶手融合进腕部惯量；先从匹配URDF恢复裸腕惯量，再添加Dex1，避免重复计算。每腕增加质量净值约0.194946kg。
- 夹爪使用官方质心、完整惯量、碰撞网格、两个耦合的移动关节。沿用mode15给出的腕部X向+0.0415m安装位姿，**尚未实测校准mode5安装关系**。
- 额外安装质量和左手载荷为明确标注的球形惯量敏感性模型，不冒充相机、支架或瓶子的精确CAD。
- 原官方文件未修改；[模型构建器](../../scripts/unitree_g1/g1_sim_model.py)在内存生成覆盖模型。上游文件各自保留原许可。

## 数值稳定性和站立外环

旧实现将PD力矩在Python端算完交给MuJoCo，积分器看不到阻尼的速度导数。新实现把同一连续时间PD表达为`kp*target - kp*q - kd*dq`的原生仿射执行器，并使用`implicitfast`。原力矩限幅保留，离线测试直接对照两种实现的关节力矩。依据：[MuJoCo积分器说明](https://mujoco.readthedocs.io/en/3.3.7/computation/index.html#integrators)。

2ms不再快速发散，但足部抬升指标仍可能超限；采用1ms作为候选物理步长，并以0.5ms交叉复核。只宣称当前场景下数值稳定、细化结果相近，不宣称所有场景严格收敛。策略依然50Hz，不随物理步长改变。TorchScript在控制前预热且不推进动作历史。仿真警告、自动重置、非有限状态均拒绝。

官方步态策略的零速度命令不保证站在原地。[位置保持外环](../../scripts/unitree_g1/standing_reference.py)使用新鲜世界系位置/速度和偏航，将相对起点的位置、速度和有限积分项转换为受限策略速度命令；平移限幅±0.3m/s，偏航±0.4rad/s。预训练模型权重未改。

明确区分两种位置来源：

1. `simulation_truth`：仿真真值，仅用于证明闭环结构和动力学能力，机器人LowState不提供该数据。
2. `kinematic`：[试验性足底里程计](../../scripts/unitree_g1/stance_odometry.py)，仅由关节位置、IMU姿态和足底FK计算，不读取仿真根部位置/速度；假设平地且至少一个支撑点不滑。180秒失败，**不得作为已验收真机定位源**。缩小接触高度阈值的探索也未消除累积漂移，不能用短时通过替代长期验证。

候选双肘中立角为0.5rad，其余初态沿用官方目标；这只是仿真起始姿态，不自动在真机摆姿或切换模式。基准0rad初态失败保留在历史数据中。

## 统一执行层

[控制核心](../../scripts/unitree_g1/whole_body_control.py)独立于旧`run_hold`，不复用全身卸载逻辑。

| 槽位 | 来源 |
| --- | --- |
| 0–11 | 官方策略腿部目标；最终按URDF限位裁剪，保留原始动作及裁剪索引 |
| 12–14 | 接管姿态保持，或显式启用模型/IMU腰部补偿 |
| 15–28 | 既有Quest关节限位、增益、速度/加速度限制及目标接口 |
| 31、33 | 既有Dex1扳机位置映射、预全开、闭合冻结逻辑 |
| 29、30、32、34 | 禁用，不写作上肢SDK权重 |

阶段为`STANDING → PREOPEN → READY → ACTIVE → UPPER_HOLD`。预全开在同一全身帧内进行；仿真中约3.15秒完成后进入任务。正常松手、输入断流、相机故障或记录队列失败只停止上肢任务，保留腿部、腰部、手臂及夹爪增益；不发旧debug的全身零增益尾帧。上肢冻结是候选保护动作，不等于经过验证的平滑物理停车。

反馈/策略来源时间上限40ms、单次推理上限20ms、伺服间隔上限20ms；检测时间回退、刷新冻结tick、模式不符、倾角、关节和电机故障。下半身错误锁存并交给外部已验证的全身故障处置，不伪造“保持旧腿目标即可站住”。新流程尚无真机启用入口。

[离线运行组件](../../scripts/unitree_g1/whole_body_runtime.py)提供独立推理线程、只保留最新输入的邮箱、单一伺服线程指令提交及有限容量记录队列；伺服线程不推理或写盘。仅注入仿真/影子输出端，无DDS发布器工厂。CRC编码已离线验证，但真实DDS独占、服务切换、反馈适配和机身定位尚未接入。

仿真轨迹包含发送前状态、IMU、策略输入/输出及来源时间、完整35槽合成指令、上肢阶段、16维动作和发送后物理状态。16维仍按左臂7＋左夹爪＋右臂7＋右夹爪提取。**它们标记为仿真数据，不是机器人实际提交记录，也未开放LeRobot导出。** 转换器只接受既有debug或历史未标记模式，明确拒绝新全身/仿真及未知模式，即使允许shadow也不能绕过。正式相机同步、IK进程及录制生命周期接入仍待真机控制门槛满足后完成。

## Dex1与载荷补偿的边界

仿真中的Dex1移动关节单位为米。统一预全开演示使用显式标注的仿射替代映射：电机−5.4～0对应指间移动范围+0.0245～−0.02m；该映射、指端接触和20N移动关节驱动**未经真实传动标定**，不能用于声称电机力矩/夹持力验收，也不能替代现有31/33电机位置标定。

[上肢负载补偿](../../scripts/unitree_g1/upper_body_balance.py)按配置好的官方模型与额外质量，用FK估计相对中立姿态的质心变化，计算腰部roll/pitch修正，并加入0.25比例的IMU小幅修正。腰部补偿限幅±0.2rad、速度0.3rad/s、加速度0.6rad/s²；这是新增候选控制，不是官方预训练策略自带功能。它知道测试中配置的载荷，**不能识别未知物体质量**。

## 本轮主要结果

沿用原筛选标准：倾角≤0.2rad、漂移≤0.15m、足部link抬升≤0.03m、实测关节限位超出≤0.01rad。足部link抬升不是精确步数或双脚始终接地的证明；本策略仍可能小幅踏步。

| 场景 | 时长 | 最大漂移 | 最大倾角 | 结论 |
| --- | --- | --- | --- | --- |
| Dex1＋位置外环，1ms / 0.5ms | 各20秒 | 6.0 / 5.6cm | 0.183 / 0.170rad | 均通过 |
| 统一控制、预全开、合成左臂/夹爪动作、退出后保持；仿真定位 | 180秒 | 5.7cm | 0.195rad | 通过 |
| 相同统一控制，足底里程计 | 180秒 | 112.5cm | 0.213rad | **失败** |
| 每腕+200g、左手+500g，未补偿 | 20秒 | 466.7cm | 0.339rad | **失败** |
| 相同载荷＋模型/IMU补偿，1ms / 0.5ms | 各60秒 | 6.2 / 5.1cm | 0.189 / 0.180rad | 均通过，使用仿真定位 |
| 横向10N / 20N、0.2秒推扰 | 各20秒 | 5.7 / 6.4cm | 0.194 / 0.192rad | 通过 |
| 相机失败 / 上肢输入超时 | 各20秒 | 5.7cm | 0.194rad | 上肢保持后下半身继续 |
| 推理延迟20ms / 60ms | 20秒 / 约5秒 | — | — | 前者通过；后者正确锁存`policy_deadline` |

153项离线回归：152通过，1项LeRobot数据环境专用测试跳过。覆盖PD力矩等价、质量替换、质心补偿、观测/动作历史、预热、指令槽位/CRC、上肢退出与故障分流、记录失败、推理超时和单伺服所有者。所有验证均无真机发布。

## 复现

沿用`.venv-g1-xr`。独立仿真依赖：

```bash
.venv-g1-xr/bin/python -m pip install --target .local/unitree_g1/sim_deps --no-deps mujoco==3.3.7 glfw==2.10.0 PyOpenGL==3.1.10 absl-py==2.3.1 etils==1.13.0
```

源码恢复（仅首次，需要网络）：

```bash
 git clone https://github.com/unitreerobotics/unitree_rl_gym.git .local/unitree_g1/unitree_rl_gym
 git -C .local/unitree_g1/unitree_rl_gym checkout --detach 276801e46c5d433564f24658bac64f254b7d2d4b
 git clone --filter=blob:none --sparse https://github.com/unitreerobotics/unitree_ros.git .local/unitree_g1/unitree_ros
 git -C .local/unitree_g1/unitree_ros sparse-checkout set robots/g1_description
 git -C .local/unitree_g1/unitree_ros checkout --detach 5994d4faef0a9cadd3287f8de0199a67eeb2a259
```

使用全新的输出路径；单场景未通过返回2。程序无`--execute`或网卡参数：

```bash
.venv-g1-xr/bin/python scripts/unitree_g1/sim_lower_body.py --dex1 --position-hold --elbow .5 --unified --arm-motion --gripper-cycle --seconds 180 --output .local/unitree_g1/experiments/whole_body_new.json
.venv-g1-xr/bin/python scripts/unitree_g1/sim_lower_body.py --dex1 --position-hold --elbow .5 --unified --arm-motion --gripper-cycle --extra-mass .2 --payload-mass .5 --load-compensation --seconds 60 --output .local/unitree_g1/experiments/whole_body_load_new.json
.venv-g1-xr/bin/python scripts/unitree_g1/validate_lower_body.py --output-dir .local/unitree_g1/experiments/whole_body_matrix_new
.venv-g1-xr/bin/python -m unittest discover -s tests/unitree_g1 -v
```

全矩阵包含故意失败和压力场景，不能仅以脚本退出码视作全通过；检查`summary.json`。可重复`--case`只运行指定场景。原始轨迹在`.local`，不随Git恢复，报告保存汇总、源码版本和轨迹哈希。

## 真机前仍需完成

2026-10-13 已新增真实反馈影子诊断 `scripts/unitree_g1/shadow_lower_body.py`，仅订阅LowState并运行策略，记录关节、IMU、47维观测和未发送目标，不创建发布器。20秒1001次推理完成，但首帧目标差2.070rad、最大目标差2.479rad；必须先验证起始姿态过渡，禁止将原始输出直接接管。影子动作历史会演进，机器人不跟随，因此结果不证明闭环平衡。详细结果见[报告](../../artifacts/unitree_g1/lower_body_shadow_20261013_001.json)。

只读复现（输出路径须不存在）：

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 .venv-g1-xr/bin/python scripts/unitree_g1/shadow_lower_body.py --interface enp130s0 --seconds 20 --output /tmp/g1_shadow_new.jsonl
```

1. 提供可靠机身世界系定位，或训练/验证不依赖持续踏步的站立策略；当前足底里程计没有解决长期漂移，禁止静默降级使用。
2. 校准mode5上的Dex1安装、相机/支架质量与电机到指间运动关系；未知负载需要估计或明确限制。
3. 验证真实反馈接入、唯一DDS发布器、下半身故障独立停止、接管和持续平衡期间退出流程。
4. 接入真实Quest、相机及录制进程，再独立验证新模式的16维导出和现场支撑下试验。
