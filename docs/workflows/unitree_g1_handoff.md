# G1 / Quest 3S 交接入口

更新：2026-10-10。本文只保留当前配置、验证边界和接手入口；逐次试验细节见对应工作流及 `artifacts/unitree_g1/` 报告。历史现场信息不代表实时机器人状态。

## 当前状态

- 已打通 Quest 3S → 本机受限 IK/执行层 → 有线 → G1 双臂与 Dex1 夹爪。用户曾确认手臂运动和夹爪开合正常；当前更高夹爪速度、接触保持逻辑及180秒采集尚未真机验收。
- 当前目标是采集用于模仿学习的三路图像、机器人状态和动作，导出 LeRobot v3。软件链路已完成离线验证，尚无正式真机训练数据。
- 用户当前无法连接机器人、相机和 Quest，要求先完成离线工作；不启动外设连接或机器人控制。
- 用户于2026-10-10手动清空 `.local/unitree_g1/experiments/`，已检查确认为空。历史原始日志、Quest输入、相机样张、会话源码快照及合成样例均已删除。报告中的旧原始路径已失效，不能再回放、重算或核验原文件哈希；总结报告及历史结论保留。

## 当前控制配置

参数来源：[quest_bridge.py](../../scripts/unitree_g1/quest_bridge.py)、[quest_grippers.py](../../scripts/unitree_g1/quest_grippers.py)及控制profile；操作细节见[Quest工作流](quest_g1_bridge.md)。

| 项目 | 当前设置 |
| --- | --- |
| 单次 ACTIVE 上限 | **180秒**，接管和卸载另计；故障或手动退出可提前结束 |
| 手柄平移比例 / 方向 | **1.0**；仅控制位置，末端方向自由，手柄旋转不控制末端方向 |
| 请求工作区 | 相对 ACTIVE 接管起点，前后/左右各 ±10 cm，向上0–10 cm |
| 双臂关节范围 | 初态 ±0.20 rad 与 URDF 限位取交集 |
| 双臂速度 / 加速度 | 0.05 rad/s / 0.1 rad/s² |
| 双臂增益 | 肩肘 Kp/Kd=80/3，腕部40/1.5；前馈力矩为零 |
| IK / 跟踪保护 | 位置残差≤1.5 cm；Quest关节跟踪误差≤0.10 rad；基础保持profile仍为0.08 rad |
| 输入 / 反馈时效 | 0.1秒 |
| 夹爪接口 | 内走线 Dex1，`rt/lowcmd` 左31/右33，与双臂共用一个发布器 |
| 夹爪位置 | 负方向打开、正方向闭合；全开目标−5.4，闭合使用已记录的左右近闭合参考 |
| 夹爪打开/闭合速度 / 加速度 | **1.2 电机位置单位/秒 / 0.6 电机位置单位/秒²** |
| 夹爪增益 / 误差保护 | Kp/Kd=5/0.05；跟踪误差0.15，反馈时效0.1秒 |

两侧侧握键共同接管；进入 ACTIVE 后，各侧食指扳机先捏到底一次（≥0.95）取得同侧夹爪控制权。此后按下闭合、松开打开；松任一侧握键共同退出。

闭合受阻时，误差≥0.06且反馈低速会立即冻结并限制接触偏移≤0.08，持续0.12秒确认后进入 `HOLDING`，固定目标、不继续收紧。疑似失去接触显示 `HOLD_UNCERTAIN`；扳机≤0.15主动释放全开，≥0.25重新允许闭合；近闭合受阻或0.5秒确认超时走故障流程。接触停止会中断正常加速度斜坡；`HOLDING` 不证明抓取成功，接触参数和夹持力尚未标定。

Quest底部HUD、终端和日志显示夹爪状态及剩余时间，最后10秒提示放下物体；HUD数据过期显示 `STALE`。控制保留目标限幅、有界IK回退和故障卸载；当前没有前臂桌面碰撞约束、自动承重检测或重力前馈，不恢复已撤回的桌面专用控制逻辑。

## 数据采集与环境

完整操作、字段、目录结构和导出命令见[三路采集工作流](g1_lerobot_recording.md)。

- 三路均 **640×480、15 FPS、不旋转**，头部保留单路原始输出，不拆双目。当前配置：头部 `USB2.0_Camera_RGB`；左腕 D405 `261523071841`；右腕 D405 `261523072307`。现场仍须核对实际视角与角色。
- `quest_bridge.py --record --grippers --task "任务描述"` 启用录制，`--output` 指定新的本机会话目录。未加 `--execute` 不发布电机指令，但仍需要外设，不能当作纯离线模拟。
- 原始数据包含相机JPEG、帧时间戳、Quest输入、IK候选、发送前机器人反馈及成功提交的动作。相机通过独立只读SSH子进程侧录；相机异常或接收超时0.3秒触发共同退出。
- `convert_dataset.py --session ... --root ... --outcome success|failure` 离线导出 **LeRobot 0.4.4 / v3.0**，包含三路MP4、Parquet和元数据，仅保存本地。每次会话一个episode，训练片段从双侧夹爪完成接管后开始，仅取连续ACTIVE，不包括卸载；成功/失败由人工标注。
- `observation.state` 和 `action` 均16维：左臂7＋左夹爪1＋右臂7＋右夹爪1。action为执行层限速/接触保持后的实际提交目标；另存全身35个电机槽位的 q/dq/tau_est、时间戳及质量信息。tau_est不是标定夹持力。
- 导出使用动作发送前已到达的状态和图像，默认接收年龄上限0.1秒，拒绝未来、缺失或过期观测。接收时间对齐不等于相机曝光同步。

| 环境 | 用途 |
| --- | --- |
| `.venv-g1` | SDK及有限保持 |
| `.venv-g1-xr` | Quest、IK、控制及相机接收 |
| `.venv-g1-data` | 独立LeRobot转换及数据验证 |

外部固定源码在 `.local/unitree_g1/`；版本与依赖见 [baseline.lock.json](../../artifacts/unitree_g1/baseline.lock.json)和[数据环境依赖锁](../../artifacts/unitree_g1/lerobot-data-requirements.lock)。环境和 `.local` 被Git忽略，clone仓库不会恢复本地环境或已删除的原始数据。

**不要将 `artifacts/unitree_g1/` 整体当作旧日志清理。** 其中包含运行profile；`quest_grippers.py` 还会读取 [负向打开报告](../../artifacts/unitree_g1/gripper_negative_open_live_20261009_001.json)中的闭合参考，以及 [全开报告](../../artifacts/unitree_g1/gripper_absolute_open_live_20261009_001.json)中的目标与现场确认。

## 已验证与待验证

截至2026-10-10的离线结果：控制环境109项测试，108通过、1项LeRobot专用测试跳过；数据环境7项全部通过，含真实MP4/Parquet写入回读、状态动作映射、异常数据拒绝及180秒/2700帧采样边界。合成样例曾生成2个episode、45帧并全部回读，现已清理。见[验证报告](../../artifacts/unitree_g1/recording_offline_validation_20261010.json)；重跑命令见采集工作流。本次文档精简未重跑控制测试或真机试验。

需要保留的现场结论：

| 已有证据 | 结论与限制 |
| --- | --- |
| [自由方向上抬试验](../../artifacts/unitree_g1/vertical_probe_live_20261009_001.json)、[Quest自由方向004](../../artifacts/unitree_g1/quest_bridge_live_20261009_004.json) | 用户确认手臂能够抬起；Quest完成当时60秒ACTIVE。独立上抬试验中5 cm指令对应约3 cm反馈FK，不等于外部实测或10 cm全工作区验收 |
| [夹爪全开试验](../../artifacts/unitree_g1/gripper_absolute_open_live_20261009_001.json)、[Quest夹爪003](../../artifacts/unitree_g1/quest_grippers_live_20261009_003.json) | 用户确认双侧全开无异响、Quest手臂及夹爪控制正常；联合验收使用旧速度0.3/加速度0.2，不能外推到当前1.2/0.6与接触保持逻辑 |
| [Quest夹爪005](../../artifacts/unitree_g1/quest_grippers_live_20261009_005.json) | 旧速度0.6下双侧完成接管，中途退出无锁存故障；不属于双侧完整往返验收 |
| [相机记录](../../artifacts/unitree_g1/camera_discovery_20261009_001.json)、[调焦记录](../../artifacts/unitree_g1/head_camera_blur_20261009_001.json) | 三台相机曾分别出图；物理调焦后头部640×480画面恢复清晰，右腕安装方向已调整，取消软件翻转。未验证三路同时持续取流，高分辨率双目调焦后效果仍未重验 |

下一步待设备恢复后验证：

1. 当前1.2/0.6夹爪运动、接触检测、有限保持、主动释放、头显HUD及实际抓取效果。
2. 三路同时15 FPS、左右视角、帧滞后与曝光/延迟同步，以及180秒完整录制、导出和回读。
3. Quest输入稳定性、轴向/重定位/断流和全工作区动作；历史曾发生0.684秒输入间断，见[Quest006](../../artifacts/unitree_g1/quest_bridge_live_20261009_006.json)，不能通过放宽超时掩盖。
4. 指令与实际抬升差距、卸载阶段跟踪误差和物理卸载。固定方向导致的旧可达性限制已通过改为自由方向缓解；旧方向残差拒绝条件已移除，但历史NaN根因并未据此证实修复。相关分析见[抬升分析](../../artifacts/unitree_g1/motion_analysis_20261009_003.json)和[卸载审查](../../artifacts/unitree_g1/unload_semantics_review_20261008.json)。

## 现场信息与操作边界

已知设备为G1 EDU+、Dex1夹爪和Quest 3S。历史现场版本为App v1.5.4、硬件v1.0、ai_sport 8.7.6.996；详细盘点见 [hardware_inventory.json](../../artifacts/unitree_g1/hardware_inventory.json)。当前状态须现场复核。

历史连接为Quest经Wi-Fi连接本机，本机经 `enp130s0` 有线连接G1；临时配置 `g1-direct-session` 曾使用 `192.168.123.222/24`，相机SSH主机曾为 `192.168.123.164`。不照搬旧Wi-Fi地址、网卡状态或证书。调试时曾手动关闭ai_sport；`GetFsmId=3102/null` 和 LowState `mode_machine=5` 都不能解释为运动FSM=0。

- 机器人运动前先说明具体操作、参数和时长，按当次用户授权执行。历史Hanger、桌面承重、静止与停止人员就绪声明不能跨会话沿用；当前仅做离线工作。
- 重新检查实际网络、服务、反馈与指令流；不自动重试故障、扩大范围、增加增益或重新开启ai_sport。不要直接运行官方运动示例替代本地受限入口。
- `--candidate-trial` 和 `trial_scene` 不代表全量硬件验收；保留 `hardware_approved=false` 的真实语义。手臂入口不负责站姿控制，启用夹爪时只额外控制指定夹爪槽位。
- `unload_submitted=true` 只说明DDS写入，不证明物理卸载完成。断网、掉电、SIGKILL等可能阻止程序卸载；硬件看门狗、独立急停及碰撞保护尚未完整验证。
- 接手先看 `git status --short`，保留未提交工作；不要reset/clean，不提交TLS私钥或本地凭据。新采集使用新的会话目录。

## 代码和文档入口

| 入口 | 用途 |
| --- | --- |
| [采集工作流](g1_lerobot_recording.md) | 数据字段、录制、离线转换、环境及测试命令 |
| [Quest工作流](quest_g1_bridge.md) / [quest_bridge.py](../../scripts/unitree_g1/quest_bridge.py) | 受限遥控、夹爪接入、HUD及录制生命周期 |
| [quest_grippers.py](../../scripts/unitree_g1/quest_grippers.py) | 夹爪参数、接触检测和保持/释放状态机 |
| [执行工作流](g1_arm_execution.md) / [hold_arms.py](../../scripts/unitree_g1/hold_arms.py) | 发布循环、反馈记录、退出联锁及历史试验细节 |
| [record_cameras.py](../../scripts/unitree_g1/record_cameras.py) / [convert_dataset.py](../../scripts/unitree_g1/convert_dataset.py) | 三相机侧录和LeRobot导出 |
| [虚拟双臂](virtual_g1_arms.md) / [输入接收](televuer_input.md) | 虚拟验证和Quest输入 |
| [设备工作流](unitree_g1.md) / [环境基线](unitree_g1_baseline.md) | 网络、只读诊断、版本与重建 |
| [总体计划](../plans/unitree_g1_vr_to_real_plan.md) | 后续采集、回放和训练路线 |

只读诊断工具在 `scripts/unitree_g1/`：`inspect_mode.py`、`read_state.py`、`inspect_gripper.py`、`inspect_command_traffic.py`。其他工作流中的逐次记录按其日期阅读，当前配置以本文及代码为准；历史原始数据已清理。
