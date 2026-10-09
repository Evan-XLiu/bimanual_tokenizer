# G1 电脑网线直连接入

2026-10-08 最新进展：有线只读通信、调试模式核验与两次人工托持的原姿态带力保持已执行；一次重复启动因未静止被只读预检拦截。003 卸载阶段触发 tracking_error，用户反馈「一切正常，当时松手了一下」。接手先读 [交接入口](unitree_g1_handoff.md)，详细结果见[执行接口与模式核查](g1_arm_execution.md)。

电脑直连 G1 的 SDK 通信已用于有限原姿态保持；Quest 输入和虚拟双臂已独立验证，但 Quest 尚未驱动真机。后续按[更新计划](../plans/unitree_g1_vr_to_real_plan.md)继续动态目标/控制保护和现场验证。Quest 采集与诊断统一使用[官方 TeleVuer 输入工具](televuer_input.md)。只读通过不等于运动控制通过。

官方 XR 独立环境与源码锁定已完成，版本及离线检查见[软件基线](unitree_g1_baseline.md)。这不改变下述真机控制与夹爪待核验状态。

## 已确认与待核验

- G1 EDU+；用户提供 App 软件版本 v1.5.4、硬件版本 v1.0（2026-10-08）。每侧 7 个机械臂关节和 1 个独立夹爪控制量，学习动作共 16 维。
- 双脚着地；用户已确认 Hanger 可靠承重、双臂有人安全托持，现场通过物理开关停止。软件在调试模式下对腰腿发零增益/零前馈，不负责站姿保持；独立停止性能尚未验证。
- 已核实 App 关闭 ai_sport/8.7.6.996 的调试入口。夹爪型号/单位/反馈、完整动态控制能力、硬件急停和重新启用 ai_sport 的恢复行为仍待确认。

## 接线与网络

控制电脑的以太网口连接 G1 对外开发网口。USB 转以太网适配器也可。先确认接口用途，不修改机器人内部网络。

```text
控制电脑有线网卡 ── 网线 ── G1 开发网口
控制电脑 Wi-Fi ── 与 Quest 可互访的局域网（地址与隔离策略需实测）
```

检查 `ip -br addr`、`ip -4 route` 和 `nmcli device`。官方默认网络示例是 `192.168.123.0/24`；本机是否仍用此网段需要实测。主机地址不能与机器人重复。直连网卡不设默认网关和 DNS，保留 Wi-Fi 默认路由。

2026-10-06 本开发主机发现有线网卡 `enp130s0`。为这台主机创建的临时配置命令如下；其他电脑应替换网卡名，并先确认地址没有冲突：

```bash
nmcli connection add save no type ethernet ifname enp130s0 con-name g1-direct-session connection.autoconnect no ipv4.method manual ipv4.addresses 192.168.123.222/24 ipv4.never-default yes ipv4.dad-timeout 3000 ipv6.method disabled
nmcli --wait 15 connection up g1-direct-session
```

`save no` 配置仅保存在内存中，NetworkManager 重启后不保留；原有有线配置不修改。不要重复创建同名配置。退出直连时可执行 `nmcli connection down g1-direct-session`，需要恢复原连接时执行 `nmcli connection up '有线连接 1'`（名称以本机为准）。

SDK 订阅按网卡和 DDS domain 初始化，不需要在脚本中猜一个机器人 IP。能 ping 通只能证明部分网络可达，验收依据是收到真实的 G1 状态。

## 独立控制环境

不把机器人控制依赖安装到模型训练环境。当前 SDK 源码固定为官方仓库提交 `814556d15970dd2ecf1c9984e845ca02ab07e206`，只读消息接口已对照该版本审查，固件兼容性需现场确认。

从本仓库根目录运行，使用 Python 3.10：

```bash
python3.10 -m venv .venv-g1
env -u PYTHONPATH .venv-g1/bin/python -m pip install 'unitree_sdk2py @ git+https://github.com/unitreerobotics/unitree_sdk2_python.git@814556d15970dd2ecf1c9984e845ca02ab07e206'
```

环境目录已被 Git 忽略。`env -u PYTHONPATH` 避免当前 shell 的 ROS 路径混入独立环境；本机清除该继承变量后 `pip check` 通过。SDK 安装若报缺少 CycloneDDS 本地库，按[官方 SDK 安装说明](https://github.com/unitreerobotics/unitree_sdk2_python)处理，不更改训练环境。

## 只读诊断

```bash
env -u PYTHONPATH .venv-g1/bin/python scripts/unitree_g1/read_state.py --interface enp130s0 --duration 30 --output /tmp/g1-read-state-first.jsonl
```

输出文件必须尚不存在。脚本仅订阅 `rt/lowstate` 的 `unitree_hg` 消息，不创建命令发布者、不调用模式切换服务，不需要先退出腰腿保持控制。

每秒输出一份诊断快照，结束输出摘要。主机接收时间为单调时钟；`device_tick_raw` 保留设备原值，不推定其单位。`reference_left_q` 与 `reference_right_q` 按官方 arm7 示例的索引 15–21、22–28 展示，尚未核验本机映射；同时保存完整电机槽位反馈。数组长度不是机器人实际自由度数。

本工具不读取夹爪反馈，也不构造虚假的 16 维实测状态。收到不断推进的 tick 后返回 0，仅表示基本状态接收通过；没有数据、tick 冻结、倒退或非有限值返回非零。诊断超时默认 2 秒，不是运动控制看门狗参数。输出为 1 Hz 诊断快照，不是高频原始采集数据。

检查：

1. `result` 是否为 `received_advancing_state`，`tick_advances` 是否持续增加。
2. `host_receive_hz`、`max_host_receive_gap_s` 与重复 tick 情况。这些是应用接收统计，不是伺服频率或端到端延迟。
3. 两臂反馈是否有限且符合当前姿态；错误状态字段交由本机对应文档解释。
4. 无数据时先核对网口、网卡、IP 网段、DDS 通信和固件消息类型，不运行运动示例来“测试连接”。

离线工具测试不需要 SDK、GPU 或机器人：

```bash
python3 -m unittest discover -s tests/unitree_g1 -v
```

## 当前控制入口与下一步

当前入口为 scripts/unitree_g1/hold_arms.py，默认影子模式，实际带力需显式参数；仅支持当前姿态渐进接管、短时保持及卸载。下一步先审查卸载阶段 tracking_error 的语义与日志，再准备受限动态目标及小幅单关节/单臂验证。仍未接入 Quest 真机运动。

官方 `g1_arm7_sdk_dds_example.py` 包含腰部控制和主动归零/抬臂，不应直接用于此项目的首次连接测试。当前仓库已有受限原姿态发布入口及实测记录，但不能直接用官方运动示例替代它。

## 2026-10-06 首次只读实测

用户确认 G1 已通过网线连接并上电后，运行 30 秒只读诊断，正常退出（退出码 0）：

| 指标 | 结果 |
| --- | --- |
| 状态消息数 | 31,145 |
| tick 推进次数 | 29,890 |
| 重复 tick 消息数 | 1,254 |
| 主机消息接收率 | 1,038.46 Hz，含重复 tick，不是伺服频率 |
| 最大主机接收间隔 | 4.92 ms |
| `mode_machine_raw` | 5；仅记录原值，不据此判断站立/阻尼模式 |
| 诊断快照 | 30 份，快照中 `motorstate` 均为 0；不代表整机已完成安全检查 |

日志位于本机 `/tmp/g1-read-state-20261006-first.jsonl`，是临时文件，正式实验应复制到外部实验目录归档。重复 tick 的来源仍待分析，本次未统计唯一设备时间对应的完整状态率，也未测端到端延迟。全程未发送运动命令或切换模式。下一步仍需确认现场模式、支撑与停止条件。

## 2026-10-06 右夹爪控制前的接口检查

用户说明吊具尚未就位，要求仅右夹爪轻微运动。先运行只读发现：

```bash
env -u PYTHONPATH .venv-g1/bin/python scripts/unitree_g1/inspect_gripper.py --interface enp130s0
```

本次约 6 秒观察窗口内：

- 未发现 `rt/dex1/right/cmd` 的订阅端或 `rt/dex1/right/state` 的发布端，也未收到左右独立 Dex1 状态。
- 发现 Dex3 状态发布端和指令订阅端，但未收到对应状态；话题被创建不代表装有该型号末端。
- 收到 6,244 条全身状态；槽位 31、33 的末次 q 分别约为 -0.000368、-0.000437，dq 也有反馈。这与官方内置 Dex1 的索引一致，只能作为接口候选，不能据此确认硬件型号或开合单位。
- 发现 `rt/lowcmd`、`rt/arm_sdk` 等通道已有发布端，不能追加竞争的控制发布者。

检查结果保存在临时文件 `/tmp/g1-gripper-inspection-20261006.json`。全程未发布运动指令。

对照官方 `xr_teleoperate` 提交 `817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b`：独立 Dex1 使用 `rt/dex1/right/cmd`；内置 Dex1 使用全身 `rt/lowcmd`，并在其控制器中配置身体各关节；遥操作入口对该模式进入调试模式，不支持其 `--motion` 路径。因此当前不能直接运行该示例来实现“不影响腰腿的右夹爪测试”，也不能假定全身消息中其他关节填零等于保持原控制。

继续条件：由宇树确认本机夹爪型号、固件对应的独立控制方法与控制权规则；若只能使用全身低层接口，则先完成支撑、腰腿保持和接管/停止流程核验。本次右夹爪运动尚未执行。

参考：[官方网络说明](https://github.com/unitreerobotics/xr_teleoperate/wiki/Network)、[固定版本 G1 arm7 示例](https://github.com/unitreerobotics/unitree_sdk2_python/blob/814556d15970dd2ecf1c9984e845ca02ab07e206/example/g1/high_level/g1_arm7_sdk_dds_example.py)、[内置夹爪控制器](https://github.com/unitreerobotics/xr_teleoperate/blob/817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b/teleop/robot_control/robot_arm.py)、[遥操作模式选择](https://github.com/unitreerobotics/xr_teleoperate/blob/817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b/teleop/teleop_hand_and_arm.py)。
