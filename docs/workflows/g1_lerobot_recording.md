# G1 / Quest 三路图像与状态动作采集

2026-10-13 控制路径迁移：Quest默认使用`arm_sdk`，新路径允许不带`--grippers`的三路原始记录，但内走线Dex1暂不支持，只有14维双臂SDK请求，不能作为原16维电机动作数据导出。会话记录`control_mode=arm_sdk`、`recording.export_supported=false`，每帧增加实际提交的`arm_sdk_request`（权重、腰臂目标与标称增益）。`convert_dataset.py`在读取会话时明确拒绝此类记录，包括`--include-failed`也不能绕过。原调试模式16维采集/导出格式保持，必须显式使用`--control-mode debug`；详见[新路径流程](quest_g1_bridge.md#官方运控与arm_sdk路径2026-10-13)。


2026-10-12 BiCAT字段与编码更新：关节字段改为 `observation.state_joint_angles` / `action_joint_angles`，仍按左臂7＋左夹爪1＋右臂7＋右夹爪1保存16维。新增 `observation.state_eef_absolute`、`action_eef_absolute`、`action_eef_delta`，每侧均为 `[x,y,z,qw,qx,qy,qz,gripper]`，左右拼接16维。基座为URDF `pelvis`，位置单位米，四元数单位化并采用首个非零分量为正的符号约定。状态使用发送前实测腰部/手臂；动作使用实际提交手臂目标、同帧实测腰部。增量为基座系目标位置减实测位置、`q_target * inverse(q_measured)`；增量字段中的夹爪仍为绝对目标，原生电机单位。

末端沿用Quest IK的虚拟 `L_ee/R_ee`（wrist_yaw局部X方向偏移0.05米，无额外旋转），不是已标定的Dex1夹持中心。离线FK参数固定在 `artifacts/unitree_g1/export_kinematics.json`，导出副本位于 `meta/g1_export_kinematics.json`，源URDF哈希及约定随数据保存。视频为MP4 / AV1（libsvtav1），30 FPS。此更新取代下文历史H.264、旧字段及“未增加末端字段”描述；未启动真机或自动标注任务结果。

2026-10-12 BiCAT对齐更新：当前导出为 **30 Hz**，图像字段为 `observation.images.top/left/right`；top为640×360，left/right为640×480。原始采集继续使用 `head/left_wrist/right_wrist` 和三路640×480，未来相机请求帧率改为30 FPS；实际帧率仍须真机验证。导出时头部整幅图像用Lanczos缩放为640×360（不裁剪，纵横比会改变），双腕不变。历史15 FPS会话仍可读取，按30 Hz动作网格选择发送前最新图像，重复图像继续计入质量报告，不代表原始相机达到30 FPS。关节状态/动作仍为G1的16维，末端位姿定义见上方最新约定。下方历史验证数字保留当时结果。

2026-10-12：用户要求下一轮真机测试启用`--record --grippers --task`；任务文本为用户指定的“左夹抓瓶子到中间”。当前控制参数以交接入口和代码为准：双臂跟踪误差0.20 rad，夹爪速度/加速度2.4/1.2、误差0.30，闭合超差冻结该侧；历史接触保持逻辑已移除。下方2026-10-10离线限制及旧控制参数为历史记录。

2026-10-10：软件链路已完成离线验证。用户当前无法连接机器人、相机、Quest，本阶段只完成离线工作；没有进行真机采集验收。用户确认用于训练，保存状态和动作；三路画面均640×480、15 FPS，不旋转，头部保持单路原始输出、不拆双目。

## 数据约定

每次Quest会话对应一个episode，ACTIVE上限180秒。原始文件保留等待、接管、ACTIVE、卸载全过程；训练导出仅取双侧夹爪都完成接管之后的连续ACTIVE片段（当前版本在首个新鲜ACTIVE扳机输入时自动接管，无需满捏；旧会话仍按日志接管标记裁剪），裁掉之前的等待/接管前缀，不把卸载动作加入训练。故障会话默认拒绝导出，保留原始数据供排查。用户必须标注任务文本及成功/失败，程序不把“正常退出”自动当作成功。

采用仓库依赖的 **LeRobot 0.4.4 / 数据格式v3.0**，通过官方`LeRobotDataset.create/add_frame/save_episode/finalize`写入，不手工拼装一个仅外观相似的数据目录。视频使用AV1（libsvtav1）、30 FPS，状态动作保存为Parquet，任务、episode和统计量使用官方元数据结构。仅保存本地数据，不上传Hugging Face。

| 字段 | 维度/含义 |
| --- | --- |
| `observation.images.top` | 640×360 RGB（从640×480整幅缩放）；`USB2.0_Camera_RGB`，MJPG采集 |
| `observation.images.left` | 640×480 RGB；D405序列号261523071841，YUYV采集 |
| `observation.images.right` | 640×480 RGB；D405序列号261523072307，YUYV采集，不翻转 |
| `observation.state_joint_angles` | 16维实测位置：左臂7、左夹爪1、右臂7、右夹爪1 |
| `action_joint_angles` | 同顺序16维绝对位置目标；执行层限速/接触保持之后，DDS写入成功的实际提交目标 |
| `observation.velocity` / `observation.effort` | 同顺序16维dq / tau_est；tau_est不是标定后的夹持力 |
| `observation.full_body_q/dq/tau_est` | LowState全部35个电机槽位的反馈，包含保留槽位；不代表35个受控关节 |
| `observation.source_timestamps` | 6维：指令发送开始、发送结束、发送前状态接收、头部/左腕/右腕图像主机接收时间 |
| `observation.camera_estimated_read_end` | 3维：通过启动时钟探测换算的相机读取结束时刻，仅作诊断 |
| `observation.camera_sequence` / `observation.robot_tick` | 相机原始序号及机器人原始tick |
| `timestamp/frame_index/episode_index/index/task_index` | LeRobot生成的30Hz数据时间和索引 |
| `next.done` / `next.success` | episode末帧结束标记，以及人工成功标注的末帧标记 |

电机索引顺序明确固定为`15..21,31,22..28,33`。双臂位置/速度单位为rad、rad/s；夹爪保留本机电机位置单位及其每秒变化量，不伪装成米、夹口宽度或牛顿。具体关节名称写入features.names及`meta/g1_recording.json`。

状态来自动作发送前采样的`observation_before_command`，不能以发送后新到达的反馈代替，也不能把下一帧实测状态当作action。原始日志继续保存全电机mode、电压、温度、错误码、Quest输入、IK候选、夹爪接触状态和故障证据。

## 录制与同步实现

- `quest_bridge.py --record --grippers --task "任务描述"`复用现有控制与日志生命周期。当前单条DDS发布路径、速度1.2/加速度0.6、接触保持、保护及180秒上限保持不变。
- `record_cameras.py`在独立子进程中，通过已有密钥认证SSH运行三路相机读取。源代码经SSH命令传入，不安装远程软件，不写入远程目录，不导入机器人SDK。三台相机独立读取，通过带长度和序号的二进制协议传输JPEG，在本机落盘。
- 接管前等待三路取流就绪；运行中相机状态文件、任一路到达时间超过0.6秒或录制报错，会进入现有共同退出流程。相机读写、JPEG编码和后续MP4编码不放在电机伺服线程；控制循环只检查小型状态文件并写既有指令日志。
- 原始相机文件保存OpenCV读取开始/结束、主机接收时刻、序号及跳帧数；8次启动时钟探测保存往返耗时与偏移估计。**读取结束不是曝光时刻，启动时钟探测不保证跨主机时钟长期无漂移。**
- 离线转换在30Hz网格上选取网格点之后的首条已提交动作，每路图像只选在该动作发送前已完整收到的最新帧。机器人状态也必须在发送前已到达，禁止使用未来观测。
- 默认允许的状态/图像接收年龄上限为0.2秒（2026-10-12按用户要求调整），可通过`--max-age-s`收紧但不可放宽。超过一帧周期的样本、重复图像、各路年龄P95和最大接收时差单独记录在质量报告中，不静默隐藏。没有图像、过期、状态非有限、电机错误、动作断流或格式不符直接拒绝导出，不插黑帧、不把缺口两侧拼接。
- 该阈值是软件接收年龄限制，不是已达成的曝光同步精度。真正三路同时15 FPS、帧滞后、左右映射和180秒连续记录须待设备恢复后实测；规划中的半周期P95同步目标尚未验收。

已审查锁定的官方EpisodeWriter及Unitree转换器：前者按队列保存图像/状态/动作，未提供本执行层的发送前反馈和逐流因果时间语义。因此这里复用本仓库的Quest会话及高频指令日志，仅补充只读相机侧录和官方LeRobot写入适配；不引入上游另一套机器人控制发布器。

## 环境与离线验证

控制与相机接收继续使用`.venv-g1-xr`。转换环境为独立的`.venv-g1-data`，安装LeRobot 0.4.4和CPU依赖，不更改XR环境。该环境继承现有Python 3.10.22基础环境；新增包已锁定于`artifacts/unitree_g1/lerobot-data-requirements.lock`。重建环境需要依赖可用，当前离线环境已经装好。

```bash
# 仅在重建环境且已有缓存/允许联网安装时使用，不连接机器人。
uv venv --python .venv-g1-xr/bin/python --system-site-packages .venv-g1-data
uv pip install --python .venv-g1-data/bin/python --torch-backend cpu \
  -r artifacts/unitree_g1/lerobot-data-requirements.lock

# 纯离线数据测试：真实MP4/Parquet写入，再通过官方LeRobot读取所有测试帧。
env HF_HOME=/tmp/g1-dataset-hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv-g1-data/bin/python -m unittest discover -s tests/unitree_g1 \
  -p test_record_dataset.py -v

# 控制/采集模拟测试；不启动相机、DDS或Quest服务。
env -u PYTHONPATH PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv-g1-xr/bin/python -m unittest discover -s tests/unitree_g1 -v
```

已完成：控制环境109项测试，108通过，1项真实LeRobot往返测试因环境隔离跳过；数据环境7项全部通过，包含该真实往返测试。覆盖模拟SSH协议、三路落盘、正常停止、状态/动作时序、相机故障触发退出、未来/过期/缺失数据拒绝、两episode边界、图像RGB顺序、动作映射及180秒/2700帧采样边界。

离线样例曾生成于`.local/unitree_g1/experiments/recording_offline_20261010_001/lerobot_demo/`：**2个episode、45帧、三路640×480视频**，全部帧已由LeRobot重新读取。内容是测试生成的纯色图像和合成关节值，明确`hardware_motion_enabled=false`，不能作为机器人示范训练。2026-10-10按用户要求，已删除该合成会话、导出样例及失败的`three_camera_record_smoke_20261010_001`测试目录；上述样例路径已不存在。历史核验结果与清理记录保留在`artifacts/unitree_g1/recording_offline_validation_20261010.json`，测试代码和依赖环境保留，可运行上方测试重新验证。

## 设备恢复后使用

2026-10-10用户已手动清空`.local/unitree_g1/experiments/`，检查确认目录为空，包括早期真机试验原始数据也已删除。历史报告保留，其引用的旧原始路径已失效；正式采集使用新的会话目录。

当前不执行以下外部采集步骤。恢复后先确认设备、相机角色/时间质量及现场支撑；使用当次实际接口、证书和地址，不能直接复制历史网络参数。

在已核对的Quest启动命令中增加`--record --grippers --task "具体任务文本"`即可录制；`--output`必须为新的会话目录。`--camera-host`和`--camera-ssh-key`可覆盖相机SSH配置。未加`--execute`仍不发布电机指令，但仍需设备反馈/相机/Quest，并非纯离线模拟。运动执行仍按既有当次现场确认流程进行。

原始目录示例：

```text
session/
  session.json                  # 任务、版本哈希、限值和录制设置
  commands.jsonl                # 高频反馈与实际提交动作，含发送前快照
  proposals.jsonl               # IK候选
  input/events.jsonl            # Quest原始输入
  cameras/
    metadata.json               # 三路设备与时钟探测
    frames.jsonl                # 逐路时间、序号及图像路径
    head/*.jpg
    left_wrist/*.jpg
    right_wrist/*.jpg
    status.json
    summary.json
  summary.json                  # 最终状态及故障；不自动推断任务成功
```

结束后，根据现场结果明确标注成功或失败，再离线导出。支持重复`--session`合并多次采集为多个episode；本次调用的`--outcome`应用于全部输入会话，因此只合并同一结果标签的会话。输出必须为新目录，失败保留`.partial-*`目录和错误证据，不覆盖现有数据集。

```bash
env HF_HOME=/tmp/g1-dataset-hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  .venv-g1-data/bin/python scripts/unitree_g1/convert_dataset.py \
  --session /path/to/recorded/session \
  --root /path/to/new/lerobot_dataset \
  --repo-id local/g1_pick_object \
  --outcome success
```

正常失败演示标注`--outcome failure`。故障会话另需显式`--include-failed`，仍必须满足数据完整性检查，且不能标注success。仅为测试导出影子数据时使用`--allow-shadow`，元数据保留影子属性。

最终目录包含官方`data/`、`videos/`、`meta/`；`meta/g1_recording.json`补充时间质量、单位、顺序、原始清单哈希和每段结果，`meta/g1_sources/`保留会话与相机元数据副本。完整原始图像/日志仍保存在会话目录，需要随数据备份，不会自动删除。

2026-10-11 兼容说明：日志字段 `gripper_command.picked_up` 保留；新版本表示同侧夹爪已由新鲜ACTIVE扳机输入自动启用，旧版本表示已完成首次满捏。导出仍拒绝未接管前缀和故障数据，不改变16维状态/动作定义。
