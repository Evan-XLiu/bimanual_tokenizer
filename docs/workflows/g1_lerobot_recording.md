# G1 / Quest 三路图像与状态动作采集

## 2026-10-15 更换头部相机验证

新头部为 RealSense D435i，USB序列号 `253843064640`，SDK报告序列号 `349622073101`（两种标识不同）。已恢复本机 enp3s0 临时直连 `192.168.123.223/24`，原有线配置保留，Wi-Fi不变。三路同时20秒RGB短测正常结束：头部587帧/29.23 FPS、左腕603帧/30.01 FPS、右腕603帧/30.02 FPS，全部640×480且JPEG解码通过；头部最大接收间隔0.0835秒。未启动机器人运动或修改机器人服务，未验证深度/IMU及180秒连续采集。

默认 `camera_capture_profile.json` 已切换头部至 V4L2 YUYV 640×480/请求30 FPS、不旋转。RGB为 `/dev/v4l/by-path/platform-3610000.xhci-usb-0:2.3:1.3-video-index0`（当前video4），须保持USB插口；该机by-id链接存在接口重名，index0实际指向深度节点，index2为红外，不可作为RGB。初轮红外测试单独保留，最终结论来自RGB复测。双腕角色及配置保持10月14日确认值。见[验证报告](../../artifacts/unitree_g1/head_camera_validation_20261015.json)。

简洁英文使用说明：[G1 Episode Collection Guide](g1_collection_quickstart.md)。

## 正式连续采集：s保存、f重采

入口：`scripts/unitree_g1/collect_episodes.py`。这是带人工质量确认的连续遥操作采集，
不是机器人自主执行任务。一个批次只启动一次VR网站和WebSocket；每条使用独立的IK、控制和相机记录。
机器人及夹爪保留当前 `arm_sdk + external-dex1` 路径。

启动时依次检查TLS证书/私钥、网卡与本机VR地址/端口、相机配置、磁盘空间、LeRobot/AV1环境、
官方运控模式、机器人及夹爪反馈、DDS网络、相机SSH连接和三路图像实际解码。
执行模式还等待头显和控制器新鲜输入，然后进入`awaiting_n`等待终端确认。
首条、保存后的下一条和失败重采都必须输入`n`并回车；等待期间不启动本条记录、控制或预全开。
确认后重新检查机器人/夹爪和磁盘，三路相机就绪后
才预全开夹爪、加载IK并开放本条输入。预检查不自动切换运控、起立或启动机器人服务。

以目标10条为例，在仓库根目录使用当前有效的地址、证书及网卡运行。以下为2026-10-14配置示例；
`--scene-confirmation True`表示本次现场已确认，`False`或省略时不能执行采集；输出目录必须不存在。
在设备、网络地址和证书不变时，可复用完整命令，只按批次调整`--episodes`、`--task`、`--output`。
其他必填参数仍需保留，不能从命令中省略；每批`--output`必须为新的目录。

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv-g1-xr/bin/python scripts/unitree_g1/collect_episodes.py \
  --episodes 10 --task "填写本批次任务" \
  --output .local/unitree_g1/collections/formal_001 \
  --interface enp3s0 --host-address 172.20.10.13 \
  --cert .local/unitree_g1/tls/rotation-20261014-001/cert.pem \
  --key .local/unitree_g1/tls/rotation-20261014-001/key.pem \
  --camera-host unitree@192.168.123.164 \
  --camera-ssh-key .local/unitree_g1/ssh/id_ed25519 \
  --execute --scene-confirmation True
```

只检查设备时，将最后一行改成 `--preflight-only`，并使用另一个新的输出目录。
该模式不发运动指令、不预全开，也不启动VR服务，所以不替代执行模式的头显实际连通检查。
默认每条ACTIVE上限180秒，`--active-seconds`可缩短；`--wait-seconds`默认300秒；
`--min-free-gb`默认10 GiB，每条检查，不承诺足以容纳任意时长的全部批次。

操作流程：

终端提示保留 `Try X | Episode Y/N`：Try是累计尝试次数，Episode是当前要保存的有效条编号，
显示从1开始；`f`后Try增加而Episode不变，`s`成功保存后两者进入下一条。磁盘目录编号仍从0开始。

1. 初次进入脚本打印的VR网站，保持网页/VR会话打开。
2. 放稳物体、复位场景后，在终端输入 `n` 并回车，开始本条检查、夹爪预全开和采集准备。
3. 终端提示“Ready. Release both side grips, then hold both.”后，松开双侧握键，再重新按住接管；新条不沿用上一条的按住状态。
4. 操作完成后松开任一侧握键结束，等待控制归还及相机记录关闭。
5. 终端输入 `s` 并回车：人工确认本条有效/成功，保存原始数据并计入accepted，加入批次结束后的导出队列。
   输入 `f` 并回车：删除本条暂存的图像、指令、输入及预全开记录，不导出、不计数。
   两种情况均返回等待 `n`，不会直接开始下一条或自动预全开夹爪。
6. 达到 `--episodes` 的有效条数后结束采集；等待`n`或判定时均可输入`q`，Ctrl-C也可提前结束。
   子进程和VR清理结束后，统一编码已接受条目并显示逐条进度；未判定条保留为pending且不导出。
   启动时的只读预检不受`n`门槛影响；`n`控制的是每条尝试的启动。

单条硬件或相机故障时，子进程先尝试现有卸载流程；采集器保留原始数据和 `collection_fault.json`，进入 `r` 重检 / `q` 退出的等待状态。修复后输入 `r`，检查卸载证据、相机进程已结束、输入路由关闭、机器人/夹爪反馈、三路图像和VR输入；通过后仍需另行输入 `n` 开始新条。故障条目不计入成功数，不提供 `s/f`，不自动重试控制。
控制归还/夹爪清零未确认、相机进程结束未确认或缺少摘要时保持阻塞，不能用传感器预检通过替代卸载证据。初始启动失败仍停止批次；编码改为采集结束后统一执行，单个导出失败保留原始数据并继续处理队列中的其他条目。`f`仅用于控制正常结束但演示内容无效的条目。
每次进入下一条仍需新鲜松开/按住手势。仅软件提交清理可验证，不把它写成物理释放已确认。

目录结构：

```text
formal_001/
  collection.json                  # 目标、有效数、每次尝试的s/f结果和状态
  checks/                          # 启动及每条只读检查；探测图像通过后删除
  vr/                              # 常驻输入服务元数据/状态；不保存全批次原始输入
  pending/attempt_000001/           # 当前尚未判定的原始记录，f后删除
  episodes/episode_000000/
    session.json, commands.jsonl, cameras/, input/, ...
    operator_label.json            # s的人工标注
    lerobot/                       # 本条独立的BiCAT/G1 LeRobot v3数据集
  logs/                            # 每次尝试的过程日志；f后仅保留小型诊断和审计记录
```

`collection.json`的`accepted`列出已保存数据集路径；每条独立数据集内部`episode_index=0`，
批次序号由外层`episode_000000...`及清单表示。保留每条原始记录，必要时可用转换器多个`--session`
合并为统一多episode数据集。不自动覆盖或续写已有批次。

当前已完成模拟s/f循环、持续输入跨条路由、故障不重启、删除边界及真实LeRobot写入读取的离线验证；
尚未进行这条连续采集流程的真机验收。

## 2026-10-14 当前默认格式：BiCAT/G1，LeRobot v3.0

本节取代下文关于 `arm_sdk` 只保留原始数据、拒绝导出及必须事后手动转换的历史说明。
`quest_bridge.py --record --grippers --task "任务描述"` 会在机器人控制、相机和子进程清理结束后，
使用独立 `.venv-g1-data` 环境自动生成 `<会话目录>/lerobot/`。
当前 `arm_sdk` 路径还需 `--gripper-backend external-dex1`；不带夹爪的采集因缺少完整状态/动作而拒绝。
所有原有设备连接、执行和现场确认参数仍需按当前流程提供。

- 标准目录为 `data/`（Parquet）、`videos/`（MP4/AV1）、`meta/`，30 Hz。
- 图像字段为 `observation.images.top/left/right`。2026-10-14 新采集顶部完整保留1280×480、双腕640×480；原始JPEG和导出视频均不缩放、不裁剪。历史640×480头部会话保留原640×360导出兼容路径。
  原始头部约15 FPS，30 Hz导出会复用图像；不能视为30 FPS独立曝光。图像复用数写入质量报告。
- 状态/动作字段为 `observation.state_joint_angles`、`action_joint_angles`；
  末端字段为 `observation.state_eef_absolute`、`action_eef_absolute`、`action_eef_delta`。
  均为float32。G1关节字段保留16维（每侧7关节＋夹爪），YAM为14维，不能直接混用关节策略。
  末端字段与BiCAT同为16维，但基座、虚拟TCP和夹爪单位为G1定义，不声称YAM动作语义等价。
- 外接夹爪使用独立服务的真实 `q/dq/tau_est`，绝不读取LowState空槽31/33充当反馈。
  服务坐标约5.4为打开、0为闭合，未标定为夹口宽度。只选择发送前收到且不超过0.2秒的反馈；
  `observation.gripper_source_timestamps` 保留两侧来源时间。新采集额外将夹爪快照写入发送前观测。
- SDK关节动作核对实际提交请求；末端动作FK使用提交的腰部和手臂目标，末端状态FK使用实测腰臂。
  SDK请求不等于下层电机已实现的位置。原始35槽全身反馈保持原样。
- 默认任务结果为 `unknown`，`next.success_known=false`；`next.success=false`不表示已经确认失败。
  已有人工作业结果时，离线转换可用 `--outcome success/failure`。不要在任务完成前预填成功标签。
  `next.done`为末帧，已确认成功的末帧 `next.success=true`。
- 默认拒绝不完整或运行期故障。仅当首个故障明确为最后ACTIVE动作之后RELEASING阶段的
  tracking_error、ACTIVE无故障、相机及夹爪清理正常时，允许导出此前连续ACTIVE段；
  `failed_session=true` 和 `release_only_fault=true` 同时保留，绝不抹去退出故障。
- 导出结果见 `export_status.json` 和 `export.log`。失败保留原始数据和中间目录，程序返回非零，
  不上传、不覆盖已有数据集。仅在格式写入和官方读取验证完成后才公开最终 `lerobot/` 目录。

参考格式：[BiCAT bottles/meta/info.json](https://huggingface.co/datasets/Evanxw0715/BiCAT-BimanualYAM-Dataset/blob/main/bottles/meta/info.json)。
当前转换是BiCAT字段布局的G1适配，不删掉G1关节来伪装成YAM的14维。

## 2026-10-14 当前相机采样配置

新电脑使用 [camera_capture_profile.json](../../artifacts/unitree_g1/camera_capture_profile.json)。2026-10-14 更换双腕相机后，用户查看预览确认：B（`255323071511`）为左腕，A（`260323072820`）为右腕；两侧均通过固定序列号 V4L2 路径采集，下文历史相机标识不适用于当前设备。

| 角色 | 当前来源 | 请求帧率 / 保存尺寸 |
| --- | --- | --- |
| head | 已有 `/camera/head/image/compressed` ROS 流 | 30 FPS / 1280×480 |
| right_wrist | D405 `260323072820`，by-id video-index4，V4L2 YUYV | 30 FPS / 640×480 |
| left_wrist | D405 `255323071511`，by-id video-index4，V4L2 YUYV | 30 FPS / 640×480 |

左腕设备由现有发布器占用，因此只读订阅其图像，不另行打开设备、不重启服务。ROS 使用本次在机器人上核实的 Foxy/CycloneDDS 环境；本机不需安装 ROS。右腕在当前 USB2 连接下不提供 sysfs 序列号，依靠已核对的 USB 插口定位，换插口后必须重新核对。ROS 消息不带硬件序列号，profile 的 expected_serial 是本次人工及设备盘点证据，不是自动身份认证。

三路不旋转、不裁剪。头部驱动虽然接受15 FPS请求，仍报告30 FPS；以实际到帧统计为准，不能视为已达成30 FPS。左腕保留上游 JPEG，不重新编码；其时间是机器人端订阅回调接收时刻，其他两路是 V4L2 读取完成时刻，均非曝光时间。帧头记录 timing 和左腕原始 source_timestamp_ns；重复或倒退的左腕时间戳不会刷新有效性。旧导出仍按本机完整接收时刻进行因果对齐，不声称曝光同步或已消除上游缓存延迟。

仅相机测试（不启动机器人控制），从仓库根目录运行并使用新的输出目录：

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 .venv-g1-xr/bin/python \
  scripts/unitree_g1/record_cameras.py \
  --host unitree@192.168.123.164 \
  --ssh-key "$PWD/.local/unitree_g1/ssh/id_ed25519" \
  --config artifacts/unitree_g1/camera_capture_profile.json \
  --duration 20 --output .local/unitree_g1/experiments/camera_capture_NEW
```

`--duration` 包含 SSH/相机初始化时间。`quest_bridge.py --record` 默认使用此配置，也可通过 `--camera-config` 指定配置；本机新 SSH 密钥仍需显式传入 `--camera-ssh-key`，不要使用旧默认密钥路径。本次没有启动 Quest 或运动采集。

**角色纠正：001/002及此前连通性测试中的左右腕标签颠倒；原始文件保留，不按旧标签用于训练。当前配置已对调采集角色，不修改机器人现有ROS话题名称。**

20秒短测002正常结束（以下左右名称为当时错误标签），三路共同接收窗口约17.52秒：head 272帧/15.46 FPS、左腕511帧/29.10 FPS、右腕529帧/30.10 FPS。1312张JPEG全部解码且尺寸正确，最大接收间隔分别0.091/0.075/0.062秒。首轮001因 ROS Foxy执行器上下文错误失败，修复后002通过，两轮记录均保留。本次不等于180秒验收或正式机器人示范数据；arm_sdk导出限制保持不变。详见[采样验证报告](../../artifacts/unitree_g1/camera_capture_validation_20261014.json)。

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

2026-10-14 批次导出更新：连续采集按 `s` 仅保存原始数据并计入 accepted，不立即编码。达到目标数、`q` 或 Ctrl-C 结束采集并完成子进程/VR清理后，统一逐条导出已接受条目。`collection.json` schema v2 的 `accepted` 为原始episode目录，`exported`为成功回读的数据集目录，分别维护计数；故障未标注条目不进入编码队列。强制终止进程或断电不能保证执行退出导出。
