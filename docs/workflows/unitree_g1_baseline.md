# G1 官方 XR 软件基线

2026-10-07 已完成本机离线软件基线固定。对应[计划](../plans/unitree_g1_vr_to_real_plan.md)阶段零的软件部分；硬件 profile、Quest 现场测试、视频实测、仿真任务和真机运动尚未验收。本次未运行控制器构造器、未初始化 DDS、未发送机器人命令。

## 固定版本与目录

外部源码位于仓库下 `.local/unitree_g1/`，环境为 `.venv-g1-xr/`，均已忽略 Git；应提交的是[版本锁](../../artifacts/unitree_g1/baseline.lock.json)、依赖锁、补丁与验证报告。原 `.venv-g1` 只读环境和训练环境未修改。

| 组件 | 固定提交 | 状态 |
| --- | --- | --- |
| xr_teleoperate | `817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b` | 官方候选提交转为本项目离线基线；控制代码未修改 |
| televuer | `766de45e74373ae0ea66321d942ce538385655a5` | 主仓库锁定子模块，已安装 |
| teleimager | `57cf2a40572227273fa001cd17833b755331ec97` | 主仓库锁定子模块，可编辑安装 |
| dex-retargeting | `d7753d38c9ff11f80bafea6cd168351fd3db9b0e` | 主仓库锁定子模块，附依赖声明补丁 |
| unitree_sdk2_python | `814556d15970dd2ecf1c9984e845ca02ab07e206` | 与既有只读基线一致，另装于 XR 环境 |
| unitree_lerobot | `41c2805742de879ddab2d8d6beaeaf215f876395` | 仅锁定源码，转换运行环境待阶段三建立 |
| 其 LeRobot 子模块 | `a5b29d430105f5235eb05bbf2db5a0d747a869d6` | 已检出，未装入 XR 环境 |

核心运行版本：Python 3.10、NumPy 1.26.4、Pinocchio 3.1.0（Conda，含 CasADi 接口）、CasADi 3.6.7、Vuer 0.0.60、CycloneDDS 0.10.2、OpenCV 4.11.0.86、Matplotlib 3.7.5、Rerun 0.20.1、CPU Torch 2.3.0、aiortc 1.14.0。完整版本与 Conda 构建见锁文件；不是只靠此摘要重建环境。

## 安装中的兼容处理

- 固定 `opencv-python==4.11.0.86`，保持 NumPy 1.26.4；固定 `params-proto==2.13.2`，供 Vuer 0.0.60 使用。
- 官方夹爪模块在导入时也加载 `dex_retargeting`，因此安装了它和 CPU Torch，即使首轮使用手柄而非手追踪。
- [依赖补丁](../../artifacts/unitree_g1/dex-retargeting-conda-pinocchio.patch)只移除 `dex-retargeting` 的 PyPI `pin` 依赖声明，以免重复安装另一套 Pinocchio；实际依赖由 Conda 精确版本提供，并由校验脚本检查。没有修改重定向算法或控制代码，补丁是基线的一部分。
- `teleimager` 普通 wheel 未带上 `unitree_logo.svg`，视频服务导入失败；按上游安装方式改用 `pip install -e` 后保留源码资源。视频服务应在独立进程运行，上游日志配置不支持随意在已初始化日志的进程中重新配置。
- 已装视频服务公共依赖，但按具体相机选择的 `pupil-labs-uvc` / `pyrealsense2` 驱动尚未安装或验证；相机型号与 PC2 环境确认后处理。
- `unitree_lerobot` 要求 Matplotlib >=3.9，而遥操作基线要求 3.7.5。转换工具先固定源码，后续用单独环境验证；本仓库训练侧使用 `lerobot==0.4.4`，不能因上游转换器默认版本而直接升级训练依赖。

## 离线校验

从本仓库根目录运行：

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 .venv-g1-xr/bin/python scripts/unitree_g1/verify_xr_baseline.py --ik
```

脚本检查 7 个仓库提交、已知补丁、G1 资产哈希、Python 包和 Conda 构建、`pip check`、输入/视频/SDK/夹爪/IK/录制模块导入，以及一次零位 FK 目标的 IK 求解。导入与 IK 子进程禁止 socket 绑定和连接，不启动 XR 服务、相机或 DDS。上游新增末端 frame 后，测试重新建立 Pinocchio Data 再计算 FK；不使用旧 frame 缓存。

[本次验证报告](../../artifacts/unitree_g1/verification.json)只证明软件可以加载并完成离线求解，不证明真实关节映射、夹爪参数、碰撞保护或操控质量。清除 `PYTHONPATH` 用于避免本机 ROS 路径污染环境；不需修改用户 shell 配置。

## 重建方法

以下用于新的工作目录或尚不存在的目标；已有目录先运行校验，不覆盖未知修改。命令从本仓库根目录执行。

```bash
git clone https://github.com/unitreerobotics/xr_teleoperate.git .local/unitree_g1/xr_teleoperate
git -C .local/unitree_g1/xr_teleoperate checkout --detach 817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b
git -C .local/unitree_g1/xr_teleoperate submodule update --init --recursive
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git .local/unitree_g1/unitree_sdk2_python
git -C .local/unitree_g1/unitree_sdk2_python checkout --detach 814556d15970dd2ecf1c9984e845ca02ab07e206
git clone https://github.com/unitreerobotics/unitree_lerobot.git .local/unitree_g1/unitree_lerobot
git -C .local/unitree_g1/unitree_lerobot checkout --detach 41c2805742de879ddab2d8d6beaeaf215f876395
git -C .local/unitree_g1/unitree_lerobot submodule update --init --recursive
```

在未应用补丁的 `dex-retargeting` checkout 中，使用补丁的绝对路径：

```bash
git -C .local/unitree_g1/xr_teleoperate/teleop/robot_control/dex-retargeting apply "$PWD/artifacts/unitree_g1/dex-retargeting-conda-pinocchio.patch"
conda create -y -p "$PWD/.venv-g1-xr" --file artifacts/unitree_g1/conda-linux-64.lock
env -u PYTHONPATH PYTHONNOUSERSITE=1 .venv-g1-xr/bin/python -m pip install --no-deps -r artifacts/unitree_g1/pip-requirements.lock
env -u PYTHONPATH PYTHONNOUSERSITE=1 .venv-g1-xr/bin/python -m pip install --no-deps --no-build-isolation ./.local/unitree_g1/unitree_sdk2_python ./.local/unitree_g1/xr_teleoperate/teleop/televuer ./.local/unitree_g1/xr_teleoperate/teleop/robot_control/dex-retargeting
env -u PYTHONPATH PYTHONNOUSERSITE=1 .venv-g1-xr/bin/python -m pip install --no-deps --no-build-isolation -e ./.local/unitree_g1/xr_teleoperate/teleop/teleimager
env -u PYTHONPATH PYTHONNOUSERSITE=1 .venv-g1-xr/bin/python scripts/unitree_g1/verify_xr_baseline.py --ik
```

Conda 锁包含 linux-64 的精确 URL/MD5，pip 锁固定版本但不含 wheel 哈希；重建仍依赖包源可用，完成后须通过校验。此锁针对当前 Linux x86-64 主机，不保证可直接用于 Jetson ARM。没有生成或提交 TLS 私钥、机器网络地址或真实数据。

## 上游能力与待适配差距

下列路径均相对锁定的 `xr_teleoperate`，结论来自该提交源码审查。

| 环节 | 本基线证据 | 下一阶段要求 |
| --- | --- | --- |
| 输入 | `televuer/example/test_televuer.py` 和 `test_tv_wrapper.py` 没有 SDK 控制器，但默认连接固定地址的图像服务 | 新建仅实例化 TeleVuer 的薄诊断入口，配置透视、关闭视频输入；再做 Quest 现场测试 |
| 主入口 | `teleop/teleop_hand_and_arm.py` 初始化 DDS、模式管理及硬件控制器 | 不能用主入口或 `--headless` 做无机器人输入测试 |
| 输入新鲜度 | `televuer.py` 的 `motion_data_ready` 在输入到达时置真 | 该标志不等于带时限的追踪有效性；需补逐帧时间、断流、重定位和离合测试 |
| 参考系 | 主入口设置 `arm_reference_mode="head_yaw"` | 标定并确认转头对手臂参考的影响，不能混用原始 WebXR 坐标 |
| 行走 | 主入口在 controller + motion 时发送摇杆 `Move`，双摇杆按钮可调用 `Damp` | 固定腰腿任务需去除这些输入路径，并核验控制权；不直接套用启动参数 |
| 夹爪 | `dex1_internal` 与 `--motion` 被显式判为不兼容 | 核验现场型号、独立控制接口与反馈；未知时禁止运动 |
| 控制器生命周期 | `robot_arm.py` 构造器创建发布器/控制线程；入口 finally 调用回零 | 接管、启动和退出都需统一适配，不能把“未按 r”视为没有任何控制输出 |
| IK | G1 IK 有位置限位约束；本次只验模型构建与一组求解 | 补双臂/躯干碰撞、工作空间、异常目标与解连续性测试；资产中的 Dex3 手模型也需与实际夹爪核验 |
| 动作语义 | 入口录制 `sol_q`；`robot_arm.py` 的发布循环另做速度裁剪 | 记录实际提交指令及发送时间，不能直接将原录制 action 认定为最终执行动作 |
| 时间与数据 | episode 录制器保存状态/动作/图像，使用录制流程时间 | 补相机和状态时间、质量事件与对齐报告，再适配 LeRobot 16 维 schema |
| 转换器 | 源码及其子模块已锁定 | 阶段三单独安装并验证 loader/batch，当前不声称转换运行通过 |

[硬件清单](../../artifacts/unitree_g1/hardware_inventory.json)是不可执行的盘点文件，`motion_enabled=false`，未知项为 null。当前 16 维表示来自用户确认；夹爪型号、单位、限位、固件、支撑和控制归属仍需现场核验。

阶段一已新增[官方 TeleVuer 独立输入诊断入口](televuer_input.md)，完成软件测试和本机 HTTPS 检查；下一步完成 Quest 3S 现场输入与异常/稳定性实测。
