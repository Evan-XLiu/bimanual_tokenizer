# Quest 驱动虚拟 G1 双臂

项目当前进展与下次接手任务见[交接入口](unitree_g1_handoff.md)。本页只描述虚拟链路；真机原姿态保持及 Quest 受限跟随已另行试验，见[真机连接流程](quest_g1_bridge.md)。

入口：`scripts/unitree_g1/virtual_arms.py`。复用固定版本的 TeleVuer 输入和官方 G1 IK，在电脑浏览器中显示完整 G1 模型；腰腿和手指锁定，只更新 14 个手臂关节。无需机器人连接，不初始化 DDS，也不实例化硬件控制器。SDK、DDS、官方硬件控制器与遥操作主入口的导入被阻断。

这是运动学可视化，不是动力学或碰撞验收。模型包含上游 Dex3 外观，不代表现场夹爪已适配。扳机不控制模型手指。Quest 使用透视输入，G1 模型在电脑端显示。

## 启动

使用已固定的 `.venv-g1-xr` 环境，安装与离线验证见 [软件基线](unitree_g1_baseline.md)。电脑和 Quest 应在可互访的局域网，证书 SAN 须包含本次电脑地址；证书生成方法见 [TeleVuer 输入采集](televuer_input.md#tls-证书)。

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv-g1-xr/bin/python scripts/unitree_g1/virtual_arms.py \
  --cert /绝对路径/cert.pem --key /绝对路径/key.pem \
  --host-address 电脑局域网IP \
  --output .local/unitree_g1/experiments/virtual-session-001
```

输出目录必须不存在。启动会打印两个地址：

- Quest：`https://电脑局域网IP:8012/?ws=wss://电脑局域网IP:8012`，进入 `pass-through`。
- 电脑：通常为 `http://127.0.0.1:7000/static/`，以实际打印地址为准。在电脑浏览器打开，可用鼠标调整观察角度。

8012 端口由此命令的输入子进程占用，不要同时启动另一个 TeleVuer 输入服务。服务只供受控局域网使用，Ctrl-C 同时停止输入子进程和模型服务器。模型服务器仅监听本机。

## 操作

1. 双手保持自然姿势，**先完全松开两侧侧握键**，进入 `ready`。
2. **同时握住两侧侧握键**，记录当前手柄与虚拟末端参考；接管这一帧不改变模型姿态。
3. 保持握住，小幅平移或旋转手柄，虚拟左右臂分别跟随对应手柄。
4. 松开任意一侧，两个手臂停止更新。移动手柄到舒适位置后重新握住，重新取参考，无需回到原手柄位置。
5. 断流、重新连接、非法位姿、明显跳变、目标超范围或 IK 失败会停止跟随；故障后须先完全松开两侧再握住。

坐标使用固定 WebXR 世界基底：原始 −Z 对应机器人向前、−X 对应向左、+Y 对应向上。位置默认按 1:1 相对位移映射，可用 `--scale` 调整；旋转使用世界坐标中的相对旋转。**没有叠加官方 `head_yaw` 转换**，转头不会主动改变手臂参考。实际左右、前后、上下方向仍须现场逐项核验；重定位前先松开侧握键。

默认输入超时为 0.25 秒；单帧位移超过 0.12 米、旋转超过 0.8 弧度或相对参考位移超过 0.35 米时停止跟随。显示关节速度上限为 1.5 rad/s。这些都是虚拟调试参数，不是已验证的硬件限制。上游没有可靠的追踪丢失/重定位标志，小幅漂移或持续发送旧位姿仍可能无法识别。

## 日志与复现

```text
session/
  session.json       # 模式、坐标、比例、限制
  viewer.json        # 电脑模型页面地址（有显示时）
  status.json        # 跟随状态、已解算帧、IK 失败计数、14 维虚拟关节
  joints.jsonl       # 每次接受的虚拟关节、IK 目标、候选 IK 位置残差
  summary.json       # 结束原因与计数
  input.log          # 输入子进程日志
  input/             # TeleVuer 原始事件和输入诊断
```

`joints.jsonl` 是虚拟关节日志，不能当作真机最终指令或训练示范。`ik_position_error_m` 是限速前候选解的残差；显示关节受速度限制，暂时可能落后于目标。

按原始时间间隔回放已有手柄日志，同样需要日志中包含松开→双侧握住的动作：

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv-g1-xr/bin/python scripts/unitree_g1/virtual_arms.py \
  --replay /绝对路径/events.jsonl \
  --output .local/unitree_g1/experiments/virtual-replay-001
```

`--headless` 只省略模型窗口，仍运行 IK 和记录日志；`--duration 30` 在输入循环运行 30 秒后退出。

## 验证范围

2026-10-08：离合、重接管、坐标旋转、跳变、超时、旧事件及半行 JSONL 测试通过；真实官方 IK 连续完成 40 帧双臂目标，检查两臂关节变化和显示速度限制。合成手柄日志通过完整 CLI 回放，接受 58 帧 IK，无失败。当时的“27 项”统计不准确；交接整理时当前完整 G1 工具测试集实际为 38 项并通过，包含后来新增的真机执行核心/保持循环离线测试。

同日完成 Quest 实时跟随，用户确认“看到了，双臂可以跟随”。按用户要求停止后，输入与虚拟跟随摘要均为 `operator_stop`，输入子进程和模型服务器已退出。[最终统计](../../artifacts/unitree_g1/virtual_follow_final.json)记录：

| 项目 | 结果 |
| --- | --- |
| 服务运行时长（含等待/暂停） | 743.09 秒 |
| 头部 / 手柄事件 | 11,443 / 3,632 |
| 格式无效事件 / 接收队列丢包 | 0 / 0 |
| 接受的虚拟跟随帧 / IK 目标拒绝 | 1,172 / 8 |
| 同连接手柄接收间隔 P50 / P95 | 33.00 / 38.58 ms |
| 最长同连接接收间断 | 1.986 秒 |
| 最长连续手柄片段（超过 250 ms 切段） | 60.62 秒，1,834 条事件 |
| 接受的候选 IK 位置残差 P50（左 / 右） | 4.63 / 4.16 mm |
| 接受的候选 IK 位置残差 P95（左 / 右） | 47.62 / 31.17 mm |

运行时观察到目标不可达、相对范围限制和输入超时停止。此版本没有持久化逐次状态切换，无法从落盘日志给出各类停止的精确次数。位置残差针对限速前候选解，不是实际显示关节误差或真机跟踪误差。全部 JSONL 可解析、计数与摘要一致、接受帧时间严格递增，报告保存原始文件 SHA256。早期 [473 帧快照](../../artifacts/unitree_g1/virtual_follow_partial.json)作为历史记录保留。浏览器模型渲染也已截图检查；本次不计为 10 分钟连续稳定性通过。

仍需系统核验左右手独立移动、三轴方向、旋转、松开保持与重新握住的连续性，以及主动断网、追踪丢失和重定位行为；10 分钟稳定性与碰撞/动力学验证另行进行。

当前自动化测试使用 XR 环境：

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv-g1-xr/bin/python -m unittest discover -s tests/unitree_g1 -v
```
