# Quest 3S 官方 TeleVuer 独立输入验证

虚拟双臂跟随入口已新增，见 [Quest 驱动虚拟 G1 双臂](virtual_g1_arms.md)。该入口复用本工具作为独立输入子进程，再接相对离合、官方 IK 和电脑模型显示，不连接真实机器人。

2026-10-07：固定基线复查（含离线 IK）、4 项自动化测试和 HTTPS 本机检查通过。Quest 3S 官方头部/双手柄输入已实测，按用户要求结束采集。第二会话正常保存 `operator_stop`，收到 6,469 个头部事件、2,170 个手柄事件，无格式无效事件或队列丢包。按 1 秒间断切段，最长手柄连续片段 68.32 秒（2,071 个事件），最大接收间隔 209.1 ms；双侧扳机/侧握键均覆盖 0–1，摇杆正负方向、摇杆按下和面键均已收到。

本轮仅确认基础输入接收与按键覆盖，未完成 10 分钟稳定性、追踪丢失、重定位、主动断网及物理轴向/尺度核验。第二会话初始有一次 26.90 秒手柄间断，原因待确认；首会话服务退出且没有正常 summary，也保留为异常证据。详见[部分现场报告](../../artifacts/unitree_g1/quest_field_partial.json)。原始日志已从 `/tmp` 复制到 Git 忽略的 `.local/unitree_g1/experiments/quest_20261007/`，报告记录了 SHA256。TLS 私钥未归档或提交。

入口：[televuer_input.py](../../scripts/unitree_g1/televuer_input.py)。仅实例化锁定版本的官方 `TeleVuer`，使用 `pass-through`、双手柄、关闭 ZMQ/WebRTC 视频；不实例化 IK、相机或机器人控制器。额外 import guard 拒绝 SDK、DDS 和 `teleop` 控制模块。没有使能机器人或自动恢复运动的接口。

## 启动

Quest 输入统一使用本工具的 **8012 端口**。虚拟双臂入口也使用同一个输入服务，不要同时启动两者。电脑模型页面通常使用 7000 端口，以启动时打印的地址为准。

先按[基线流程](unitree_g1_baseline.md)校验环境。电脑与 Quest 连接可互访的局域网；IP 从 `ip -br addr` 获取。校园/访客网络若隔离设备，可换用可互访热点。无需 Quest Link。

### TLS 证书

证书及日志放在本地实验目录，勿提交密钥或原始数据。首次使用时，将 `QUEST_HOST` 替换为实际电脑 IPv4 地址，并选择尚不存在的证书目录：

```bash
export QUEST_HOST=电脑局域网IP
export QUEST_TLS_DIR="$PWD/.local/unitree_g1/tls/session-001"
mkdir -p "$(dirname "$QUEST_TLS_DIR")"
mkdir -m 700 "$QUEST_TLS_DIR" && (
  umask 077
  openssl req -x509 -nodes -newkey rsa:2048 -days 30 \
    -keyout "$QUEST_TLS_DIR/key.pem" -out "$QUEST_TLS_DIR/cert.pem" \
    -subj '/CN=quest-televuer-local' \
    -addext "subjectAltName=IP:${QUEST_HOST},IP:127.0.0.1,DNS:localhost"
)
```

已有匹配地址的证书可直接复用；电脑 IP 改变时应重新生成包含新地址的证书。自签名证书可能触发浏览器提示，只对自己生成的本机地址确认继续；若仍无法进入 XR，使用受信任证书。

### 输入采集

以下变量使用实际证书路径和尚不存在的新日志目录：

```bash
# 使用实际 IP、证书路径和未存在的新日志目录
export QUEST_HOST=电脑局域网IP
export QUEST_CERT=/外部实验目录/cert.pem
export QUEST_KEY=/外部实验目录/key.pem
export QUEST_SESSION=/外部实验目录/quest-session-001

env -u PYTHONPATH PYTHONNOUSERSITE=1 .venv-g1-xr/bin/python \
  scripts/unitree_g1/televuer_input.py \
  --cert "$QUEST_CERT" --key "$QUEST_KEY" \
  --host-address "$QUEST_HOST" --output "$QUEST_SESSION" \
  --device-note 'Quest 3S; 填写系统和浏览器版本'
```

证书 SAN 必须对应本次地址。固定上游监听 `0.0.0.0:8012`，`--host-address` 只改变打印链接，不改变监听地址。浏览器打开终端打印的 `https://IP:8012/?ws=wss://IP:8012`，确认本机证书后选择 `pass-through`。不需要启动 `teleop_hand_and_arm.py`，不需要连接 G1。

Vuer 静态文件目录重定向至日志目录中的空文件夹，避免其默认 `/static` 暴露工作目录。服务没有访问鉴权，只在受控局域网测试，用完 Ctrl-C 停止。`--duration` 是从程序启动起的墙钟时长，并非有效追踪时长；10 分钟稳定性须根据真实事件区间审核。

## 记录与解释

- `session.json`：基线锁、入口哈希、设备备注、模式和限制。
- `events.jsonl`：官方 `CAMERA_MOVE` / `CONTROLLER_MOVE` 原始事件、连接 ID、主机接收单调时间、数据结构有效性；每个事件写入，不轮询复制缓存。
- `official_head_yaw`：同一连接头部数据在超时范围内时，附带官方 wrapper 转换结果。手柄原始矩阵按列优先解析，转换结果为常规二维数组；官方扳机值会转换为 `10 - 10 * raw`，不是原始 0–1 值。
- `status.json`：每秒刷新。`receiving` 仅表示合法头部与双手柄事件均在默认 1 秒内到达；`stale` 表示接收超时或无效，`waiting` 表示尚无事件。上游共享内存旧值不会使此状态继续有效。
- `summary.json`：正常结束时的计数、无效数据、丢队列事件和手柄主机接收间隔统计；保留 `field_acceptance=pending_operator_review`，不会自动宣称现场通过。

本入口不做时钟同步，接收间隔不等于设备帧间隔或端到端延迟。有限且正交的矩阵仅说明格式合理，不证明真实追踪有效。上游缺少可靠的单手追踪有效性、XR 退出和参考系重定位事件；即使仍有合法数据，也可能是浏览器保留的旧位姿。因此遮挡、重定位及重新进入须由操作者按时间标注，并对照本入口的原始事件日志检查。WebSocket 重连可区分连接；同一连接内重进 XR 不保证产生新 ID。当前入口用于诊断，未实现遥操作离合或伺服输入保护。

## 现场验收步骤

按顺序记录每步开始/结束的本机时间、实际现象和异常，不把服务运行时间视为头显输入时长：

| 步骤 | 操作与证据 |
| --- | --- |
| 设备信息 | Quest 型号、系统版本、浏览器版本、Wi-Fi/热点拓扑 |
| 30 秒基础 | 左右手依次左右/上下/前后移动和旋转，确认身份、独立变化、米制位移；分别扳机/侧握键/摇杆/面键 |
| 追踪丢失 | 遮挡或放下一侧，记录上游实际行为；若仍发送旧值必须记录为局限 |
| 重定位 | 操作系统重定位，记录时间并检查位姿跳变；不声称上游自动报告 reset |
| 断网 | 关闭头显 Wi-Fi，确认 `status.json` 变 stale；恢复网络并重新进入，检查连接/首帧 |
| 退出重进 | 退出 XR、重新进入，记录是否沿用连接以及坐标变化 |
| 10 分钟 | 单独保留连续有效输入区间，审核间隔、跳变、按键和掉线；解释每个异常 |

第一轮报告还应明确显示仅测试系统透视；机器人相机视频、IK/仿真和真机控制均不属于本入口验收。

## 软件测试

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 .venv-g1-xr/bin/python \
  -m unittest discover -s tests/unitree_g1 -p test_televuer_input.py -v
```

覆盖无效矩阵/按键、接收超时与换连接、机器人导入阻断、官方事件回调与坐标/扳机转换。官方回调测试禁止启动服务进程，不生成硬件发布者。
