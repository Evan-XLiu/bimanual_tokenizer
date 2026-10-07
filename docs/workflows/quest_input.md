# Quest 3S 双手柄原始输入采集

2026-10-07 起，[G1 计划](../plans/unitree_g1_vr_to_real_plan.md)以官方 `xr_teleoperate` 为主线。本入口保留为阶段一的原始输入诊断工具；正式接入需另行验证官方 `televuer` 链路。它不导入机器人 SDK，不做 IK、不发送机器人指令，也不生成 LeRobot 训练数据。仅需 Python 3.10+ 和 Quest 浏览器，无额外 Python 或 npm 依赖。

2026-10-06：完成采集器、浏览器客户端和离线/本机 HTTP 测试；HTTPS 服务本机访问通过。Quest 3S 尚未实测，不能标为阶段二完成。

## 通过 Wi-Fi 连接

电脑和 Quest 连接能够互通的局域网。校园/访客 Wi-Fi 可能隔离设备，即使 SSID 一样也可能无法互访；这种情况下换可互访热点或使用下述 USB 方案。无需连接 G1，也无需启动 Quest Link。

1. 用 `ip -br addr` 查看电脑 Wi-Fi 地址。本次为 `172.25.100.44`，地址变化时替换证书 SAN 和访问地址。
2. 在实验目录生成本地 TLS 证书。下面示例只供初次试采，目录和密钥放在 `/tmp`，重启后可能丢失；正式采集改为外部持久目录。已有目录时不要覆盖密钥。

   ```bash
   mkdir -m 700 /tmp/quest-input-tls
   openssl req -x509 -nodes -newkey rsa:2048 -days 30 \
     -keyout /tmp/quest-input-tls/key.pem \
     -out /tmp/quest-input-tls/cert.pem \
     -subj '/CN=quest-input-local' \
     -addext 'subjectAltName=IP:172.25.100.44,IP:127.0.0.1,DNS:localhost'
   python3 scripts/unitree_g1/quest_record.py \
     --bind 0.0.0.0 --port 8443 \
     --cert /tmp/quest-input-tls/cert.pem --key /tmp/quest-input-tls/key.pem \
     --output /tmp/quest-input-session-001
   ```

3. 终端显示 `#token=...` 连接码。Quest 浏览器打开 `https://电脑Wi-Fi地址:8443/`，在页面填入连接码；也可打开包含 `#token=...` 的完整链接。自签名证书可能触发证书提示，只对本机生成的这个地址确认继续。若继续后页面仍提示 WebXR 不可用，使用受信任证书或 USB localhost 方案；不关闭浏览器全局安全检查。
4. 拿起两个手柄，选择“透视 AR”，点击“进入 XR 并开始录制”，允许浏览器 XR 权限。若浏览器不支持此模式，选择“沉浸 VR”再试。左手显示蓝点，右手显示橙点。AR 仅显示系统透视，程序不读取或录制透视相机图像。
5. 电脑打开同一 URL，观察接收帧数增长、左右手位姿和按键值。不要在电脑端点击开始。
6. 点击“结束并保存”，或用头显系统菜单退出 XR。每次重新进入新建一个 episode；电脑端 `Ctrl-C` 关闭服务。

WebXR 使用安全上下文并由用户手势发起会话，参见 [WebXR 规范](https://www.w3.org/TR/webxr/)。宇树输入模块同样使用 HTTPS 连接 Quest，见 [TeleVuer 连接说明](https://github.com/unitreerobotics/televuer)。本入口直接采集 WebXR 原始输入，未使用 TeleVuer 的机器人坐标变换或共享内存缓存。

## USB localhost 备选

需要在 Quest 开启开发者模式、安装电脑端 Android Platform Tools，并在头显接受本电脑的 USB 调试授权。本次电脑尚未发现 `adb` 或 USB Quest，以下步骤未在本机头显实测：

```bash
adb devices
adb reverse tcp:8443 tcp:8443
python3 scripts/unitree_g1/quest_record.py \
  --bind 127.0.0.1 --port 8443 --output /tmp/quest-input-usb-001
```

先停止占用 8443 端口的 HTTPS 服务。Quest 浏览器访问 `http://localhost:8443/` 并填写新进程的连接码。这里使用 ADB 端口转发，不是 Quest Link 串流；localhost 可作为安全上下文，局域网明文 HTTP 地址不可以。结束后可执行 `adb reverse --remove tcp:8443`。

## 保存的数据与时间语义

```text
session/
  session.json                # schema、Git HEAD、采集器和前端的 SHA256
  <episode_uuid>/
    metadata.json             # 浏览器、显示模式、坐标与时钟约定
    raw.jsonl                 # 每个 XR 帧及输入/重定位/可见性事件
    summary.json              # 正常结束或断流原因、帧数、间隔统计
```

- 坐标为 WebXR `local-floor`：右手系，+X 向右、+Y 向上、−Z 向前，位置单位米，姿态为 `xyzw` 四元数。尚未标定到机器人坐标系。
- `head`、`left/right.grip`、`target_ray` 保留各自位姿；缺失记为 `null`。`emulated_position` 原样保存，不以旧姿态填补追踪丢失。
- `buttons` 保存完整 value/pressed/touched，`axes` 保存完整摇杆值，另存 profiles 和 gamepad mapping。只有 `xr-standard` 映射时，button 0/1 才按标准解释为 trigger/squeeze，参见 [WebXR Gamepads 规范](https://www.w3.org/TR/webxr-gamepads-module-1/)。当前不将按键转换成夹爪命令，也未实现遥操作离合。
- `xr_frame_time_ms` 是浏览器 XR 回调给出的预测显示时间，`client_time_ms` 是打包时的 `performance.now()`；它们不是原始传感器采样时间。`performance_time_origin_ms` 仅保留浏览器时钟原点，不声称已完成跨设备时钟同步。
- 浏览器每个 XR 帧采样，约每 100 ms 串行发送一批；所有帧保留，不降采样到 20 Hz。每批保留主机 `host_receive_monotonic_s`，不能当作逐帧采样时间或用其估计真实端到端延迟。此传输仅用于输入采集，不适用于机器人伺服。
- 每条输入具有连续 `seq`；重复、缺号、时间倒退或非法位姿会拒绝整批。前端发送失败/积压时停止当前会话，不静默重试可能已写入的数据。默认超过 2 秒没有批次则结束 episode，恢复必须重新进入 XR。
- `reference_reset` 增加 `reference_epoch`，事件还保存可获得的原点变换；后续处理必须在重定位、长时间缺帧处切段。没有收到 reset 事件不证明没有漂移。
- `summary.json` 的 `valid_both_frames` 仅表示头部和双手柄 grip 位姿可用、非模拟位置、会话可见，不代表标定正确或整段质量验收通过。帧间隔 P95 是浏览器时钟内的统计，不是机器人/相机同步误差。
- 每批刷新原始文件，结束时 fsync。进程被强杀或断电时可能没有 summary 或尾行不完整，应判为未正常完成并人工检查，不能当成成功 episode。

## 首次现场检查

1. 录制 30 秒：逐一移动左右手的左右/上下/前后方向，检查身份、轴向和米制尺度；确认双手位置独立变化。
2. 分别按左右扳机、侧握键和摇杆，检查原始值；不要在接口未核验前赋予机器人控制意义。
3. 暂时遮挡/放下一个手柄，检查位姿缺失或模拟标志；退出、重新进入，确认生成新 episode。
4. 测试系统重定位，检查 `reference_reset`、`reference_epoch`；若浏览器不报告，记录该限制。
5. 断开 Wi-Fi，确认电脑端约 2 秒后显示 `stream_timeout` 且不再显示旧位置；恢复后重新进入开始新轨迹。
6. 连续采集 10 分钟，查看 frame gap、有效帧占比和事件。保留异常轨迹，记录浏览器/系统版本，再决定是否进入标定和 IK。

本轮尚缺机器人状态、实际提交的 16 维动作和相机。因此这些日志是 Quest 接入证据，不是可直接用于 BTT/VLA 的示范数据。后续需按计划接统一执行层、相机与时钟对齐后再实现 LeRobot 转换。

## 软件验证

```bash
python3 -m unittest discover -s tests/unitree_g1 -v
node tests/unitree_g1/test_quest_client.cjs
```

Python 测试包含本机回环 HTTP 集成，需要允许创建本地 socket；JS 测试验证发送队列、异常和结束时的最后一批写入，不代替 Quest 浏览器实测。
