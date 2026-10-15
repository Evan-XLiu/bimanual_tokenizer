# G1 Episode Collection Guide

Use `collect_episodes.py` to collect G1 demonstrations with Quest VR and external Dex1 grippers. Keep the VR page open throughout the batch. Each attempt requires terminal confirmation and a fresh VR takeover gesture.

## Start collection

One-time robot setup (requires the robot user's sudo password in your terminal):

```bash
.venv-g1-xr/bin/python scripts/unitree_g1/dex1_service.py --install
```

This installs a root-owned, fixed Dex1 recovery helper and grants `unitree` passwordless access to that helper with no arguments and the exact command `/usr/bin/systemctl restart dex1_gripper.service`. No other systemctl commands are granted passwordless access. It backs up the original service definition to `/etc/systemd/system/dex1_gripper.service.d/original-unit.backup`, then replaces the original unit's `After=multi-user.target` line with `After=basic.target systemd-udev-trigger.service`. Ordering dependencies cannot be removed by clearing `After=` in a drop-in. This releases Dex1 from the wait on `data_sync.service` without changing data synchronization. It reloads systemd, verifies the effective dependencies, and starts Dex1 if inactive; it does not restart an active Dex1. The repair persists across reboots. Rerun the installation command once with sudo authentication to add restart permission to an older installation.

After restoring a disconnected gripper USB cable, explicitly restart Dex1 to reopen its serial ports, then run the read-only preflight before collecting:

```bash
ssh -i .local/unitree_g1/ssh/id_ed25519 unitree@192.168.123.164 \
  'sudo -n /usr/bin/systemctl restart dex1_gripper.service'
```

Each `--execute` collection invokes the helper once before robot checks and records its result in `checks/dex1_service.json`. Missing installation, authorization failure, or startup failure stops collection. Both grippers must still pass the existing live DDS feedback checks. Recovery is not retried during an episode; `--preflight-only` does not start or repair robot services. The robot SSH host/key are the existing `--camera-host` / `--camera-ssh-key` arguments.

Run from the repository root in an interactive terminal. The example uses the setup from October 14, 2026; check the network addresses, interface, and certificate paths before reuse.

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv-g1-xr/bin/python scripts/unitree_g1/collect_episodes.py \
  --episodes 10 \
  --task "Bottle 01" \
  --output .local/unitree_g1/collections/bottle_01 \
  --interface enp3s0 \
  --host-address 172.20.10.13 \
  --cert .local/unitree_g1/tls/rotation-20261014-001/cert.pem \
  --key .local/unitree_g1/tls/rotation-20261014-001/key.pem \
  --camera-host unitree@192.168.123.164 \
  --camera-ssh-key .local/unitree_g1/ssh/id_ed25519 \
  --execute --scene-confirmation True
```

With the same setup, normally change only `--episodes`, `--task`, and `--output`. Keep the other arguments. The output directory must not already exist. `--scene-confirmation True` confirms that the robot is standing, the workspace is clear, and you can stop the run.

At startup, the script checks the robot NIC's IPv4 address. If it has no IPv4 address, it activates the existing NetworkManager profile `g1-direct-thinkcentre` (`enp3s0`, `192.168.123.223/24`, no default gateway). This also applies to `--preflight-only`; robot checks remain read-only. NetworkManager may require local authorization. The Wi-Fi address used by `--host-address` is separate. For another machine, select its dedicated profile with `--network-connection NAME`; an empty value disables automatic activation. The script refuses to replace an unrelated IPv4 configuration or activate a profile bound to another NIC.

For device checks only, replace `--execute --scene-confirmation True` with `--preflight-only` and use a separate output directory. This mode does not move the robot or start VR.

## Collect each episode

1. Wait for startup checks, then open the printed VR URL and enter VR.
2. Reset the scene. Enter **`n` + Enter** to start this attempt's checks and gripper opening.
3. Wait for **`Ready`**. Release both side grips, then hold both to take control.
4. Perform the task. Release either side grip to end the episode.
5. Wait for **`Finished`**, then enter a decision:

| Key | Action |
| --- | --- |
| `s` + Enter | Confirm valid/successful data. Save raw data and queue export, then increment the accepted count. |
| `f` + Enter | Delete this attempt's raw data without counting it. Small diagnostic logs remain. |
| `q` + Enter | Quit. Keep any unlabelled attempt in `pending/`. |

After `s` or `f`, the script waits for another **`n`**. This also applies to retries. When the accepted count reaches `--episodes`, collection ends and queued exports begin. `q` or Ctrl-C ends collection early and exports already accepted episodes after child/VR cleanup. Encoding is sequential, with one progress message per episode and details in its `export.log`. An export failure retains raw data and does not prevent the remaining queued episodes from being processed. A forced kill or power loss cannot run exit-time export.

`Try` counts attempts. `Episode` identifies the current accepted-data slot: after `f`, Try increases but Episode stays the same. Both display from 1; directory numbering starts at 0.

## Output and failures

- Accepted raw data: `<output>/episodes/episode_000000/`; encoded data appears in `lerobot/` after batch export.
- Batch progress: `<output>/collection.json`. Schema v2 `accepted` lists raw episode directories; `exported` lists verified dataset directories. `accepted_count` and `exported_count` are separate, and `export_results` records each outcome.
- Checks: `<output>/checks/`
- Attempt logs: `<output>/logs/attempt_000001.log`
- Unlabelled or interrupted data: `<output>/pending/`

Each saved episode is an independent BiCAT/G1 LeRobot v3 dataset, with 30 Hz output and 16-dimensional G1 joint vectors. The command above defaults to [camera_capture_profile.json](../../artifacts/unitree_g1/camera_capture_profile.json), updated on October 15, 2026 for the new RealSense D435i head camera. Head RGB uses V4L2 YUYV at 640×480, requesting 30 FPS; its fixed USB path is `/dev/v4l/by-path/platform-3610000.xhci-usb-0:2.3:1.3-video-index0`, so keep the current USB port. The 20-second simultaneous capture test measured head 29.23 FPS and both wrists about 30 FPS, with all JPEGs decoding successfully. Raw head JPEGs are 640×480; the current exporter resizes them to 640×360 for `observation.images.top`. Both wrists remain 640×480 and use stable serial-number V4L2 paths: left `255323071511`, right `260323072820`, confirmed by the operator from previews. Repeated images are recorded in the quality metadata. See the [camera validation report](../../artifacts/unitree_g1/head_camera_validation_20261015.json).

An attempt fault retains its raw data in `pending/attempt_*`, with `collection_fault.json` recording the failure and cleanup result. The child attempts its existing bounded unload before returning. The collector stays open at **`r` + Enter: recheck / `q`: quit**, keeping the VR service running. After fixing the issue, `r` checks arm/gripper release evidence, camera recorder termination, input-route closure, robot/gripper feedback, three-camera capture, disk space, and fresh VR input. A failed recheck returns to the same prompt; a passing recheck returns to the separate **`n`** prompt. It never starts motion automatically. Faulted attempts remain uncounted and cannot be labelled with `s/f`; `f` applies only to demonstrations that ended normally.

If the summary is missing, motor release is unconfirmed, or camera recorder termination is unknown, rechecking sensors cannot clear the block. The program stays paused; use `q` to exit for external recovery rather than editing the recorded evidence. A dead VR service also blocks recovery. Initial startup failures still stop the batch. Export failures occur only after collection has ended. Recovery check logs have unique paths and do not overwrite earlier diagnostics.

During release (`RELEASING`), joint tracking deviation no longer triggers a new fault as control authority fades. Tracking checks remain enabled during engagement and active control, and faults already triggered remain latched. Release still checks feedback validity/freshness, clocks and servo deadlines, robot mode and command ownership, and command submission. Camera completion and arm/gripper cleanup must still pass before the save/discard prompt. The shared arm execution state machine applies this rule to all callers; it does not reclassify historical faulted recordings.

For the current `external-dex1` grippers, tracking deviation no longer raises `Gripper tracking_error` (`tracking_error: null`). The separate closing-obstruction threshold remains 0.50: it freezes further closing on the affected side, permits an opening request, and does not abort the attempt. Feedback/input timeouts, valid motor data, command position/gain bounds, and trajectory limits remain enabled. The legacy internal-gripper backend retains its original tracking fault behavior.

Defaults: 180 seconds of active recording per attempt, 300 seconds to wait for takeover, and 10 GiB minimum free space. Override with `--active-seconds`, `--wait-seconds`, and `--min-free-gb`.

Quest input freshness now allows 1.0 second across the VR service, IK target checks, arm targets, and gripper trigger inputs. Robot/gripper feedback freshness remains 0.2 seconds. The IK position residual limit is 0.03 m (3 cm). Left and right arms backtrack independently through fractions 1, 0.5, 0.25, 0.125, and 0; only a side exceeding the residual limit reduces its fraction. Every paired solve revalidates both arms, with at most nine solves. If the previous pose cannot be revalidated, the attempt still faults. Diagnostics record `accepted_fractions` and `ik_limited_sides` in left/right order; legacy `accepted_fraction` is their minimum, not a shared control fraction.

The continuous workflow has passed offline tests; physical robot and Quest acceptance is still pending. See [recording details](g1_lerobot_recording.md) for the full implementation notes.
