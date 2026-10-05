# Task 1: tokenizer reconstruction FK evaluation — 2026-09-28

本报告只评估 GT action → tokenize → detokenize，不加载 VLA policy。

## Protocol

- 数据：robotwin_new_demo_clean，2500 episodes 中按训练时相同 seed=42、val_ratio=0.05 划分的 125 个 held-out validation episodes。
- 四个 checkpoint 的训练配置已逐一核对：相同数据根目录、相同验证 episode 集合。此集合曾用于 checkpoint selection，不称为独立 test set。
- 815 个完整、不跨 episode、不 padding、不重叠的 32×14 action chunks；步长 32，尾部不足 32 帧舍弃。每个模型处理完全相同的 26080 个 action waypoints。
- q0 取 chunk 起始帧的 observation.state，保存于 shared_samples.npz。RoboTwin take_action(action_type=qpos) 使用绝对关节目标，因此直接计算 action 的 FK，不从 q0 累积。
- 关节排列：左 6 关节、左夹爪、右 6 关节、右夹爪；FK 使用 fl_joint1…6 与 fr_joint1…6，不包含夹爪通道。
- FK 使用 aloha-agilex 的 arx5_description_isaac.urdf、robot_pose、SAPIEN joint-axis frame、global_trans_matrix 和 delta_matrix。末端定义严格匹配 RoboTwin get_arm_pose/get_*_ee_pose，偏移为 gripper_bias − 0.12 m；没有另行替换成 fingertip/TCP。
- 相对位置：世界坐标系中 ||(p_R-p_L)−(p̂_R-p̂_L)||；相对旋转：R_rel=R_LᵀR_R，再比较 GT 与 reconstructed 的 SO(3) geodesic。
- 每个 chunk 先在 32 步内求平均，action MSE 在 32×14 上求平均，再对所有 chunks 等权求均值。
- 95% CI：以 episode 为 cluster bootstrap 5000 次，保留 chunk 权重，报告 mean ± 1.9599639845×bootstrap SD，seed=42。同一 episode 的 chunks 不被当作独立样本。CI 只表示评估数据不确定性，不表示重新训练的 seed 方差。
- Gripper MAE/MSE 使用原始归一化夹爪命令单位，单独报告，不转成 cm，不裁剪预测；action MSE 是包含夹爪的原始 14 维动作 MSE。
- 此结果是绝对目标关节的几何 FK，不模拟 TOPP 插值、动力学、接触或实际执行跟踪误差。

## Results (mean ± 95% CI)

| Metric | BTT | OAT | QueST | Acodec |
|---|---:|---:|---:|---:|
| Left position (cm) | 0.777 ± 0.077 | 0.894 ± 0.092 | 0.683 ± 0.074 | 0.631 ± 0.068 |
| Right position (cm) | 0.783 ± 0.094 | 1.209 ± 0.159 | 0.850 ± 0.103 | 0.775 ± 0.113 |
| Left rotation (deg) | 1.923 ± 0.247 | 2.472 ± 0.327 | 2.038 ± 0.312 | 1.797 ± 0.277 |
| Right rotation (deg) | 1.693 ± 0.225 | 2.735 ± 0.413 | 1.922 ± 0.297 | 2.092 ± 0.345 |
| Relative position (cm), primary | 1.314 ± 0.083 | 1.731 ± 0.138 | 1.272 ± 0.089 | 1.172 ± 0.096 |
| Relative rotation (deg), primary | 3.013 ± 0.286 | 4.256 ± 0.405 | 3.330 ± 0.376 | 3.218 ± 0.400 |
| Left gripper MAE | 0.00185612 ± 0.000197975 | 0.00398569 ± 0.000363604 | 0.00197771 ± 0.000205382 | 0.00152603 ± 0.000189191 |
| Right gripper MAE | 0.00206723 ± 0.00020141 | 0.00414166 ± 0.000319415 | 0.00185925 ± 0.00024486 | 0.0014682 ± 0.000140315 |
| Left gripper MSE | 2.6267e-05 ± 6.71065e-06 | 7.25037e-05 ± 1.61936e-05 | 3.30673e-05 ± 7.99476e-06 | 1.99454e-05 ± 5.46657e-06 |
| Right gripper MSE | 2.73698e-05 ± 6.38656e-06 | 7.58731e-05 ± 1.33122e-05 | 3.71545e-05 ± 1.18699e-05 | 1.62154e-05 ± 3.60408e-06 |
| Action MSE (32×14) | 0.000523788 ± 8.06022e-05 | 0.001067 ± 0.000198551 | 0.000704542 ± 0.000147144 | 0.000647544 ± 0.00014494 |

## Paired primary-metric differences

负值表示 BTT 误差更小。直接对相同 chunk 的误差差值做相同 episode-cluster bootstrap；未进行多重比较校正。

| Comparison | Relative position difference (cm) | Relative rotation difference (deg) |
|---|---:|---:|
| BTT - OAT | -0.4171 ± 0.0848 | -1.2424 ± 0.2450 |
| BTT - QueST | 0.0422 ± 0.0467 | -0.3169 ± 0.1997 |
| BTT - Acodec | 0.1426 ± 0.0584 | -0.2046 ± 0.2130 |

## Verification

- SAPIEN CPU 独立核验 256 个随机 GT waypoints × 2 arms：最大位置差 5.51943084e-07 m，最大旋转矩阵元素差 7.55758435e-07；均小于 1e-5 阈值。
- 单元测试覆盖 0/90/180° geodesic、absolute/delta 语义、相对位置抵消、相对旋转定义、夹爪分离与 cluster 统计。
- 从保存的 reconstructed actions/poses 重算全部 44 项均值及置信区间，结果一致。

## Checkpoints

- BTT: `/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_btt_robotwin_new/621340.hopper-m-02/train/checkpoints/best/step_297000/pretrained_model`
- OAT: `/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_oat_robotwin_new_2500/600693.hopper-m-02/train/checkpoints/best/step_296000/pretrained_model`
- QueST: `/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_quest_robotwin_new/613294.hopper-m-02/train/checkpoints/best/step_299000/pretrained_model`
- Acodec: `/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_acodec_robotwin_new/627717.hopper-m-02/train/checkpoints/recent/step_300000/pretrained_model`

## Files and reproduction

- Evaluator: `scripts/tokenizer/evaluate_fk_reconstruction.py`
- Independent SAPIEN verification: `scripts/tokenizer/verify_robotwin_fk.py`
- Artifacts: `outputs/task1_fk_reconstruction/20260928_validation/`
- `manifest.json`: split, robot/model provenance, dataset hashes and statistical definitions.
- `shared_samples.npz`: all GT chunks, q0, episode/frame indices, full GT joint and end-effector trajectories.
- `<model>_reconstruction.npz`: token IDs, all reconstructed actions, joint trajectories and 4×4 end-effector transforms.
- `<model>_per_chunk.csv`: per-chunk errors; `summary.csv/json`: all estimates and intervals.
- `paired_differences.json`: matched-sample primary-metric differences.
- `fk_verification.json`: independent numerical verification. Executed source snapshots are retained alongside results.

```bash
.venv/bin/python scripts/tokenizer/evaluate_fk_reconstruction.py --output-dir outputs/task1_fk_reconstruction/reproduce --device cpu --threads 8 --batch-size 32
/scratch/xuanweiliu/conda_envs/robotwin/bin/python scripts/tokenizer/verify_robotwin_fk.py outputs/task1_fk_reconstruction/reproduce
.venv/bin/python -m pytest tests/scripts/tokenizer/test_fk_reconstruction.py -q
```
