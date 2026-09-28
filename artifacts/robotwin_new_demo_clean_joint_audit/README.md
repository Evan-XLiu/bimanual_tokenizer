# robotwin_new_demo_clean joint distribution audit

- Dataset: `/scratch/xuanweiliu/bimanual_tokenizer/data/lerobot/local/robotwin_new_demo_clean`
- Rows / episodes / task instructions: **548,985 / 2,500 / 2,410**
- Joint layout used for analysis: `[left_joint_0..5, left_gripper, right_joint_0..5, right_gripper]` (the 14-D order used by the policy adapter).

## Findings

- **No NaN/Inf or shape corruption:** state/action are `(548985, 14)` and all values are finite.
- **Temporal indexing is consistent:** every episode starts at frame 0, frame indices increment by 1, and intra-episode timestamps are 1/15 s.
- **Action/state alignment is correct:** for all 546,485 intra-episode pairs, `action[t]` equals `observation.state[t+1]` to float32 precision. This is next-state action semantics, not a data shift bug.
- **Grippers are valid:** all values stay in `[0, 1]`; they contain both endpoint values and short continuous transitions, which is expected.
- **No large intra-episode arm jumps:** `0` arm transitions exceed 1 rad; `5` exceed 0.5 rad (all are isolated <= 0.830 rad events).
- **Single-arm zero padding is present and systematic:** 665 left-only, 716 right-only, and 1,119 bimanual episodes. Inactive-arm zeros are therefore expected, not missing data.
- **No obvious 2π wrap/jump signature** appears inside episodes; the largest arm adjacent delta is below 1 rad.

## Potential Issue

- `meta/info.json` names both 14-D features generically as `joint_0` ... `joint_13`, while the actual order is the bimanual ALOHA layout above. Numeric order is internally consistent with the policy code, but the metadata should be corrected to canonical left/right joint and gripper names for downstream tooling and human inspection.

## Per-joint action ranges

| joint | min | q01 | median | q99 | max | std | zero fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| left_joint_0 | -1.3947 | -0.9551 | -0.0668 | 0.1734 | 0.4364 | 0.2928 | 0.288 |
| left_joint_1 | -0.0003 | -0.0000 | 1.5256 | 2.5710 | 3.3889 | 0.9922 | 0.288 |
| left_joint_2 | -0.1191 | -0.0000 | 0.7896 | 2.4529 | 3.4515 | 0.7818 | 0.286 |
| left_joint_3 | -1.9268 | -1.5885 | -0.0000 | 1.3255 | 1.7898 | 0.6739 | 0.288 |
| left_joint_4 | -1.4217 | -0.4365 | 0.0000 | 1.2155 | 1.5633 | 0.2503 | 0.288 |
| left_joint_5 | -3.4688 | -2.1044 | 0.0000 | 1.4361 | 3.0108 | 0.5463 | 0.288 |
| left_gripper | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.4527 | 0.284 |
| right_joint_0 | -2.9456 | -0.1490 | 0.0863 | 1.0682 | 1.3893 | 0.3129 | 0.336 |
| right_joint_1 | -0.5743 | 0.0000 | 1.5450 | 2.5955 | 2.9951 | 1.0173 | 0.336 |
| right_joint_2 | -0.0053 | -0.0000 | 0.8560 | 2.4843 | 3.3334 | 0.8054 | 0.334 |
| right_joint_3 | -1.9894 | -1.6976 | 0.0000 | 1.3109 | 1.9531 | 0.7209 | 0.336 |
| right_joint_4 | -1.6339 | -1.0276 | 0.0000 | 1.0605 | 1.3971 | 0.2616 | 0.336 |
| right_joint_5 | -2.9398 | -1.5555 | 0.0000 | 2.1072 | 3.4248 | 0.5937 | 0.336 |
| right_gripper | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.4519 | 0.282 |

## Artifacts

- `joint_stats.csv`: full state/action quantiles and temporal-jump counts per joint.
- `episode_stats.csv`: per-episode length, active arms, extrema, and instruction.
- `episode_outliers.csv`: top 50 episodes by adjacent arm delta.
- `task_stats.csv`: episode/frame counts and active-arm pattern per instruction.
- `report.json`: machine-readable checks and official-shard comparison.

