# BTT right-first VLA training and conditional handover_mic evaluation

Submitted on 2026-10-01 (Asia/Singapore).

- Training: `641958.hopper-m-02`, `bttRL30k`.
- Evaluation: `641959.hopper-m-02`, `bttRLmic100`.
- Dependency: `afterok:641958.hopper-m-02`.
- Initial scheduler state: training `Q` (small queue running-job limit); evaluation `H` (dependency).

## Configuration

The reference is `outputs/hopper_bar_btt_pali_robotwin_new_30k/622038.hopper-m-02/resolved_config.yaml`, traced from `outputs/robotwin_eval/rollout8task_100/BTT/handover_mic/run_manifest.json`.

All training fields match the reference except `policy.btt_arm_order`, `run_name`, and `output_dir`. The frozen tokenizer's four artifact hashes are verified against the original evaluation manifest before training.

Tokenizer: `outputs/hopper_btt_robotwin_new/621340.hopper-m-02/train/checkpoints/best/step_297000/pretrained_model`.

Training uses `policy.btt_arm_order=right_then_left`, two blocks `(8, 8)`, PaliGemma2 3B full fine-tuning, batch size 64, seed 42, 30000 optimizer steps, original learning rates and EMA settings. Labels become `[R0..7,L0..7]`; the second block receives first-block tokens. Generated IDs are restored to `[L0..7,R0..7]` before calling the unchanged codec decoder. Default `left_then_right` preserves existing checkpoints and behavior. The arm-order field is persisted in the policy checkpoint and is not a pretrained runtime override.

W&B: online, project `praxis-vla`, entity `evan_lxw-harbin-institute-of-technology`, run name `robotwin_new_bar_btt_right_then_left_pali_30k`, run ID `btt-right-first-641958`. Account connectivity verified before submission; successful training additionally requires checkpoint W&B run metadata.

Training output: `outputs/hopper_bar_btt_right_first_pali_robotwin_new_30k/641958.hopper-m-02/`.

The successful training script verifies `latest/training_state/training_step.json` equals 30000, exports the final pretrained model to `train/checkpoints/step_30000/pretrained_model`, and writes `training_complete.json`. The dependent evaluator checks this marker and the persisted right-first configuration.

Evaluation uses `handover_mic`, `demo_clean`, seen instructions, seed 0, 100 episodes, 100 recorded videos, max seed attempts 5000, the original shared evaluation helper, and the final step-30000 model. Output: `outputs/robotwin_eval/rollout8task_100/BTT_right_then_left/step30000/handover_mic/`.

## Validation

- Six new tests pass: left/right training label order, previous-block teacher forcing, two conditional inference steps, native codec decode layout, config serialization/backward compatibility, invalid codec/order and incomplete decode rejection.
- Existing one-block tests and token-sequence tests pass. Broader selected suite: 38 passed, 4 failed. The four pre-existing failures use sampling spies lacking the `generator` keyword that the unchanged HEAD decoder already passes; unrelated production code was preserved.
- Exact training config comparison and tokenizer hash audit pass.
- Hydra resolves the prepared training config successfully.
- Both PBS scripts and the shell launcher pass `bash -n`; `git diff --check` passes.

## Launchers

Training: `artifacts/robotwin_hopper/train_bar_btt_right_first_pali_30k.pbs`.

Evaluation: `artifacts/robotwin_hopper/eval_btt_right_first_step30000_handover_mic_100ep.pbs`.

Preparation/export helper: `scripts/tokenizer/prepare_btt_right_first_training.py`.

## 2026-10-04 cancellation and training-only resume

Original training 641958 and dependent evaluation 641959 were cancelled at the user's request. Training reached step 6406; the recoverable latest checkpoint contains step 6000, optimizer state, and EMA. No evaluation was resubmitted.

The initial suspicion that TorchCodec's missing FFmpeg libraries explained the slowdown was not supported by comparisons: successful jobs 622038, 616455, 630822, 634432, 633464, and 638303 also fell back to PyAV and completed 30000 steps in approximately 41–42 hours. Their monitored GPU averages were about 83–84%, compared with 12.8% for 641958 (86.8% zero-utilization samples). The original BTT run reached step 1000 in 1:07:40; 641958 took 9:34:43. Loading the same VLM checkpoint shards took under one second in 622038 and 2:19 in 641958, showing the slowdown existed before optimization. A node/file-access issue on hopper-27 is suspected, but not conclusively isolated by the available telemetry.

Both launchers used the same PyTorch 2.7/CUDA 12.8 container, 1 H200, 12 CPUs, 225 GB RAM, and 72-hour limit. A concrete PBS difference was that the right-first PBS omitted the pre-container TMPDIR export used in the successful BTT PBS; both still set TMPDIR inside the container. This difference alone has not been established as the cause. W&B 0.24.2 succeeded on the reference run; on 641958 its backend did not produce the service port file within 30 seconds, and Logger disabled W&B while retaining local metrics.

Training-only resume submitted as **645909.hopper-m-02**, job name `bttRLresume`, using `artifacts/robotwin_hopper/resume_bar_btt_right_first_pali_30k.pbs`. It follows the successful 622038 launcher including the outer TMPDIR export, pins to the previously successful hopper-11 node, uses the original container and dependencies unchanged, and resumes from the original step-6000 checkpoint into a new job-specific output directory. Configuration changes are exactly output_dir, run_mode, and resume_checkpoint. The goal remains 30000 steps with right_then_left arm order. No dependency installation or evaluation scheduling was performed. Shell syntax, checkpoint/arm-order checks, exact configuration parity, and Hydra resolution passed before submission. Runtime speed and W&B initialization still need verification once the job starts.
