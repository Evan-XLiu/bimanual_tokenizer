#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_ROOT"
RUN_ID=${PBS_JOBID:?}
PREVIOUS_OUT="$SOURCE_ROOT/outputs/hopper_bar_btt_right_first_pali_robotwin_new_30k/641958.hopper-m-02"
RESUME_CHECKPOINT="$PREVIOUS_OUT/train/checkpoints/latest"
OUT="$SOURCE_ROOT/outputs/hopper_bar_btt_right_first_pali_robotwin_new_30k/$RUN_ID"
JOB_TMPDIR=${JOB_TMPDIR:-$SOURCE_ROOT/.t/$RUN_ID}
test -f "$RESUME_CHECKPOINT/training_state/training_step.json"
test ! -e "$OUT/resolved_config.yaml"
mkdir -p "$OUT/train" "$JOB_TMPDIR"
exec > >(tee "$OUT/train.log") 2>&1

export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export HF_HOME=${HF_HOME:-$SOURCE_ROOT/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export WANDB_MODE=${WANDB_MODE:-online}
export TMPDIR="$JOB_TMPDIR"
export PYTHONUNBUFFERED=1

"$SOURCE_ROOT/.venv/bin/python" - "$PREVIOUS_OUT" "$OUT" "$RESUME_CHECKPOINT" <<'PY'
import json
import sys
from pathlib import Path
from omegaconf import OmegaConf

previous, out, checkpoint = map(Path, sys.argv[1:])
assert json.loads((checkpoint / 'training_state/training_step.json').read_text())['step'] == 6000
model_config = json.loads((checkpoint / 'pretrained_model/config.json').read_text())
assert model_config['btt_arm_order'] == 'right_then_left'
cfg = OmegaConf.load(previous / 'resolved_config.yaml')
assert cfg.policy.btt_arm_order == 'right_then_left'
assert cfg.trainer.max_steps == 30000 and cfg.trainer.batch_size == 64
assert cfg.trainer.use_wandb and cfg.trainer.wandb_project == 'praxis-vla'
cfg.output_dir = str(out / 'train')
cfg.run_mode = 'resume'
cfg.resume_checkpoint = str(checkpoint)
OmegaConf.save(cfg, out / 'resolved_config.yaml')
OmegaConf.save(cfg, out / 'train/train_config.yaml')
(out / 'resume_provenance.json').write_text(json.dumps({
    'previous_job': '641958.hopper-m-02',
    'resume_checkpoint': str(checkpoint), 'resume_step': 6000,
    'target_step': 30000, 'launcher_reference_job': '622038.hopper-m-02',
    'changed_config_fields': ['output_dir', 'run_mode', 'resume_checkpoint'],
    'evaluation_scheduled': False,
}, indent=2) + '\n')
print(f'CONFIG_AUDIT_OK right_then_left resume_step=6000 target_step=30000 checkpoint={checkpoint}')
PY

"$SOURCE_ROOT/.venv/bin/praxis-train" --config-path "$OUT" --config-name resolved_config --cfg job > "$OUT/hydra_config.yaml"
echo "TRAIN_START job=$RUN_ID host=$(hostname) resume_checkpoint=$RESUME_CHECKPOINT output=$OUT"
nvidia-smi
(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits
    sleep 10
  done
) > "$OUT/gpu_usage.csv" &
GPU_MONITOR_PID=$!
cleanup() {
  kill "$GPU_MONITOR_PID" 2>/dev/null || true
  wait "$GPU_MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"$SOURCE_ROOT/.venv/bin/praxis-train" --config-path "$OUT" --config-name resolved_config
"$SOURCE_ROOT/.venv/bin/python" scripts/tokenizer/prepare_btt_right_first_training.py --output-dir "$OUT" --finalize
echo "BTT_RIGHT_FIRST_TRAIN_COMPLETE output=$OUT"
