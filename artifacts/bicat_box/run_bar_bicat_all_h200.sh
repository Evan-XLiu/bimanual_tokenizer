#!/usr/bin/env bash
set -Eeuo pipefail
cd "${PBS_O_WORKDIR:?}"
ROOT="$PWD"
OUT="$ROOT/outputs/BiCAT_all_BAR_BTT_Molmo2ER_H200/${PBS_JOBID:?}"
mkdir -p "$OUT" "$ROOT/.t/$PBS_JOBID"
exec > >(tee -a "$OUT/job.log") 2>&1
export BICAT_DATA_ROOT="$ROOT/data/BiCAT-BimanualYAM-Dataset"
export HF_HOME="$ROOT/.cache/huggingface"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TMPDIR="$ROOT/.t/$PBS_JOBID"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi
PY="$ROOT/.venv/bin/python"
for BS in 128 64; do
  PROBE="$OUT/probe_bs$BS"
  mkdir -p "$PROBE"
  "$PY" -m praxis.scripts.train --config-name train_bicat_all_bar_btt_molmo_h200 trainer.batch_size="$BS" --cfg job --resolve > "$PROBE/resolved_config.yaml"
  if "$PY" artifacts/bicat_box/test_bar_h200_memory.py "$PROBE/resolved_config.yaml" "$PROBE" > "$PROBE/probe.log" 2>&1; then
    echo "Batch size $BS passed three real optimizer steps. Starting full training."
    echo "$BS" > "$OUT/selected_batch_size.txt"
    break
  fi
  STATUS=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$PROBE/memory_result.json")
  if [[ "$STATUS" != out_of_memory ]]; then
    cat "$PROBE/probe.log"
    exit 1
  fi
  echo "Batch size $BS exceeded GPU memory."
  if [[ "$BS" == 64 ]]; then exit 1; fi
done
"$PY" -m praxis.scripts.train --config-name train_bicat_all_bar_btt_molmo_h200 trainer.batch_size="$BS" trainer.global_batch_size=512 output_dir="$OUT/train" --cfg job --resolve > "$OUT/resolved_config.yaml"
exec "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 --module praxis.scripts.train --config-name train_bicat_all_bar_btt_molmo_h200 trainer.batch_size="$BS" trainer.global_batch_size=512 output_dir="$OUT/train"
