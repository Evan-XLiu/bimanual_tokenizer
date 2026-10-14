#!/usr/bin/env bash
#SBATCH --job-name=BiCAT-FM-box-act
#SBATCH --partition=debug
#SBATCH --nodelist=crane6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/data/home/shailesh/Evan/bimanual_tokenizer/outputs/fm-box-act-%j.out

set -Eeuo pipefail
cd /data/home/shailesh/Evan/bimanual_tokenizer

CKPT="${CKPT:-$PWD/outputs/BiCAT_box_Molmo2ER_FM_30k/67256/train/checkpoints/recent/step_3000/pretrained_model}"
PORT="${PORT:-8213}"
if [[ ! -f "$CKPT/model.safetensors" || ! -f "$CKPT/policy_io.json" || ! -f "$CKPT/norm_stats.pt" ]]; then
  echo "Incomplete FM checkpoint: $CKPT" >&2
  exit 2
fi

export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

echo "FM HTTP /act checkpoint=$CKPT port=$PORT node=$(hostname -s)"
exec "$PWD/.venv/bin/python" scripts/serve_fm_box_http.py \
  --checkpoint "$CKPT" --host 0.0.0.0 --port "$PORT" --device cuda:0
