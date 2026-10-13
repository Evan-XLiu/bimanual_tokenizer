#!/usr/bin/env bash
#SBATCH --job-name=BiCAT-WandB-Mirror
#SBATCH --partition=debug
#SBATCH --nodelist=crane6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
#SBATCH --output=/data/home/shailesh/Evan/bimanual_tokenizer/outputs/wandb-mirror-%j.out

set -Eeuo pipefail
cd /data/home/shailesh/Evan/bimanual_tokenizer

exec "$PWD/.venv/bin/python" scripts/infra/mirror_train_metrics_to_wandb.py \
  --metrics "$PWD/outputs/BiCAT_box_Molmo2ER_FM_30k/67256/train/train_metrics.jsonl" \
  --key-file "$HOME/.config/bimanual_tokenizer/wandb_evan_lxw.key" \
  --expected-username evan_lxw \
  --entity evan_lxw-harbin-institute-of-technology \
  --project BiCAT_RealWorld \
  --run-id fm67256e \
  --run-name BiCAT_box_Molmo2ER_FM_67256 \
  --source-run 'https://wandb.ai/shaileshxml-nus/BiCAT_RealWorld/runs/60rm7wio' \
  --follow \
  --final-step 30000
