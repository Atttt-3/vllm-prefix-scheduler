#!/usr/bin/env bash
#SBATCH --job-name=m3-real-model
#SBATCH --partition=q-3090
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
# Submit only after setting private NATIVE_PYTHON/MODEL_CONFIG_DIR/M3_CACHE/M3_RUN.
set -uo pipefail
bash scripts/run_m3.sh "$@"
