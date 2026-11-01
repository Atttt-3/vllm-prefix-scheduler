#!/usr/bin/env bash
#SBATCH --partition=q-3090
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=prefix-readiness
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
# Slurm copies this file to its spool, so BASH_SOURCE is not the checkout path.
TASK_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the checkout root}}"
if [[ ! -f "$TASK_ROOT/configs/upstream.json" ]]; then
  echo "Submit this job from the repository root, or set PROJECT_ROOT explicitly." >&2
  exit 2
fi
exec bash "$TASK_ROOT/scripts/run_readiness.sh"
