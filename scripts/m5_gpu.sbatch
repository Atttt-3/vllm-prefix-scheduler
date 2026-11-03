#!/usr/bin/env bash
#SBATCH --job-name=m5-paired
#SBATCH --partition=q-3090
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --signal=B:TERM@45
set -euo pipefail
if [[ $# != 10 ]]; then
    echo 'M5 allocation requires a bound campaign, authorization, phase and role' >&2
    exit 2
fi
native_python=$1
shift
exec "$native_python" -u scripts/run_m5_allocation.py "$@"
