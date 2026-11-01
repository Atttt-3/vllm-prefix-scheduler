#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run readiness inside an allocated Slurm compute job." >&2
  exit 2
fi
export PROJECT_ROOT="$TASK_ROOT"
export PATH="$TASK_ROOT/.venv-gpu/bin:$TASK_ROOT/.tools/bin:$PATH"
export PYTHONPATH="$TASK_ROOT/src:$TASK_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$TASK_ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$TASK_ROOT/.cache"
export TMPDIR="$TASK_ROOT/.tmp"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PROJECT_COMMIT="$(git -C "$TASK_ROOT" rev-parse HEAD)"
export PROJECT_DIRTY="$(git -C "$TASK_ROOT" status --porcelain --untracked-files=normal)"
TASK_RUN_ID="$(date -u '+%Y%m%dT%H%M%SZ')-${SLURM_JOB_ID}"
export RUN_DIR="${RUN_DIR:-$TASK_ROOT/runs/readiness/$TASK_RUN_ID}"
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing run directory: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR" "$TMPDIR"
printf '%s\n' "$RUN_DIR"
cp "$TASK_ROOT/configs/baseline-3090.json" "$TASK_ROOT/configs/upstream.json" "$RUN_DIR/"
if [[ -f "$TASK_ROOT/.local/environment.json" ]]; then
  cp "$TASK_ROOT/.local/environment.json" "$RUN_DIR/environment.json"
fi
"$TASK_ROOT/.venv-gpu/bin/python" -u "$TASK_ROOT/scripts/validate_model.py" \
  2>&1 | tee "$RUN_DIR/engine.log"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv \
  > "$RUN_DIR/gpu-after-process-exit.csv"
echo "Readiness complete: $RUN_DIR"
