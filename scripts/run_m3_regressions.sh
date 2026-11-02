#!/usr/bin/env bash
# Frozen M3 regression on unchanged production strategy; reuse the existing runtime.
set -uo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${SLURM_JOB_ID:?Run inside the owned allocation}"
: "${NATIVE_PYTHON:?Existing pinned native interpreter}"
: "${MODEL_CONFIG_DIR:?Existing model config directory}"
: "${M3_REGRESSION_RUN:?New private regression directory}"
cd "$TASK_ROOT" || exit 2
export PYTHONPATH="$TASK_ROOT/src:$TASK_ROOT/scripts"
export PYTHONHASHSEED=42 VLLM_USE_V2_MODEL_RUNNER=1 HF_HUB_OFFLINE=1
unset PYTHONOPTIMIZE
mkdir -p "$M3_REGRESSION_RUN"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv > "$M3_REGRESSION_RUN/gpu-before.csv"
M3_COMPAT="$M3_REGRESSION_RUN/compat-data"
"$NATIVE_PYTHON" scripts/run_evidence.py --task-id M3-M2B-compat --kind native \
  --output "$M3_REGRESSION_RUN/compat" --timeout 600 --seed 42 --config configs/native-m2b.json \
  --artifact "$M3_COMPAT/runtime.json" --artifact "$M3_COMPAT/inputs.json" \
  --artifact "$M3_COMPAT/clock-plans.json" --artifact "$M3_COMPAT/case-index.json" \
  --artifact "$M3_COMPAT/trace.jsonl" --artifact "$M3_COMPAT/summary.json" \
  --artifact "$M3_COMPAT/trace-acceptance.json" \
  --criterion '432 current M2B compatibility cases and independent raw trace checks at the M3 freeze' \
  -- make native-m2b-compat NATIVE_PYTHON="$NATIVE_PYTHON" \
  ARGS="--config configs/native-m2b.json --model-config '$MODEL_CONFIG_DIR' --output '$M3_COMPAT'"
M3_COMPAT_RC=$?
M3_M1="$M3_REGRESSION_RUN/m1-data"
"$NATIVE_PYTHON" scripts/run_evidence.py --task-id M3-M1-regression --kind native \
  --output "$M3_REGRESSION_RUN/m1" --timeout 180 --seed 42 --config configs/native-bc-replay.json \
  --artifact "$M3_M1/runtime.json" --artifact "$M3_M1/inputs.json" \
  --artifact "$M3_M1/trace.jsonl" --artifact "$M3_M1/summary.json" \
  --criterion '12 original M1 native replay cases at the M3 freeze' \
  -- make native-replay NATIVE_PYTHON="$NATIVE_PYTHON" \
  ARGS="--config configs/native-bc-replay.json --model-config '$MODEL_CONFIG_DIR' --output '$M3_M1'"
M3_M1_RC=$?
make native-trace-check NATIVE_PYTHON="$NATIVE_PYTHON" \
  ARGS="--config configs/native-bc-replay.json --trace '$M3_M1/trace.jsonl'" \
  > "$M3_REGRESSION_RUN/m1-trace-check.log" 2>&1
M3_M1_TRACE_RC=$?
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv > "$M3_REGRESSION_RUN/gpu-after.csv"
M3_GPU_RC=$?
printf '{"compat":%s,"m1":%s,"m1_trace":%s,"gpu_after_query":%s}\n' \
  "$M3_COMPAT_RC" "$M3_M1_RC" "$M3_M1_TRACE_RC" "$M3_GPU_RC" > "$M3_REGRESSION_RUN/exit-codes.json"
if (( M3_COMPAT_RC != 0 || M3_M1_RC != 0 || M3_M1_TRACE_RC != 0 || M3_GPU_RC != 0 )); then exit 1; fi
