#!/usr/bin/env bash
# Unchanged baseline entry plus full M2B fault/closed contract at the M4 freeze.
set -uo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${SLURM_JOB_ID:?Existing owned allocation required}"
: "${NATIVE_PYTHON:?Existing pinned interpreter required}"
: "${MODEL_CONFIG_DIR:?Existing fixed model config required}"
: "${M3_REGRESSION_RUN:?New M4-owned regression directory required}"
cd "$TASK_ROOT" || exit 2
bash scripts/run_m3_regressions.sh
M4_BASE_RC=$?
export PYTHONPATH="$TASK_ROOT/src:$TASK_ROOT/scripts"
export PYTHONHASHSEED=42 VLLM_USE_V2_MODEL_RUNNER=1 HF_HUB_OFFLINE=1
M4_FULL_DATA="$M3_REGRESSION_RUN/full-data"
"$NATIVE_PYTHON" scripts/run_evidence.py --task-id M4-M2B-full-contract --kind native \
  --output "$M3_REGRESSION_RUN/full" --timeout 240 --seed 42 \
  --config configs/native-m2b.json --config configs/m4-stability.json \
  --artifact "$M4_FULL_DATA/runtime.json" --artifact "$M4_FULL_DATA/inputs.json" \
  --artifact "$M4_FULL_DATA/clock-plans.json" --artifact "$M4_FULL_DATA/case-index.json" \
  --artifact "$M4_FULL_DATA/trace.jsonl" --artifact "$M4_FULL_DATA/summary.json" \
  --artifact "$M4_FULL_DATA/trace-acceptance.json" \
  --criterion '636 unchanged M2B native leaves including active-object fault/closed lifecycle at this M4 freeze' \
  -- "$NATIVE_PYTHON" scripts/replay_m2b.py --suite full --config configs/native-m2b.json \
  --model-config "$MODEL_CONFIG_DIR" --output "$M4_FULL_DATA"
M4_FULL_RC=$?
printf '{"compat_and_m1":%s,"full636":%s}\n' "$M4_BASE_RC" "$M4_FULL_RC" > "$M3_REGRESSION_RUN/m4-exit-codes.json"
if (( M4_BASE_RC != 0 || M4_FULL_RC != 0 )); then exit 1; fi
