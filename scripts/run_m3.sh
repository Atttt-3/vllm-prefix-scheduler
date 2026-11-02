#!/usr/bin/env bash
# Reuse an existing allocation/environment. Machine locations are provided privately.
set -uo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${SLURM_JOB_ID:?M3 requires an existing Slurm allocation}"
: "${NATIVE_PYTHON:?Set the existing pinned Linux interpreter}"
: "${MODEL_CONFIG_DIR:?Set the existing fixed model directory}"
: "${M3_RUN:?Set a new private run directory}"
: "${M3_CACHE:?Set the existing cache directory}"
cd "$TASK_ROOT" || exit 2
export PYTHONPATH="$TASK_ROOT/src:$TASK_ROOT/scripts"
export PYTHONHASHSEED=42 VLLM_USE_V2_MODEL_RUNNER=1 VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export HF_HOME="$M3_CACHE/huggingface" XDG_CACHE_HOME="$M3_CACHE"
export TMPDIR="$TASK_ROOT/.tmp"
unset PYTHONOPTIMIZE
mkdir -p "$TMPDIR"
M3_DATA="${M3_RUN}-data"
M3_INPUTS="${M3_RUN}-inputs.json"
"$NATIVE_PYTHON" scripts/prepare_m3_inputs.py --config configs/m3-real-model.json \
  --model "$MODEL_CONFIG_DIR" --output "$M3_INPUTS" || exit $?
"$NATIVE_PYTHON" scripts/run_evidence.py --task-id M3-real-model-integration --kind gpu \
  --output "$M3_RUN" --timeout 1080 --seed 42 --config configs/m3-real-model.json \
  --config "$M3_INPUTS" --config configs/upstream.json --config configs/baseline-3090.json \
  --artifact "$M3_DATA/sources.json" --artifact "$M3_DATA/trace.jsonl" \
  --artifact "$M3_DATA/results.json" --artifact "$M3_DATA/resource-checkpoints.json" \
  --artifact "$M3_DATA/raw-artifacts.tar.gz" \
  --criterion 'A fixed sources production smoke and actual diagnostic instances' \
  --criterion 'B real runner output and native async consumption closure' \
  --criterion 'C real prefix reorder and exact 500ms protection three repeats' \
  --criterion 'D cancellation drain shutdown process and allocation release' \
  --criterion 'E native off comparison and honest dispatch measurement' \
  --criterion 'F independent complete raw trace review and fail closed controls' \
  -- "$NATIVE_PYTHON" -u scripts/run_m3_suite.py --model "$MODEL_CONFIG_DIR" --output "$M3_DATA" --inputs "$M3_INPUTS" "$@"
M3_RC=$?
# Always attempt final resource capture even when collection failed.
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv > "${M3_RUN}-gpu-after.csv"
printf '{"collection_exitcode":%s,"gpu_query_exitcode":%s}\n' "$M3_RC" "$?" > "${M3_RUN}-exit.json"
exit "$M3_RC"
