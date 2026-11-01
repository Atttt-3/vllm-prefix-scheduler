#!/usr/bin/env bash
# Run inside an authorized Linux NVIDIA Slurm allocation.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT="$TASK_ROOT"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run bootstrap_gpu.sh inside an allocated Slurm compute job." >&2
  exit 2
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "This lock targets Linux x86_64 NVIDIA nodes." >&2
  exit 2
fi
nvidia-smi --query-gpu=name,driver_version --format=csv
mkdir -p "$TASK_ROOT"/{.tools/bin,.tools/python,.cache/uv,.cache/huggingface,.tmp,.local}
export UV_PYTHON_INSTALL_DIR="$TASK_ROOT/.tools/python"
export UV_CACHE_DIR="$TASK_ROOT/.cache/uv"
export HF_HOME="$TASK_ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$TASK_ROOT/.cache"
export TMPDIR="$TASK_ROOT/.tmp"
export UV_LINK_MODE=copy
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
if [[ ! -x "$TASK_ROOT/.tools/bin/uv" ]]; then
  curl --fail --location --silent --show-error --retry 3 \
    https://astral.sh/uv/0.13.6/install.sh -o "$TASK_ROOT/.tmp/install-uv.sh"
  UV_INSTALL_DIR="$TASK_ROOT/.tools/bin" UV_NO_MODIFY_PATH=1 \
    sh "$TASK_ROOT/.tmp/install-uv.sh"
fi
TASK_UV="$TASK_ROOT/.tools/bin/uv"
"$TASK_UV" --version
if [[ ! -x "$TASK_ROOT/.venv-gpu/bin/python" ]]; then
  "$TASK_UV" venv "$TASK_ROOT/.venv-gpu" --python 3.12.16 --managed-python --seed
fi
"$TASK_ROOT/.venv-gpu/bin/python" -c \
  'import sys; assert sys.version_info[:3] == (3, 12, 16), "Existing .venv has a different Python version; keep it intact and choose another directory."'
export PATH="$TASK_ROOT/.venv-gpu/bin:$TASK_ROOT/.tools/bin:$PATH"
"$TASK_UV" pip sync --python "$TASK_ROOT/.venv-gpu/bin/python" \
  --torch-backend=cu129 \
  --extra-index-url https://flashinfer.ai/whl \
  --extra-index-url https://flashinfer.ai/whl/cu129 \
  "$TASK_ROOT/requirements/gpu-cu129.lock"
"$TASK_UV" pip check --python "$TASK_ROOT/.venv-gpu/bin/python"
"$TASK_UV" pip freeze --python "$TASK_ROOT/.venv-gpu/bin/python" \
  > "$TASK_ROOT/.local/environment-freeze.txt"
"$TASK_ROOT/.venv-gpu/bin/python" - <<'PY'
import json
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import torch

root = Path(os.environ["PROJECT_ROOT"])
record = {
    "python": sys.version,
    "python_executable": sys.executable,
    "hostname": platform.node(),
    "vllm": version("vllm"),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "bf16_supported": torch.cuda.is_bf16_supported(),
    "gpu_name": torch.cuda.get_device_name(0),
    "compute_capability": list(torch.cuda.get_device_capability(0)),
    "flashinfer_python": version("flashinfer-python"),
    "flashinfer_cubin": version("flashinfer-cubin"),
    "flashinfer_jit_cache": version("flashinfer-jit-cache"),
}
assert record["vllm"] == "0.32.0"
assert record["torch"] == "2.13.0+cu129"
assert record["cuda_available"] and record["bf16_supported"]
(root / ".local" / "environment.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PY
