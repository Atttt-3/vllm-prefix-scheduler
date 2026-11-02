"""Opt-in full matrix wrapper; native CLI never treats missing dependencies as skip."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.native
def test_complete_m2a_native_matrix(tmp_path):
    model = os.environ.get("M2A_NATIVE_MODEL_CONFIG")
    if model is None:
        pytest.skip("Set M2A_NATIVE_MODEL_CONFIG on the fixed Linux runtime; not native acceptance")
    env = os.environ.copy()
    env.update(PYTHONHASHSEED="42", VLLM_USE_V2_MODEL_RUNNER="1", HF_HUB_OFFLINE="1")
    result = subprocess.run(
        [
            os.environ.get("M2A_NATIVE_PYTHON", sys.executable),
            str(ROOT / "scripts/replay_m2a.py"),
            "--config",
            str(ROOT / "configs/native-m2a.json"),
            "--model-config",
            model,
            "--output",
            str(tmp_path / "native"),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"matrix_verified": true' in result.stdout
