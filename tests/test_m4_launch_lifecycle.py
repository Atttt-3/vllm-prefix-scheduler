"""Local launcher sealing checks; mocked hardware is never native/GPU evidence."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def test_launch_failure_preserves_allocation_exit(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "m4_allocation_test", root / "scripts/run_m4_allocation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "task_id": "M4-scheduler-stability",
                "environment": {},
                "timing": {"allocation_work_timeout_s": 1110},
            }
        )
    )
    output = tmp_path / "allocation"
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        module.subprocess, "check_output", lambda *a, **k: "a" * 40 if k.get("text") else b""
    )
    monkeypatch.setattr(module, "gpu_snapshot", lambda: {"returncode": 0, "stdout": "LOCAL MOCK"})
    monkeypatch.setattr(module, "cleanup_engines", lambda *a, **k: [])

    def fail_launch(*args, **kwargs):
        raise OSError("injected local launch failure")

    monkeypatch.setattr(module.subprocess, "Popen", fail_launch)
    for name, value in {
        "SLURM_JOB_ID": "123",
        "NATIVE_PYTHON": sys.executable,
        "MODEL_CONFIG_DIR": str(tmp_path),
        "M4_CACHE": str(tmp_path),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        sys, "argv", ["run_m4_allocation.py", "--config", str(config), "--output", str(output)]
    )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(OSError, match="injected local launch failure"):
        module.main()
    exit_record = json.loads((output / "exit.json").read_text())
    assert exit_record["returncode"] == 1
    assert exit_record["steps"][0]["returncode"] is None
    assert exit_record["steps"][0]["status"] == "launching"
    assert (output / "allocation-resource-checkpoints.json").exists()
