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
                "execution_groups": json.loads((root / "configs/m4-stability.json").read_text())[
                    "execution_groups"
                ],
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


def test_process_exit_preserves_actual_emitter_envelope(tmp_path, monkeypatch):
    import os

    from vllm_prefix_scheduler import m3_events

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    from run_m4_suite import emit_process_exit

    monkeypatch.setenv("M3_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("M3_ENGINE_ID", "m4-test-stream")
    monkeypatch.setenv("M3_MODE", "guarded")
    monkeypatch.delenv("M3_CAMPAIGN_ID", raising=False)
    monkeypatch.delenv("M3_CAMPAIGN_SHA256", raising=False)
    child = os.getpid() + 100000
    m3_events.emit("m4_launcher_test", layer="gpu")
    emit_process_exit("M4-scheduler-stability", "calibration", {"pid": child, "exitcode": 0})
    rows = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    assert [row["pid"] for row in rows] == [os.getpid(), os.getpid()]
    assert rows[1]["child_pid"] == child
    assert rows[1]["seq"] > rows[0]["seq"]
    assert rows[1]["event"] == "m4_process_exit"


@pytest.mark.parametrize("group", ["metadata", "gpu"])
def test_fixed_groups_execute_and_seal_only_the_declared_pipeline(tmp_path, monkeypatch, group):
    import os

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "m4_group_test", root / "scripts/run_m4_allocation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tmp_path / "config.json"
    config.write_bytes((root / "configs/m4-stability.json").read_bytes())
    output = tmp_path / "allocation"
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module.os, "environ", dict(os.environ))
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        module.subprocess, "check_output", lambda *a, **k: "a" * 40 if k.get("text") else b""
    )
    monkeypatch.setattr(module, "gpu_snapshot", lambda: {"returncode": 0, "stdout": "LOCAL MOCK"})
    monkeypatch.setattr(module, "cleanup_engines", lambda *a, **k: [])
    monkeypatch.setattr(module, "owned_processes", lambda *a: [])
    launched = []

    class Child:
        pid = 123456
        returncode = 0

        def __init__(self, command, **kwargs):
            launched.append(command)

        def wait(self, **kwargs):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", Child)
    for name, value in {
        "SLURM_JOB_ID": "123",
        "NATIVE_PYTHON": sys.executable,
        "MODEL_CONFIG_DIR": str(tmp_path),
        "M4_CACHE": str(tmp_path),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_m4_allocation.py",
            "--config",
            str(config),
            "--output",
            str(output),
            "--group",
            group,
        ],
    )
    monkeypatch.chdir(tmp_path)
    assert module.main() == 0
    binding = json.loads((output / "binding.json").read_text())
    exits = json.loads((output / "exit.json").read_text())
    checkpoints = json.loads((output / "allocation-resource-checkpoints.json").read_text())
    wanted = json.loads(config.read_text())["execution_groups"][group]
    assert binding["execution_group"] == group
    assert binding["declared_steps"] == wanted
    assert [step["name"] for step in exits["steps"]] == wanted
    assert checkpoints["launcher_steps"] == exits["steps"]
    assert len(launched) == len(wanted)
    assert exits["returncode"] == 0
    assert exits["owned_pids_remaining"] == []
