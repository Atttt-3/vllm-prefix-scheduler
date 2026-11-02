"""M4 fixed-source identity helpers; no model or scheduler construction here."""

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def native_identity(config):
    import json

    if not __debug__ or sys.flags.optimize:
        raise RuntimeError("M4 prohibits optimized interpreters")
    if sys.platform != "linux" or platform.python_version() != "3.12.16":
        raise RuntimeError("M4 requires the fixed Linux Python 3.12.16 runtime")
    if not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise RuntimeError("M4 native validation requires an existing owned allocation")
    versions = {name: importlib.metadata.version(name) for name in ("vllm", "torch")}
    if versions != {"vllm": "0.32.0", "torch": "2.13.0+cu129"}:
        raise RuntimeError("M4 fixed native package mismatch")
    distribution = importlib.metadata.distribution("vllm")
    expected = json.loads((ROOT / "configs/native-m2b.json").read_text())["source_sha256"]
    expected.update(config["extra_upstream_sha256"])
    modules = {}
    for name, wanted in expected.items():
        path = Path(distribution.locate_file(name))
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != wanted:
            raise RuntimeError(f"M4 fixed native source mismatch: {name}")
        modules[name] = {"path": str(path.resolve()), "sha256": actual}
    if subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=ROOT):
        raise RuntimeError("M4 requires a clean frozen checkout")
    files = subprocess.check_output(
        ["git", "ls-files", "src", "scripts", "tests", "configs", "Makefile"], cwd=ROOT, text=True
    ).splitlines()
    return {
        "task_id": config["task_id"],
        "project_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "project_dirty": False,
        "python_version": platform.python_version(),
        "python_optimize": sys.flags.optimize,
        "packages": versions,
        "upstream_commit": config["upstream_commit"],
        "upstream_modules": modules,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files
        },
        "job_id": os.environ["SLURM_JOB_ID"],
        "substitutions": [
            "native fixtures use synthetic ModelRunnerOutput and one-layer metadata; no model or real KV tensors"
        ],
    }
