"""Execute the complete pinned M2B native matrix; unavailable native is failure."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "native"))

from vllm_prefix_scheduler.m2b_contract import (  # noqa: E402
    expected_cases,
    require_assertions,
    verify_result_cases,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    require_assertions()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/native-m2b.json")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("full", "compat"), default="full")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    cases = expected_cases(config, suite=args.suite)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        write_json(args.output / "inputs.json", {"suite": args.suite, "matrix": config})
        write_json(args.output / "case-index.json", cases)
        write_json(args.output / "clock-plans.json", config["clock_plans"])
        for key, value in {
            "PYTHONHASHSEED": str(config["seed"]),
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "HF_HUB_OFFLINE": "1",
        }.items():
            if os.environ.get(key) != value:
                raise RuntimeError(f"Set {key}={value} before launching the interpreter")
        if sys.platform != "linux":
            raise RuntimeError("M2B native validation requires the fixed Linux environment")
        import vllm

        native_root = Path(vllm.__file__).resolve().parent.parent
        modules = {
            relative: {
                "path": str(native_root / relative),
                "sha256": fingerprint(native_root / relative),
                "expected_sha256": expected,
            }
            for relative, expected in config["source_sha256"].items()
        }
        runtime = {
            "suite": args.suite,
            "python_executable": sys.executable,
            "python_version": sys.version,
            "python_optimize": sys.flags.optimize,
            "platform": platform.platform(),
            "vllm_version": vllm.__version__,
            "upstream_commit": config["vllm_commit"],
            "project_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "modules": modules,
            "custom_sources": {
                name: fingerprint(ROOT / "src/vllm_prefix_scheduler" / name)
                for name in (
                    "config.py",
                    "scheduler.py",
                    "probe.py",
                    "wait_guard.py",
                    "m2b_contract.py",
                )
            },
            "validation_sources": {
                name: fingerprint(ROOT / name)
                for name in (
                    "tests/native/m2b_harness.py",
                    "tests/native/m2b_snapshot.py",
                    "tests/native/m2b_fault_cases.py",
                    "tests/native/m2b_trace_check.py",
                    "tests/native/m2a_harness.py",
                    "tests/native/m2a_snapshot.py",
                    "tests/native/m2a_trace_check.py",
                    "scripts/replay_m2b.py",
                )
            },
            "model_config_sha256": fingerprint(args.model_config / "config.json"),
            "environment": {
                key: os.environ.get(key)
                for key in (
                    "PYTHONHASHSEED",
                    "VLLM_USE_V2_MODEL_RUNNER",
                    "HF_HUB_OFFLINE",
                    "SLURM_JOB_ID",
                    "CUDA_VISIBLE_DEVICES",
                )
            },
            "substitutions": [
                "Synthetic ModelRunnerOutput; no model weights, tokenizer or KV tensors",
                "One-layer FullAttentionSpec metadata geometry",
                "Controlled private protection clock; native clocks remain unchanged",
            ],
            "core_native_replacements": [],
            "test_injections": [
                "Named window branches set native request fields and inject queue/probe errors",
                "Named fault leaves raise at native add/free/schedule/preemption/shutdown boundaries",
                "Postcheck leaves mutate one actual constructed object before the real support check",
                "Missing-record leaves explicitly remove one protection record before scheduling",
                "Fault teardown may retry freeing the same current terminal object or explicitly detach an unbound failed admission",
            ],
            "injection_scope": "Natural hot-flow, allocation-retry, chunk and preemption leaves retain native allocation, output update and lifecycle behavior",
        }
        write_json(args.output / "runtime.json", runtime)
        if runtime["vllm_version"] != config["vllm_version"]:
            raise RuntimeError("Pinned vLLM version mismatch")
        if runtime["model_config_sha256"] != config["model_config_sha256"]:
            raise RuntimeError("Pinned model metadata mismatch")
        if any(v["sha256"] != v["expected_sha256"] for v in modules.values()):
            raise RuntimeError("Pinned native module fingerprint mismatch")
        from m2b_harness import run_matrix

        results = run_matrix(config, args.model_config, args.output, suite=args.suite)
        verify_result_cases(results, config, suite=args.suite)
        from m2b_trace_check import assert_trace

        checked = assert_trace(args.output / "trace.jsonl", config, suite=args.suite)
        write_json(args.output / "trace-acceptance.json", checked)
        write_json(
            args.output / "summary.json",
            {"suite": args.suite, "metric": config["metric"], "cases": results, "verified": True},
        )
        print(
            json.dumps({"suite": args.suite, "native_cases": len(cases), "matrix_verified": True})
        )
    except BaseException:
        (args.output / "failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
