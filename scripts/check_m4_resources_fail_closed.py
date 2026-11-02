"""Real subprocess M4 resource controls; preserve all original allocation records."""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_prefix_scheduler.m4_contract import require, require_assertions, sha256  # noqa: E402


def main():
    require_assertions()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    original = json.loads(args.resources.read_text())
    original_hash = sha256(args.resources)
    cases = {
        "positive": None,
        "missing_terminal": "terminal/checkpoint",
        "omitted_job": "budget",
        "wrong_exit": "declared terminal",
        "active_job": "remains in final queue",
        "watcher_missing": "watcher exit",
        "watcher_started_late": "ready before submission",
        "omitted_attempt": "submission attempt ledger",
        "optimized": "assertions enabled",
    }
    results = []
    for name, error in cases.items():
        directory = args.output / name
        directory.mkdir()
        resource = copy.deepcopy(original)

        def write(filename, value, directory=directory):
            path = directory / filename
            with path.open("x") as stream:
                stream.write(json.dumps(value, indent=2) + "\n")
            return {"path": str(path.resolve()), "sha256": sha256(path)}

        if name == "missing_terminal":
            resource["allocations"][0]["evidence"].pop("terminal")
        elif name == "omitted_job":
            resource["allocations"] = []
        elif name == "wrong_exit":
            resource["allocations"][0]["exit_code"] = "99:0"
        elif name == "active_job":
            raw = dict(resource["active_query"])
            raw["stdout"] += str(resource["allocations"][0]["job_id"]) + "|RUNNING|m4\n"
            resource["active_query"] = raw
            resource["active_query_evidence"] = write("active-derived.json", raw)
        elif name in {"watcher_missing", "watcher_started_late", "omitted_attempt"}:
            ledger = json.loads(Path(resource["submission_ledger"]["path"]).read_text())
            if name == "watcher_missing":
                ledger["attempts"][0].pop("monitor_exit")
            elif name == "watcher_started_late":
                attempt = ledger["attempts"][0]
                ready = json.loads(Path(attempt["monitor_ready"]["path"]).read_text())
                receipt = json.loads(Path(attempt["submission"]["path"]).read_text())
                ready["observed_monotonic_ns"] = receipt["started_monotonic_ns"] + 1
                attempt["monitor_ready"] = write("ready-derived.json", ready)
            else:
                ledger["attempts"] = []
            resource["submission_ledger"] = write("ledger-derived.json", ledger)
        path = Path(write("resources-derived.json", resource)["path"])
        command = [
            sys.executable,
            *(["-O"] if name == "optimized" else []),
            str(ROOT / "scripts/check_m4_resources.py"),
            "--config",
            str(args.config.resolve()),
            "--resources",
            str(path),
            "--output",
            str(directory / "audit.json"),
        ]
        child = subprocess.run(command, text=True, capture_output=True, timeout=60)
        (directory / "stdout.log").write_text(child.stdout)
        (directory / "stderr.log").write_text(child.stderr)
        result = {
            "operation": name,
            "command": command,
            "returncode": child.returncode,
            "expected_error": error,
            "derived_inputs": [
                {"path": str(path), "sha256": sha256(path)}
                for path in sorted(directory.glob("*-derived.json"))
            ],
            "working_directory": str(ROOT),
            "success": child.returncode == 0
            if error is None
            else child.returncode != 0 and error in child.stderr,
        }
        results.append(result)
        (args.output / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
        require(result["success"], f"resource subprocess control failed: {name}")
    require(sha256(args.resources) == original_hash, "resource control changed original index")
    (args.output / "results.json").write_text(
        json.dumps(
            {
                "scope": "resource_contract_controls",
                "original_sha256": original_hash,
                "controls": results,
                "all_controls_passed": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"controls": len(results), "all_controls_passed": True}))


if __name__ == "__main__":
    main()
