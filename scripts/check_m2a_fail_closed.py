"""Run real Linux subprocess negatives without changing the shared environment."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("This entry verifies missing native dependencies on Linux")
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / "configs/native-m2a.json").read_text())
    empty = args.output / "empty.jsonl"
    empty.write_text("")
    replay = ROOT / "scripts/replay_m2a.py"
    audit = ROOT / "tests/native/m2a_trace_check.py"
    env = os.environ.copy()
    env.pop("PYTHONOPTIMIZE", None)
    commands = [
        (
            "missing_native",
            [
                sys.executable,
                "-S",
                str(replay),
                "--model-config",
                str(args.output),
                "--output",
                str(args.output / "missing-native-data"),
            ],
            "No module named 'vllm'",
        ),
        ("empty_trace", [sys.executable, str(audit), "--trace", str(empty)], "nonempty"),
        ("optimized_replay", [sys.executable, "-O", str(replay), "--help"], "assertions enabled"),
        ("optimized_audit", [sys.executable, "-OO", str(audit), "--help"], "assertions enabled"),
    ]
    for label, mutation in (
        ("empty_matrix", lambda c: c.update(scenarios={})),
        ("insufficient_repeats", lambda c: c.update(repeats=2)),
        ("missing_scenario", lambda c: c["scenarios"].pop("preemption")),
    ):
        changed = json.loads(json.dumps(config))
        mutation(changed)
        path = args.output / (label + ".json")
        path.write_text(json.dumps(changed))
        commands.append(
            (
                label,
                [
                    sys.executable,
                    str(replay),
                    "--config",
                    str(path),
                    "--model-config",
                    str(args.output),
                    "--output",
                    str(args.output / (label + "-data")),
                ],
                "M2A:",
            )
        )
    results = []
    for label, command, expected in commands:
        result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=30)
        (args.output / (label + ".stdout")).write_text(result.stdout)
        (args.output / (label + ".stderr")).write_text(result.stderr)
        passed = result.returncode != 0 and expected in result.stderr
        results.append(
            {
                "case": label,
                "argv": command,
                "returncode": result.returncode,
                "expected_error": expected,
                "passed": passed,
                "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
            }
        )
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        if not passed:
            raise RuntimeError(
                f"Fail-closed subprocess {label} did not fail for the expected reason"
            )
    print(json.dumps({"fail_closed_cases": len(results), "all_passed": True}))


if __name__ == "__main__":
    main()
