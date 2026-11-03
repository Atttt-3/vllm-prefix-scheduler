"""Execute every frozen M4 native leaf; absence of native dependencies is failure."""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from m4_runtime import native_identity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests/native"))


def main():
    if not __debug__:
        raise RuntimeError("M4 native refuses optimized interpreter")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"task_id": config["task_id"], "layer": "native", "status": "failed", "cases": []}
    try:
        runtime = native_identity(config)
        (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "1":
            raise RuntimeError("M4 requires actual V2 native config")
        from m4_harness import run_case

        cases = config["native_matrix"]["leaves"]
        if not cases or len({json.dumps(case, sort_keys=True) for case in cases}) != len(cases):
            raise RuntimeError("Native frozen case matrix empty or duplicate")
        (args.output / "inputs.json").write_text(json.dumps(config, indent=2) + "\n")
        with (args.output / "trace.jsonl").open("x") as trace:

            def emit(row):
                trace.write(
                    json.dumps(
                        {
                            "job_id": runtime["job_id"],
                            "project_commit": runtime["project_commit"],
                            **row,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                trace.flush()

            for case in cases:
                result["cases"].append(run_case(config, case, args.model_config, emit))
        result["status"] = "succeeded"
        return 0
    except BaseException:
        (args.output / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
