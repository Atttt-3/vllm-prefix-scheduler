"""Check one D6 raw host trace and its cross-process clock evidence."""

import argparse
import json
from pathlib import Path

from m5_d3_audit import native_clock
from m5_d6_audit import audit_step_trace

from vllm_prefix_scheduler.m5_campaign import digest, read_json, require
from vllm_prefix_scheduler.m5_d6 import D6_PLAN_SHA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "config", "proposal"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--leaf-end-ns", type=int, required=True)
    args = parser.parse_args()
    try:
        require(digest(args.proposal) == D6_PLAN_SHA, "D6 reviewed plan differs")
        plan = read_json(args.proposal)
        client, engine, trace = (
            read_json(args.raw / name) for name in ("client.json", "engine.json", "d6-step.json")
        )
        clock = native_clock(client, engine, read_json(args.config)["gates"], args.leaf_end_ns)
        require(clock["verified"], f"D6 clock check failed: {', '.join(clock['issues'])}")
        report = audit_step_trace(trace, client, plan["trace"], leaf_end_ns=args.leaf_end_ns)
        require(report["verified"], "D6 trace is incomplete")
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"D6 trace rejected: {error}\n")
    print(json.dumps({"clock": clock, "trace": report}, allow_nan=False))


if __name__ == "__main__":
    main()
