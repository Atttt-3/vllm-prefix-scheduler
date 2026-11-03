"""Check one D9 dual-window trace or independently replay its allocation."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from check_m5_d2 import review
from m5_d3_audit import timing_leaf, verify_native_sources
from m5_d6_audit import verify_step_sources
from m5_d9_audit import audit_d9_trace
from m5_seal import ROOT

from vllm_prefix_scheduler.m5_campaign import D9_ROLE, digest, read_json, require
from vllm_prefix_scheduler.m5_contract import client_requests
from vllm_prefix_scheduler.m5_d9 import check_d9_profile, d9_plan


def check_leaf(raw, config, proposal, leaf_end_ns):
    plan = d9_plan(proposal, ROOT / "configs/m5-d8.json", ROOT / "configs/m5-d2r110.json")
    require(digest(config) == plan["runtime_config_sha256"], "D9 runtime config differs")
    table = raw.parent / "input.json"
    require(digest(table) == plan["target"]["input_sha256"], "D9 leaf input differs")
    client, engine, source, trace = (
        read_json(raw / name)
        for name in ("client.json", "engine.json", "sources.json", "d9-step.json")
    )
    verify_native_sources(plan, source)
    verify_step_sources(plan, source)
    require(client["tier"] == engine["tier"] == plan["target"]["tier"], "D9 leaf tier differs")
    check_d9_profile(engine["profile"], plan)
    inputs = read_json(table)
    gates = read_json(config)["gates"]
    expected = {row["request_id"]: row for row in inputs["requests"]}
    _, submit_lag = client_requests(client, inputs, expected, gates, "calibration")
    timing = timing_leaf(client, engine, inputs, gates, leaf_end_ns=leaf_end_ns)
    require(
        timing["clock"]["verified"] and timing["status"] == "complete",
        f"D9 original native clock or timing is unverified: {timing['clock']['issues']}",
    )
    audited = audit_d9_trace(trace, client, plan["trace"], leaf_end_ns=leaf_end_ns)
    require(audited["verified"], "D9 trace is incomplete")
    return {
        "clock": timing["clock"],
        "submit_lag_ns": submit_lag,
        "trace": {key: value for key, value in audited.items() if key != "windows"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--attempt", type=Path)
    mode.add_argument("--raw", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--leaf-end-ns", type=int)
    args = parser.parse_args()
    try:
        require(not sys.flags.optimize, "run unoptimized D9 auditor")
        if args.attempt:
            require(
                args.output and not args.config and args.leaf_end_ns is None,
                "D9 attempt requires only --output",
            )
            require(read_json(args.proposal)["role"] == D9_ROLE, "D9 proposal role differs")
            report = review(args.attempt.resolve(), args.proposal.resolve())
            with args.output.open("x") as stream:
                json.dump(report, stream, indent=2, default=str, allow_nan=False)
                stream.write("\n")
        else:
            require(
                args.config and args.leaf_end_ns is not None and not args.output,
                "D9 leaf requires --config and --leaf-end-ns",
            )
            print(
                json.dumps(
                    check_leaf(args.raw, args.config, args.proposal, args.leaf_end_ns),
                    allow_nan=False,
                )
            )
    except (KeyError, OSError, TypeError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"M5 D9 evidence rejected: {error}\n")


if __name__ == "__main__":
    main()
