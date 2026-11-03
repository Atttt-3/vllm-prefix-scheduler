"""Check one D10 measured leaf or independently replay its sealed allocation."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from check_m5_d2 import review
from m5_d3_audit import timing_leaf, verify_native_sources
from m5_d10_audit import ENDPOINTS, observed_placement
from m5_seal import ROOT

from vllm_prefix_scheduler.m5_campaign import D10_ROLE, digest, read_json, require
from vllm_prefix_scheduler.m5_contract import client_requests
from vllm_prefix_scheduler.m5_d9 import check_d9_profile
from vllm_prefix_scheduler.m5_d10 import d10_plan


def check_leaf(raw, config, proposal, leaf_end_ns, identity):
    plan = d10_plan(proposal, ROOT / "configs/m5-d8.json", ROOT / "configs/m5-d2r110.json")
    require(digest(config) == plan["runtime_config_sha256"], "D10 runtime config differs")
    table = raw.parent / "input.json"
    require(digest(table) == plan["target"]["input_sha256"], "D10 leaf input differs")
    client, engine, source, thermal = (
        read_json(raw / name)
        for name in ("client.json", "engine.json", "sources.json", "d2-thermal.json")
    )
    require(
        client["pair_id"] == identity
        and client["tier"] == engine["tier"] == plan["target"]["tier"]
        and thermal["status"] == "admitted"
        and thermal["plan_sha256"] == digest(proposal)
        and not list(raw.glob("*-step.json")),
        "D10 measured endpoint, thermal role or capture differs",
    )
    verify_native_sources(plan, source)
    check_d9_profile(engine["profile"], plan)
    inputs = read_json(table)
    gates = read_json(config)["gates"]
    expected = {row["request_id"]: row for row in inputs["requests"]}
    _, submit_lag = client_requests(client, inputs, expected, gates, "calibration")
    require(
        len(client["requests"]) == plan["target"]["count_per_engine"]
        and all(
            row["status"] == "completed"
            and row["first_token_ns"] <= row["planned_ns"] + gates["deadline_ns"]
            for row in client["requests"]
        ),
        "D10 complete on-time denominator differs",
    )
    timing = timing_leaf(client, engine, inputs, gates, leaf_end_ns=leaf_end_ns)
    require(
        timing["clock"]["verified"] and timing["status"] == "complete",
        f"D10 original native clock or timing is unverified: {timing['clock']['issues']}",
    )
    order = dict(zip(ENDPOINTS, plan["placement_order"], strict=True))
    placement = observed_placement(
        client, engine, identity, order, thermal=thermal, leaf_end_ns=leaf_end_ns
    )
    require(placement["verified"], "D10 measured CPU placement is unverified")
    return {"clock": timing["clock"], "submit_lag_ns": submit_lag, "placement": placement}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--attempt", type=Path)
    mode.add_argument("--raw", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--leaf-end-ns", type=int)
    parser.add_argument("--identity", choices=ENDPOINTS)
    args = parser.parse_args()
    try:
        require(not sys.flags.optimize, "run unoptimized D10 auditor")
        if args.attempt:
            require(
                args.output and not args.config and args.leaf_end_ns is None and not args.identity,
                "D10 attempt requires only --output",
            )
            require(read_json(args.proposal)["role"] == D10_ROLE, "D10 proposal role differs")
            report = review(args.attempt.resolve(), args.proposal.resolve())
            with args.output.open("x") as stream:
                json.dump(report, stream, indent=2, default=str, allow_nan=False)
                stream.write("\n")
        else:
            require(
                args.config and args.leaf_end_ns is not None and args.identity and not args.output,
                "D10 leaf requires --config, --leaf-end-ns and --identity",
            )
            print(
                json.dumps(
                    check_leaf(
                        args.raw, args.config, args.proposal, args.leaf_end_ns, args.identity
                    ),
                    allow_nan=False,
                )
            )
    except (KeyError, OSError, TypeError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"M5 D10 evidence rejected: {error}\n")


if __name__ == "__main__":
    main()
