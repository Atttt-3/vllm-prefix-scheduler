"""Bind the single approved D2 diagnostic to its plan and historical inputs."""

from .m5_campaign import D2_ROLE, digest, read_json, require
from .m5_d1 import C3_BLOCKED_SHA, D1_INPUT_SHA

D2_PROPOSAL_SHA = "1d23948556bc102b1151e3bb4215ca2e161ce17fa786b5752ff8fe42f7eed96c"
D2_D1_REANALYSIS_SHA = "a635b73e59ba3d8725c61cfc868369dfeaf146b9e6750ff12bf5ed1979de42ad"


def d2_specs(config, proposal, table, original):
    require(digest(proposal) == D2_PROPOSAL_SHA, "D2 reviewed plan differs")
    require(digest(table) == D1_INPUT_SHA, "D2 archived input differs")
    require(
        digest(original) == C3_BLOCKED_SHA and read_json(original)["status"] == "blocked",
        "D2 requires the retained blocked C3",
    )
    plan = read_json(proposal)
    rules = plan["stability_gate"]
    gates = config["gates"]
    require(
        plan["role"] == D2_ROLE
        and plan["d1_analysis_sha256"] == D2_D1_REANALYSIS_SHA
        and rules["ttft_p95_absolute_floor_ns"] == gates["overhead_ttft_ns"]
        and rules["ttft_p95_relative_margin"] == gates["overhead_ttft_rel"]
        and rules["throughput_both_last_terminal_relative_margin"]
        == gates["overhead_throughput_rel"]
        and plan["budget"]["allocation_work_deadline_s"] == gates["max_allocation_work_s"],
        "D2 scope or thresholds differ",
    )
    target = plan["target"]
    workload = read_json(table)
    require(
        target["input_sha256"] == D1_INPUT_SHA
        and workload["scenario"] == target["scenario"]
        and workload["seed"] == target["seed"]
        and workload["rate_rps"] == target["arrival_rate_rps"]
        and len(workload["requests"]) == target["count_per_engine"],
        "D2 input and frozen run specs disagree",
    )
    return [
        {
            "id": identity,
            "scenario": target["scenario"],
            "seed": target["seed"],
            "count": target["count_per_engine"],
            "mode": target["mode"],
            "tier": target["tier"],
            "rate_rps": target["arrival_rate_rps"],
            "interval_ms": None,
            "mechanism": None,
        }
        for identity in target["fresh_engines"]
    ]
