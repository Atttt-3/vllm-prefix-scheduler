"""Bind each one-allocation thermal diagnostic to its reviewed plan and inputs."""

import hashlib
import json

from .m5_campaign import D2_ROLE, D2R_ROLE, digest, read_json, require
from .m5_d1 import C3_BLOCKED_SHA, D1_INPUT_SHA

D2_PROPOSAL_SHA = "1d23948556bc102b1151e3bb4215ca2e161ce17fa786b5752ff8fe42f7eed96c"
D2_D1_REANALYSIS_SHA = "a635b73e59ba3d8725c61cfc868369dfeaf146b9e6750ff12bf5ed1979de42ad"
D2_ANALYSIS_SHA = "0f369e956dd62373a310d467054d684e371b63303662563b7eba790ed12400ba"
D2R_MEASURED_CAP_S = 95
D2R_PROPOSAL_NAME = "proposed-d2-r.json"
_D2_INVARIANTS_SHA = "8c3695364073476f638a125a2404d0155c61fd98728c56578ee39a6ccc0f78dc"


def d2_specs(config, proposal, table, original):
    require(digest(proposal) == D2_PROPOSAL_SHA, "D2 reviewed plan differs")
    return _specs(config, read_json(proposal), table, original, D2_ROLE)


def d2r_plan(plan):
    """Validate the retry against the original plan, changing only its cap and budget arithmetic."""
    require(
        plan["role"] == D2R_ROLE
        and plan["retained_D2"]
        == {
            "status": "diagnostic-incomplete",
            "proposal_sha256": D2_PROPOSAL_SHA,
            "analysis_sha256": D2_ANALYSIS_SHA,
        },
        "D2-R role or retained D2 analysis differs",
    )
    conditioning, budget = plan["conditioning"], plan["budget"]
    require(
        conditioning["per_measured_engine_cap_s"] == D2R_MEASURED_CAP_S
        and budget["four_conditioning_envelope_seconds"] == 4 * D2R_MEASURED_CAP_S
        and budget["estimated_work_envelope_seconds"] == 1015,
        "D2-R prospective measured cap or budget differs",
    )
    normalized = {
        "target": plan["target"],
        "stability_gate": plan["stability_gate"],
        "d1_analysis_sha256": plan["d1_analysis_sha256"],
        "retained_C3": plan["retained_C3"],
        "conditioning": {**conditioning, "per_measured_engine_cap_s": 75},
        "budget": {
            **budget,
            "four_conditioning_envelope_seconds": 300,
            "estimated_work_envelope_seconds": 935,
        },
    }
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    require(
        hashlib.sha256(canonical).hexdigest() == _D2_INVARIANTS_SHA,
        "D2-R original input, thermal checks, stability gate or budget changed",
    )
    return plan


def d2r_specs(config, proposal, table, original):
    return _specs(config, d2r_plan(read_json(proposal)), table, original, D2R_ROLE)


def _specs(config, plan, table, original, role):
    require(digest(table) == D1_INPUT_SHA, "D2 archived input differs")
    require(
        digest(original) == C3_BLOCKED_SHA and read_json(original)["status"] == "blocked",
        "D2 requires the retained blocked C3",
    )
    rules = plan["stability_gate"]
    gates = config["gates"]
    require(
        plan["role"] == role
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
