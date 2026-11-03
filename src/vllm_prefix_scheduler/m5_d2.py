"""Bind each one-allocation thermal diagnostic to its reviewed plan and inputs."""

import hashlib
import json

from .m5_campaign import (
    D2_ROLE,
    D2R110_OFFICIAL_ANALYSIS_SHA256,
    D2R110_ROLE,
    D2R_OFFICIAL_ANALYSIS_SHA256,
    D2R_ROLE,
    D3_ROLE,
    digest,
    is_hash,
    read_json,
    require,
)
from .m5_d1 import C3_BLOCKED_SHA, D1_INPUT_SHA

D2_PROPOSAL_SHA = "1d23948556bc102b1151e3bb4215ca2e161ce17fa786b5752ff8fe42f7eed96c"
D2_D1_REANALYSIS_SHA = "a635b73e59ba3d8725c61cfc868369dfeaf146b9e6750ff12bf5ed1979de42ad"
D2_ANALYSIS_SHA = "0f369e956dd62373a310d467054d684e371b63303662563b7eba790ed12400ba"
D2R_MEASURED_CAP_S = 95
D2R_PROPOSAL_NAME = "proposed-d2-r.json"
D2R_PROPOSAL_SHA = "fe0fde4fa66c498ffafc1c5f1f8fb89c25492a7a3ee6edbb7d06fc1dddcd3eba"
D2R110_PROPOSAL_NAME = "proposed-d2-r110.json"
D2R110_PROPOSAL_SHA = "891231f0444a0347be7104bb53ac4c4586d91c7d68e5500a5d971e29046b55c3"
D2R110_MEASURED_CAP_S = 110
D3_PROPOSAL_NAME = "proposed-d3.json"
D3_SCHEDULER_SOURCE = "vllm/v1/core/sched/scheduler.py"
D3_NATIVE_SOURCES = frozenset(
    (
        D3_SCHEDULER_SOURCE,
        "vllm/outputs.py",
        "vllm/v1/engine/__init__.py",
        "vllm/v1/engine/async_llm.py",
        "vllm/v1/engine/core.py",
        "vllm/v1/engine/output_processor.py",
        "vllm/v1/metrics/stats.py",
    )
)
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


def d2r110_plan(plan):
    require(
        plan["role"] == D2R110_ROLE
        and plan["retained_D2R"]
        == {
            "status": "diagnostic-incomplete",
            "proposal_sha256": D2R_PROPOSAL_SHA,
            "analysis_sha256": D2R_OFFICIAL_ANALYSIS_SHA256,
        },
        "D2-R110 role or retained D2-R analysis differs",
    )
    conditioning, budget = plan["conditioning"], plan["budget"]
    require(
        all(
            type(actual) is int and actual == expected
            for actual, expected in (
                (conditioning["per_measured_engine_cap_s"], D2R110_MEASURED_CAP_S),
                (plan["shadow_conditioning_cap_s"], D2R_MEASURED_CAP_S),
                (budget["four_conditioning_envelope_seconds"], 4 * D2R110_MEASURED_CAP_S),
                (budget["job_overhead_and_cleanup_seconds"], 120),
                (budget["estimated_work_envelope_seconds"], 1035),
            )
        ),
        "D2-R110 measured cap, shadow or budget differs",
    )
    d2r_plan(
        {
            **plan,
            "role": D2R_ROLE,
            "conditioning": {**conditioning, "per_measured_engine_cap_s": D2R_MEASURED_CAP_S},
            "budget": {
                **budget,
                "four_conditioning_envelope_seconds": 4 * D2R_MEASURED_CAP_S,
                "job_overhead_and_cleanup_seconds": 160,
                "estimated_work_envelope_seconds": 1015,
            },
        }
    )
    return plan


def d2r110_specs(config, proposal, table, original):
    return _specs(config, d2r110_plan(read_json(proposal)), table, original, D2R110_ROLE)


def d3_plan(plan):
    require(
        plan["role"] == D3_ROLE
        and plan["retained_D2R110"]
        == {
            "status": "diagnostic-complete",
            "proposal_sha256": D2R110_PROPOSAL_SHA,
            "analysis_sha256": D2R110_OFFICIAL_ANALYSIS_SHA256,
        },
        "D3 role or retained D2-R110 analysis differs",
    )
    d2r110_plan({**plan, "role": D2R110_ROLE})
    native = plan["native_timing"]
    sources = native["source_sha256"]
    require(
        set(native) == {"schema_version", "source_sha256"}
        and type(native["schema_version"]) is int
        and native["schema_version"] == 1
        and set(sources) == D3_NATIVE_SOURCES
        and all(is_hash(value) for value in sources.values()),
        "D3 native timing protocol or source inventory differs",
    )
    return plan


def d3_specs(config, proposal, table, original):
    plan = d3_plan(read_json(proposal))
    configured = config["extra_upstream_sha256"]
    require(
        all(
            configured.get(path) == value
            for path, value in plan["native_timing"]["source_sha256"].items()
            if path != D3_SCHEDULER_SOURCE  # Bound by native-m2b.json in collect_sources.
        ),
        "D3 native timing sources are detached from the frozen M5 configuration",
    )
    return _specs(config, plan, table, original, D3_ROLE)


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
