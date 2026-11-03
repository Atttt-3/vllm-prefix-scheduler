"""Bind D5's single-variable admission-cap contrast to retained D4 evidence."""

from .m5_campaign import D4_ROLE, D5_ROLE, digest, read_json, require
from .m5_d2 import d3_plan, d3_specs

D5_PLAN_SHA = "dc4817bb421c774e2730aba541bf4752d946cb4a3d05a0edce093fdb1df85d1f"
D5_PROPOSAL_NAME = "proposed-d5.json"
D5_PREVIOUS_ANALYSIS = "previous-d4-analysis.json"
D5_RUNS = ("thermal-pilot", "E1", "E2", "E3", "E4")


def d5_plan(proposal, inherited_d3):
    require(digest(proposal) == D5_PLAN_SHA, "D5 reviewed plan differs")
    plan = read_json(proposal)
    require(digest(inherited_d3) == plan["D3_plan_sha256"], "D5 inherited thermal plan differs")
    return plan, d3_plan(read_json(inherited_d3))


def d5_profile(plan, leaf):
    target = plan["target"]
    caps = (
        target["pilot_max_num_active_seqs"],
        *target["measure_order_max_num_active_seqs"],
    )
    require(leaf in D5_RUNS, "D5 leaf identity differs from the frozen order")
    return {
        "max_num_active_seqs": caps[D5_RUNS.index(leaf)],
        "max_num_seqs": target["max_num_seqs"],
        "max_num_batched_tokens": target["max_num_batched_tokens"],
        "num_gpu_blocks": target["num_gpu_blocks_override"],
    }


def d5_specs(config, proposal, input_8, original_c3, inherited_d3, previous_d4_analysis):
    plan, inherited = d5_plan(proposal, inherited_d3)
    target, d3_target = plan["target"], inherited["target"]
    require(
        plan["role"] == D5_ROLE
        and all(
            target[key] == d3_target[key]
            for key in (
                "scenario",
                "seed",
                "mode",
                "tier",
                "input_sha256",
                "count_per_engine",
                "arrival_rate_rps",
            )
        )
        and plan["budget"]["work_deadline_s"] == config["gates"]["max_allocation_work_s"],
        "D5 input, runtime or allocation limit differs",
    )
    specs = d3_specs(config, inherited_d3, input_8, original_c3)
    require(
        digest(previous_d4_analysis) == plan["D4_analysis_sha256"],
        "D5 retained D4 analysis differs",
    )
    prior = read_json(previous_d4_analysis)
    require(
        prior["role"] == D4_ROLE
        and prior["status"] == "diagnostic-complete"
        and prior["proposal_sha256"] == plan["D4_plan_sha256"]
        and prior["decision"]["thermal_control"] == "verified"
        and prior["d4_assessment"]["pressure_signal"] == "supported",
        "D5 retained D4 result scope differs",
    )
    return [{**spec, **d5_profile(plan, spec["id"])} for spec in specs]
