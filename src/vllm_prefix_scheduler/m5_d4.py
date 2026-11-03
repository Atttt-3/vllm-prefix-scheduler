"""Bind D4 to the reviewed rate contrast, exact input tables and D3 lineage."""

from .m5_campaign import D3_ROLE, D4_ROLE, digest, read_json, require
from .m5_contract import workload_gate
from .m5_d2 import d3_plan, d3_specs

D4_PLAN_SHA = "d3b200bcda5a4017b40095c741d27b584eac9b1da6fc0a70b6a6806ecd5f075f"
D4_PROPOSAL_NAME = "proposed-d4.json"
D4_INPUT_8_NAME = "mixed-8rps.json"
D4_INPUT_6_NAME = "mixed-6rps.json"
D4_PREVIOUS_ANALYSIS = "previous-d3-analysis.json"


def d4_plan(proposal, inherited_d3):
    require(digest(proposal) == D4_PLAN_SHA, "D4 reviewed plan differs")
    plan = read_json(proposal)
    require(digest(inherited_d3) == plan["D3_plan_sha256"], "D4 inherited thermal plan differs")
    return plan, d3_plan(read_json(inherited_d3))


def d4_tables(config, plan, input_8, input_6):
    tables = {}
    target = plan["target"]
    for rate, path in ((8, input_8), (6, input_6)):
        binding = target["arrival_tables"][str(rate)]
        require(digest(path) == binding["sha256"], f"D4 {rate} rps input hash differs")
        table = read_json(path)
        require(
            table["rate_rps"] == rate
            and len(table["requests"]) == target["count_per_engine"]
            and table["requests"][-1]["offset_ns"] == binding["last_offset_ns"],
            f"D4 {rate} rps input scope differs",
        )
        workload_gate(table, config, target, "calibration")
        tables[rate] = table
    return tables


def d4_specs(config, proposal, input_8, input_6, original_c3, inherited_d3, previous_d3_analysis):
    plan, _ = d4_plan(proposal, inherited_d3)
    specs = d3_specs(config, inherited_d3, input_8, original_c3)
    d4_tables(config, plan, input_8, input_6)
    require(
        digest(previous_d3_analysis) == plan["D3_analysis_sha256"],
        "D4 retained D3 analysis differs",
    )
    previous = read_json(previous_d3_analysis)
    require(
        previous["role"] == D3_ROLE
        and previous["status"] == "diagnostic-complete"
        and previous["proposal_sha256"] == plan["D3_plan_sha256"]
        and previous["decision"]["thermal_control"] == "verified"
        and previous["decision"]["endpoint_stability_under_control"] == "unstable"
        and plan["role"] == D4_ROLE,
        "D4 retained D3 result scope differs",
    )
    rates = [plan["target"]["preheat_rate_rps"], *plan["target"]["measure_order_rps"]]
    return [{**spec, "rate_rps": rate} for spec, rate in zip(specs, rates, strict=True)]
