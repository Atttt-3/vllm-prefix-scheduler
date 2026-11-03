"""Bind D7's late-window ABBA replay to its sealed predecessors."""

from .m5_campaign import D6_ROLE, digest, read_json, require
from .m5_d6 import D6_RUNS, d6_plan, d6_specs

D7_PLAN_SHA = "d74e98aa091f0b54d6da48dc4ffdc281d560d6b87df460586f4d516ed6d82b21"
D7_PROPOSAL_NAME = "proposed-d7.json"
D7_PREVIOUS_ANALYSIS = "previous-d6-analysis.json"
D7_LATE_TIMELINE = "d6-late-timeline.json"
D7_RUNS = (*D6_RUNS, "E4")


def d7_plan(proposal, inherited_d6, inherited_d5r, inherited_d5, inherited_d3):
    require(digest(proposal) == D7_PLAN_SHA, "D7 reviewed plan differs")
    plan = read_json(proposal)
    require(digest(inherited_d6) == plan["D6_plan_sha256"], "D7 inherited D6 plan differs")
    _, d5r, d5, d3 = d6_plan(inherited_d6, inherited_d5r, inherited_d5, inherited_d3)
    return plan, d5r, d5, d3


def d7_predecessor(plan, previous_d6, late_timeline):
    require(digest(previous_d6) == plan["D6_analysis_sha256"], "D7 retained D6 analysis differs")
    require(digest(late_timeline) == plan["late_timeline_sha256"], "D7 late timeline differs")
    prior = read_json(previous_d6)
    require(
        prior["role"] == D6_ROLE
        and prior["status"] == "diagnostic-complete"
        and prior["proposal_sha256"] == plan["D6_plan_sha256"]
        and prior["decision"]["thermal_control"] == "verified"
        and prior["decision"]["localization"] == "unresolved"
        and prior["decision"]["qualification_started"] is False,
        "D7 retained D6 diagnostic outcome differs",
    )
    return prior


def d7_specs(
    config,
    proposal,
    table,
    original_c3,
    inherited_d6,
    inherited_d5r,
    inherited_d5,
    inherited_d3,
    previous_d4,
    previous_d5,
    previous_d5r,
    previous_d6,
    late_timeline,
):
    plan, _, _, _ = d7_plan(proposal, inherited_d6, inherited_d5r, inherited_d5, inherited_d3)
    d7_predecessor(plan, previous_d6, late_timeline)
    require(
        plan["budget"]["work_deadline_s"] <= config["gates"]["max_allocation_work_s"],
        "D7 work envelope differs",
    )
    specs = d6_specs(
        config,
        inherited_d6,
        table,
        original_c3,
        inherited_d5r,
        inherited_d5,
        inherited_d3,
        previous_d4,
        previous_d5,
        previous_d5r,
    )
    specs.append({**specs[1], "id": D7_RUNS[-1]})
    for spec, capture in zip(specs, plan["capture"], strict=True):
        require(spec.pop("d6_capture") == capture, "D7 staged capture order differs")
        spec["d7_capture"] = capture
    return specs
