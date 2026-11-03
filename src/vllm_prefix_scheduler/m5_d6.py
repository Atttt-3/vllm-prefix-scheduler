"""Bind D6 to the D5-R heat control and fixed four-slot replay."""

from .m5_campaign import D5R_ROLE, digest, read_json, require
from .m5_d5 import d5_profile
from .m5_d5r import d5r_plan, d5r_specs

D6_PLAN_SHA = "0cce4570b6354e90d9d73a6cdc1771e98c51ea89f4ba16cf52bf71dd920b3275"
D6_PROPOSAL_NAME = "proposed-d6.json"
D6_PREVIOUS_ANALYSIS = "previous-d5-r-analysis.json"
D6_RUNS = ("thermal-pilot", "E1", "E2", "E3")
D6_STEPPED = ("E2", "E3")


def d6_plan(proposal, inherited_d5r, inherited_d5, inherited_d3):
    require(digest(proposal) == D6_PLAN_SHA, "D6 reviewed plan differs")
    plan = read_json(proposal)
    require(digest(inherited_d5r) == plan["D5R_plan_sha256"], "D6 inherited D5-R plan differs")
    d5r, d5, d3 = d5r_plan(inherited_d5r, inherited_d5, inherited_d3)
    return plan, d5r, d5, d3


def d6_sources(plan, source):
    require(
        all(
            source["upstream_sha256"].get(name) == sha
            for name, sha in plan["runtime_sources"].items()
        ),
        "D6 installed native boundary source differs",
    )


def d6_specs(
    config,
    proposal,
    table,
    original_c3,
    inherited_d5r,
    inherited_d5,
    inherited_d3,
    previous_d4,
    previous_d5,
    previous_d5r,
):
    plan, _, d5, _ = d6_plan(proposal, inherited_d5r, inherited_d5, inherited_d3)
    require(
        digest(previous_d5r) == plan["D5R_analysis_sha256"],
        "D6 retained D5-R analysis differs",
    )
    prior = read_json(previous_d5r)
    require(
        prior["role"] == D5R_ROLE
        and prior["status"] == "diagnostic-complete"
        and prior["proposal_sha256"] == plan["D5R_plan_sha256"]
        and prior["decision"]["thermal_control"] == "verified"
        and prior["decision"]["endpoint_stability_under_control"] == "unstable"
        and prior["d5_assessment"]["cap4_backlog_reproduced"]
        and plan["budget"]["work_deadline_s"] <= config["gates"]["max_allocation_work_s"],
        "D6 prior diagnostic or work envelope differs",
    )
    specs = d5r_specs(
        config,
        inherited_d5r,
        table,
        original_c3,
        inherited_d5,
        inherited_d3,
        previous_d4,
        previous_d5,
    )
    return [
        {**spec, **d5_profile(d5, "E1"), "d6_capture": capture}
        for spec, capture in zip(specs[: len(D6_RUNS)], plan["capture"], strict=True)
    ]
