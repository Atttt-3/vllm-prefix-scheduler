"""Bind the single-factor CPU placement diagnostic to the D8 condition."""

from pathlib import Path

from .m5_campaign import D9_ROLE, D10_PLAN_SHA, D10_ROLE, digest, read_json, require
from .m5_d8 import d8_plan
from .m5_d9 import D9_PROPOSAL_NAME, D9_RETAINED_FILENAMES, staged_d9_specs

D10_PROPOSAL_NAME = "proposed-d10.json"
D10_PREVIOUS_ANALYSIS = "previous-d9-analysis.json"
D10_RETAINED_FILENAMES = (*D9_RETAINED_FILENAMES, D9_PROPOSAL_NAME, D10_PREVIOUS_ANALYSIS)


def d10_plan(proposal, inherited_d8, inherited_d2r110):
    require(digest(proposal) == D10_PLAN_SHA, "D10 reviewed plan differs")
    plan = read_json(proposal)
    require(
        digest(inherited_d8) == plan["retained_D8"]["proposal_sha256"]
        and digest(inherited_d2r110) == plan["D2R110_plan_sha256"],
        "D10 inherited D8 or D2-R110 plan differs",
    )
    original = d8_plan(inherited_d8, inherited_d2r110)
    require(
        all(original["target"][key] == value for key, value in plan["target"].items())
        and all(
            original["stability_gate"][key] == value
            for key, value in plan["stability_gate"].items()
        )
        and plan["retained_C3"] == original["retained_C3"]
        and plan["retained_D7"] == original["retained_D7"],
        "D10 original D8 condition or gate differs",
    )
    return plan


def d10_predecessor(plan, previous_d9):
    expected = plan["retained_D9"]
    require(digest(previous_d9) == expected["analysis_sha256"], "D10 retained D9 analysis differs")
    prior = read_json(previous_d9)
    decision = prior["decision"]
    require(
        prior["role"] == D9_ROLE
        and prior["status"] == expected["status"]
        and prior["proposal_sha256"] == expected["proposal_sha256"]
        and prior["execution_commit"] == prior["analysis_commit"] == expected["execution_commit"]
        and prior["input_sha256"] == plan["target"]["input_sha256"]
        and prior["retained_c3_sha256"] == plan["retained_C3"]["sha256"]
        and decision["thermal_control"] == "verified"
        and decision["capture_equivalence"] == expected["capture_equivalence"]
        and decision["qualification_started"] is False
        and prior["resource_release"]["complete"] is True,
        "D10 retained D9 diagnostic outcome differs",
    )
    return prior


def d10_lineage(plan, proposal_sha256):
    return {
        "role": D10_ROLE,
        "proposal_sha256": proposal_sha256,
        "input_sha256": plan["target"]["input_sha256"],
        "retained_c3_sha256": plan["retained_C3"]["sha256"],
        "inherited_d8_proposal_sha256": plan["retained_D8"]["proposal_sha256"],
        "inherited_d2r110_proposal_sha256": plan["D2R110_plan_sha256"],
        "retained_d7_analysis_sha256": plan["retained_D7"]["analysis_sha256"],
        "retained_d8_analysis_sha256": plan["retained_D8"]["analysis_sha256"],
        "retained_d9_analysis_sha256": plan["retained_D9"]["analysis_sha256"],
    }


def d10_specs(parent_specs, plan):
    require([spec["id"] for spec in parent_specs] == plan["runs"], "D10 frozen leaf order differs")
    return [
        {
            **{key: value for key, value in spec.items() if key != "d9_capture"},
            "d10_placement": placement,
        }
        for spec, placement in zip(
            parent_specs, ("inherited", *plan["placement_order"]), strict=True
        )
    ]


def staged_d10_specs(config_path, proposal):
    config_path, proposal = Path(config_path), Path(proposal)
    staged = proposal.parent
    _, inherited_d8, inherited_d2r110, _, _, inherited_d9, previous_d9 = (
        staged / name for name in D10_RETAINED_FILENAMES
    )
    plan = d10_plan(proposal, inherited_d8, inherited_d2r110)
    d10_predecessor(plan, previous_d9)
    config = read_json(config_path)
    require(
        digest(inherited_d9) == plan["retained_D9"]["proposal_sha256"]
        and digest(config_path) == plan["runtime_config_sha256"]
        and digest(config_path.with_name(Path(config["baseline_config"]).name))
        == plan["baseline_config_sha256"]
        and plan["budget"]["work_deadline_s"] == config["gates"]["max_allocation_work_s"],
        "D10 original runtime configuration or work envelope differs",
    )
    return d10_specs(staged_d9_specs(config_path, inherited_d9), plan)
