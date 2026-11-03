"""Bind the dual-window diagnostic to the original four-slot D8 condition."""

from pathlib import Path

from .m5_campaign import D8_ROLE, D9_PLAN_SHA, digest, read_json, require
from .m5_d2 import D2R110_PROPOSAL_NAME
from .m5_d8 import D8_PREVIOUS_ANALYSIS, D8_PROPOSAL_NAME, d8_plan, d8_specs

D9_PROPOSAL_NAME = "proposed-d9.json"
D9_PREVIOUS_ANALYSIS = "previous-d8-analysis.json"
D9_RETAINED_FILENAMES = (
    "original-c3-result.json",
    D8_PROPOSAL_NAME,
    D2R110_PROPOSAL_NAME,
    D8_PREVIOUS_ANALYSIS,
    D9_PREVIOUS_ANALYSIS,
)


def d9_plan(proposal, inherited_d8, inherited_d2r110):
    require(digest(proposal) == D9_PLAN_SHA, "D9 reviewed plan differs")
    plan = read_json(proposal)
    require(
        digest(inherited_d8) == plan["retained_D8"]["proposal_sha256"],
        "D9 inherited D8 plan differs",
    )
    original = d8_plan(inherited_d8, inherited_d2r110)
    require(
        all(original["target"][key] == value for key, value in plan["target"].items())
        and all(
            original["stability_gate"][key] == value
            for key, value in plan["stability_gate"].items()
        )
        and plan["retained_C3"] == original["retained_C3"],
        "D9 original D8 condition or gate differs",
    )
    return plan


def d9_predecessor(plan, previous_d8):
    expected = plan["retained_D8"]
    require(digest(previous_d8) == expected["analysis_sha256"], "D9 retained D8 analysis differs")
    prior = read_json(previous_d8)
    decision = prior["decision"]
    require(
        prior["role"] == D8_ROLE
        and prior["status"] == expected["status"]
        and prior["proposal_sha256"] == expected["proposal_sha256"]
        and prior["execution_commit"] == prior["analysis_commit"] == expected["execution_commit"]
        and prior["input_sha256"] == plan["target"]["input_sha256"]
        and decision["thermal_control"] == "verified"
        and decision["observed_endpoint_stability"] == expected["observed_endpoint_stability"]
        and decision["qualification_started"] is False
        and prior["resource_release"]["complete"] is True,
        "D9 retained D8 diagnostic outcome differs",
    )
    return prior


def d9_specs(
    config, proposal, table, original, inherited_d8, inherited_d2r110, previous_d7, previous_d8
):
    plan = d9_plan(proposal, inherited_d8, inherited_d2r110)
    d9_predecessor(plan, previous_d8)
    specs = d8_specs(config, inherited_d8, table, original, inherited_d2r110, previous_d7)
    require([spec["id"] for spec in specs] == plan["runs"], "D9 original leaf order differs")
    return [
        {**spec, "d9_capture": capture}
        for spec, capture in zip(specs, plan["capture"], strict=True)
    ]


def check_d9_profile(profile, plan):
    expected = plan["profile"]
    require(
        all(
            key in profile and type(profile[key]) is type(value) and profile[key] == value
            for key, value in expected.items()
            if key != "num_gpu_blocks_override"
        )
        and profile.get("num_gpu_blocks_override") is None,
        "D9 actual runner slots, active override, KV blocks or batch budget differ",
    )


def staged_d9_specs(config_path, proposal):
    config_path = Path(config_path)
    proposal = Path(proposal)
    staged = proposal.parent
    original, inherited_d8, inherited_d2r110, previous_d7, previous_d8 = (
        staged / name for name in D9_RETAINED_FILENAMES
    )
    plan = d9_plan(proposal, inherited_d8, inherited_d2r110)
    config = read_json(config_path)
    require(
        digest(config_path) == plan["runtime_config_sha256"]
        and digest(config_path.with_name(Path(config["baseline_config"]).name))
        == plan["baseline_config_sha256"]
        and plan["budget"]["work_deadline_s"] == config["gates"]["max_allocation_work_s"],
        "D9 original runtime configuration or work envelope differs",
    )
    return d9_specs(
        config,
        proposal,
        staged / "input.json",
        original,
        inherited_d8,
        inherited_d2r110,
        previous_d7,
        previous_d8,
    )
