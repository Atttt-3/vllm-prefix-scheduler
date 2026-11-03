"""Freeze a same-condition endpoint stability check on the original baseline."""

from .m5_campaign import D7_ROLE, digest, read_json, require
from .m5_d2 import D2R110_PROPOSAL_SHA, d2r110_specs

D8_PLAN_SHA = "18907f0bdd93c6ab1f51f26446b4a689c411fa880578a8b7540bbe8498ec8772"
D8_PROPOSAL_NAME = "proposed-d8.json"
D8_PREVIOUS_ANALYSIS = "previous-d7-analysis.json"


def d8_plan(proposal, inherited_d2r110):
    require(digest(proposal) == D8_PLAN_SHA, "D8 reviewed plan differs")
    plan = read_json(proposal)
    require(
        digest(inherited_d2r110) == plan["D2R110_plan_sha256"] == D2R110_PROPOSAL_SHA,
        "D8 inherited D2-R110 plan differs",
    )
    original = read_json(inherited_d2r110)
    require(
        all(
            plan[key] == original[key]
            for key in ("target", "conditioning", "stability_gate", "budget", "retained_C3")
        ),
        "D8 original same-condition protocol differs",
    )
    return plan


def d8_predecessor(plan, previous_d7):
    expected = plan["retained_D7"]
    require(digest(previous_d7) == expected["analysis_sha256"], "D8 retained D7 analysis differs")
    prior = read_json(previous_d7)
    require(
        prior["role"] == D7_ROLE
        and prior["status"] == expected["status"]
        and prior["proposal_sha256"] == expected["proposal_sha256"]
        and prior["execution_commit"] == prior["analysis_commit"] == expected["execution_commit"]
        and prior["decision"]["thermal_control"] == "verified"
        and prior["decision"]["qualification_started"] is False
        and prior["d7_assessment"]["capture_equivalence"] == "supported",
        "D8 retained D7 diagnostic outcome differs",
    )
    return prior


def d8_specs(config, proposal, table, original, inherited_d2r110, previous_d7):
    plan = d8_plan(proposal, inherited_d2r110)
    d8_predecessor(plan, previous_d7)
    return d2r110_specs(config, inherited_d2r110, table, original)
