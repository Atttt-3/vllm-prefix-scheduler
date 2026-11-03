"""Bind the D5-R pilot selection change to the original D5 contrast."""

from .m5_campaign import D5_ROLE, digest, read_json, require
from .m5_d5 import d5_plan, d5_specs

D5R_PLAN_SHA = "0bbda0693f6bffd9eff82c963ca9f06d41d17d83d3fd6f9009ada302852d4d9a"
D5R_PROPOSAL_NAME = "proposed-d5-r.json"
D5R_PREVIOUS_ANALYSIS = "previous-d5-analysis.json"


def d5r_plan(proposal, inherited_d5, inherited_d3):
    require(digest(proposal) == D5R_PLAN_SHA, "D5-R reviewed plan differs")
    plan = read_json(proposal)
    require(digest(inherited_d5) == plan["D5_plan_sha256"], "D5-R inherited D5 plan differs")
    d5, d3 = d5_plan(inherited_d5, inherited_d3)
    return plan, d5, d3


def d5r_specs(
    config,
    proposal,
    table,
    original_c3,
    inherited_d5,
    inherited_d3,
    previous_d4_analysis,
    previous_d5_analysis,
):
    plan, _, _ = d5r_plan(proposal, inherited_d5, inherited_d3)
    require(
        digest(previous_d5_analysis) == plan["D5_analysis_sha256"],
        "D5-R retained D5 analysis differs",
    )
    prior = read_json(previous_d5_analysis)
    require(
        prior["role"] == D5_ROLE
        and prior["status"] == "diagnostic-incomplete"
        and prior["proposal_sha256"] == plan["D5_plan_sha256"]
        and prior["runs"]["thermal-pilot"]["status"] == "target-frozen"
        and prior["runs"]["E1"]["status"] == "invalid"
        and prior["d5_assessment"]["optimization_signal"] == "unverified",
        "D5-R retained D5 result scope differs",
    )
    return d5_specs(config, inherited_d5, table, original_c3, inherited_d3, previous_d4_analysis)
