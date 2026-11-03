"""Bind the logical C3 qualification to both same-node job inventories."""

import hashlib
import json
from pathlib import Path

from plan_m5_calibration import C3_SPLIT_STAGES, aggregate_c3

from vllm_prefix_scheduler.m5_campaign import (
    D2R110_ROLE,
    digest,
    id_ok,
    qualification_roles,
    read_json,
    require,
)
from vllm_prefix_scheduler.m5_resources import (
    attempt_evidence_paths,
    audit_attempt,
    submitted_attempts,
)


def verify_d2r110_qualification(history, authorization):
    report_path = history.parent / "d2-r110-review/analysis.json"
    report_hash = authorization["d2r110_analysis_sha256"]
    require(
        digest(report_path) == report_hash,
        "split C3 authorization is detached from D2-R110 review",
    )
    report = read_json(report_path)
    attempts = [
        path
        for path in submitted_attempts(history)
        if read_json(path / "planned.json")["role"] == D2R110_ROLE
    ]
    require(len(attempts) == 1, "split C3 requires one released D2-R110 attempt")
    attempt = attempts[0]
    planned = read_json(attempt / "planned.json")
    resource = json.loads(json.dumps(audit_attempt(attempt, require_completed=True), default=str))
    require(
        report["role"] == D2R110_ROLE
        and report["status"] == "diagnostic-complete"
        and report["execution_commit"] == report["analysis_commit"] == planned["execution_commit"]
        and report["proposal_sha256"] == planned["auxiliary_sha256"]
        and report["resource_release"] == resource
        and report["decision"]["thermal_control"] == "verified"
        and report["decision"]["endpoint_stability_under_control"] == "stable"
        and report["decision"]["qualification_started"] is False,
        "D2-R110 did not verify controlled endpoint stability",
    )
    return report_hash


def c3_bundle(config_path, history, stages, commit):
    config = read_json(config_path)
    config_sha256 = digest(config_path)
    authorizations = [
        read_json(stages[role] / "authorization.json") for role in qualification_roles(config)
    ]
    require(
        len({record["d2r110_analysis_sha256"] for record in authorizations}) == 1,
        "qualification roles bind different D2-R110 reviews",
    )
    prerequisite_sha = verify_d2r110_qualification(history, authorizations[0])
    c2 = read_json(stages["C2"] / "node/calibration-result.json")
    require(
        c2["stage"] == "C2"
        and c2["status"] == "qualified"
        and c2["config_sha256"] == config_sha256,
        "C3 selected rates detached from qualified C2",
    )
    rates = c2["selected_rates"]
    reports, bindings = [], {}
    for stage in C3_SPLIT_STAGES:
        attempt = stages[stage]
        planned = read_json(attempt / "planned.json")
        resource = audit_attempt(attempt, require_completed=True)
        require(
            planned["role"] == stage
            and planned["phase"] == "calibration"
            and planned["execution_commit"] == commit
            and planned["config_sha256"] == config_sha256,
            "C3 component source or role differs",
        )
        node = attempt / "node"
        report = node / "calibration-result.json"
        reports.append(report)
        index = read_json(node / "index.json")
        require(index["stage"] == stage, "C3 component allocation index differs")
        measured = read_json(report)["runs"]
        require(
            list(measured) == [row["id"] for row in index["runs"]],
            "C3 run order differs from allocation index",
        )
        leaves = {}
        for row in index["runs"]:
            identity = row["id"]
            require(id_ok(identity) and identity not in leaves, "invalid C3 leaf identity")
            leaf = node / identity
            files = (
                "input.json",
                "audit.json",
                "raw/client.json",
                "raw/engine.json",
                "raw/sources.json",
                "collector/manifest.json",
                "collector/acceptance.json",
            )
            leaves[identity] = {name: digest(leaf / name) for name in files}
            require(
                leaves[identity]["audit.json"]
                == row["audit_sha256"]
                == measured[identity]["audit_sha256"]
                and leaves[identity]["input.json"] == measured[identity]["input_sha256"]
                and row["elapsed_seconds"] == measured[identity]["elapsed_seconds"],
                "C3 leaf detached from allocation index or qualification report",
            )
        bindings[stage] = {
            "job_id": resource["job_id"],
            "release_sha256": resource["terminal_sha256"],
            "resources": {
                path.relative_to(attempt).as_posix(): digest(path)
                for path in attempt_evidence_paths(attempt)
            },
            "allocation_source_sha256": digest(node / "allocation-sources.json"),
            "leaves": leaves,
        }
    require(
        len({row["job_id"] for row in bindings.values()}) == len(C3_SPLIT_STAGES),
        "C3 components share an allocation identity",
    )
    assessment = aggregate_c3(
        config,
        reports,
        rates,
        config_sha256=config_sha256,
        original_c3=Path(history) / "c3-01/node/calibration-result.json",
    )
    require(assessment["status"] == "qualified", "C3 component did not qualify")
    return {
        "schema_version": 1,
        "execution_commit": commit,
        "d2r110_analysis_sha256": prerequisite_sha,
        "qualification": assessment,
        "components": bindings,
    }


def c3_bundle_bytes(bundle):
    return (
        json.dumps(bundle, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def c3_bundle_digest(bundle):
    return hashlib.sha256(c3_bundle_bytes(bundle)).hexdigest()
