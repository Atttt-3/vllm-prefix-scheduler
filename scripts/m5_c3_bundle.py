"""Bind the logical C3 qualification to both same-node job inventories."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from check_m5_d1 import compare_pair
from check_m5_d2 import PAIRS, RUNS, compare_thermal
from plan_m5_calibration import C3_SPLIT_STAGES, aggregate_c3

from vllm_prefix_scheduler.m5_campaign import (
    D2R110_OFFICIAL_ANALYSIS_SHA256,
    D2R110_ROLE,
    check_authorization,
    digest,
    id_ok,
    qualification_authorization_version,
    qualification_batch,
    qualification_roles,
    read_json,
    require,
)
from vllm_prefix_scheduler.m5_d1 import C3_BLOCKED_SHA, D1_INPUT_SHA
from vllm_prefix_scheduler.m5_d2 import D2R110_PROPOSAL_SHA
from vllm_prefix_scheduler.m5_resources import (
    attempt_evidence_paths,
    audit_attempt,
    audit_history,
    receipt_job,
    submitted_attempts,
)

ROOT = Path(__file__).resolve().parents[1]
HISTORY_FIELDS = (
    "job_id",
    "role",
    "phase",
    "state",
    "exit_code",
    "slurm_start",
    "slurm_end",
    "requested_minutes",
    "max_attempts",
    "terminal_sha256",
)


def verify_d2r110_qualification(history, authorization):
    report_path = history.parent / "d2-r110-review/analysis.json"
    report_hash = authorization["d2r110_analysis_sha256"]
    require(
        digest(report_path) == report_hash == D2R110_OFFICIAL_ANALYSIS_SHA256,
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


def qualification_history_sha256(rows):
    """Canonical digest of an ordered, already audited release prefix."""
    summary = [{key: row[key] for key in HISTORY_FIELDS} for row in rows]
    content = json.dumps(
        summary, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False
    )
    return hashlib.sha256(content.encode()).hexdigest()


def replay_predecessor(attempt, predecessor):
    """Run the frozen diagnostic auditor using a temporary local clone."""
    with TemporaryDirectory(prefix="m5-qualification-review-") as temp:
        scratch = Path(temp)
        checkout = scratch / "checkout"
        output = scratch / "replayed.json"
        try:
            subprocess.run(
                ["git", "clone", "--shared", "--no-checkout", "--quiet", str(ROOT), str(checkout)],
                check=True,
                capture_output=True,
                timeout=60,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(checkout),
                    "-c",
                    "core.hooksPath=/dev/null",
                    "checkout",
                    "--detach",
                    "--quiet",
                    predecessor["execution_commit"],
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
            proposal = checkout / "configs" / predecessor["proposal_file"]
            require(
                digest(proposal) == predecessor["proposal_sha256"],
                "qualification predecessor proposal differs from its execution source",
            )
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(checkout / "scripts/check_m5_d2.py"),
                    "--attempt",
                    str(attempt.resolve()),
                    "--proposal",
                    str(proposal),
                    "--output",
                    str(output),
                ],
                cwd=checkout,
                check=True,
                capture_output=True,
                timeout=300,
            )
        except subprocess.SubprocessError as error:
            raise ValueError("frozen qualification predecessor raw re-audit failed") from error
        require(
            digest(output) == predecessor["analysis_sha256"],
            "qualification predecessor raw replay differs from the independently approved digest",
        )
        return read_json(output), read_json(proposal)


def verify_same_condition_report(report, plan, predecessor, resource):
    frozen_path = ROOT / "configs/m5-d2r110.json"
    require(digest(frozen_path) == D2R110_PROPOSAL_SHA, "original stability protocol differs")
    frozen = read_json(frozen_path)
    require(
        plan["role"] == predecessor["role"]
        and all(plan[key] == frozen[key] for key in ("target", "conditioning", "stability_gate"))
        and report["role"] == predecessor["role"]
        and report["proposal_sha256"] == predecessor["proposal_sha256"]
        and report["input_sha256"] == D1_INPUT_SHA
        and report["retained_original_C3"]
        == {"status": "blocked", "stage_result_sha256": C3_BLOCKED_SHA}
        and report["execution_commit"]
        == report["analysis_commit"]
        == predecessor["execution_commit"]
        and report["resource_release"] == json.loads(json.dumps(resource, default=str)),
        "qualification predecessor does not bind the original same-condition evidence",
    )
    runs, pairs = report["runs"], report["pairs"]
    require(
        report["status"] == "diagnostic-complete"
        and tuple(runs) == RUNS
        and runs[RUNS[0]]["status"] == "target-frozen"
        and runs[RUNS[0]]["prelude"]["verified"] is True
        and report["verified_thermal_target"] is not None
        and tuple((row["reference"], row["observed"]) for row in pairs) == PAIRS,
        "qualification predecessor lacks the frozen complete three-pair matrix",
    )
    for name in RUNS[1:]:
        run = runs[name]
        counts = run["metrics"]["strata"]["all"]["counts"]
        require(
            run["status"] == "admitted"
            and run["prelude"]["verified"] is True
            and run["pre_t0_verified"] is True
            and run["endpoint_evidence_complete"] is True
            and run["counts"] == counts
            and all(
                counts[key] == 60
                for key in ("planned", "submitted", "completed", "timely_first_token")
            ),
            "qualification predecessor lacks a complete admitted endpoint leaf",
        )
    gates = {
        "overhead_ttft_ns": 2_000_000,
        "overhead_ttft_rel": 0.05,
        "overhead_throughput_rel": 0.05,
    }
    for pair in pairs:
        reference, observed = runs[pair["reference"]], runs[pair["observed"]]
        thermal = compare_thermal(
            reference["in_window"], observed["in_window"], plan["conditioning"]
        )
        performance = compare_pair(reference["metrics"], observed["metrics"], gates)
        require(
            thermal["comparable"] is True
            and pair["thermal"] == thermal
            and performance["within_tolerance"] is True
            and all(pair["performance"].get(key) == value for key, value in performance.items()),
            "qualification predecessor did not meet an original thermal or paired performance gate",
        )
    require(
        report["decision"]
        == {
            "thermal_control": "verified",
            "observed_endpoint_stability": "stable",
            "endpoint_stability_under_control": "stable",
            "qualification_started": False,
        },
        "qualification predecessor summary disagrees with recomputed pair evidence",
    )


def verify_split_qualification(
    config_path, history, authorization_path, *, audits=None, require_complete=False
):
    """Bind the current split batch to a release prefix and freshly re-audited predecessor."""
    config, history = read_json(config_path), Path(history)
    authorization = read_json(authorization_path)
    if qualification_authorization_version(config) == 1:
        return {"d2r110_analysis_sha256": verify_d2r110_qualification(history, authorization)}
    commit = authorization["execution_commit"]
    roles = check_authorization(authorization, config_path, commit, "calibration")
    batch = qualification_batch(authorization, config)
    releases = audit_history(history) if audits is None else audits
    calibration = [row for row in releases if row["phase"] == "calibration"]
    prior_count = batch["prior_allocation_count"]
    prior, suffix = calibration[:prior_count], calibration[prior_count:]
    require(
        len(prior) == prior_count
        and qualification_history_sha256(prior) == batch["prior_history_sha256"]
        and len(suffix) <= batch["new_max_attempts"],
        "qualification release prefix or new-attempt budget differs from authorization",
    )
    attempts = submitted_attempts(history)
    by_job = {receipt_job(read_json(path / "submission.json")): path for path in attempts}
    predecessor = batch["predecessor"]
    attempt = history / predecessor["attempt"]
    planned = read_json(attempt / "planned.json")
    resource = audit_attempt(attempt, require_completed=True)
    require(
        resource in prior
        and resource["role"] == predecessor["role"]
        and by_job[resource["job_id"]] == attempt
        and planned["execution_commit"] == predecessor["execution_commit"]
        and planned["auxiliary_sha256"] == predecessor["proposal_sha256"],
        "qualification predecessor differs from its approved prior release",
    )
    authorization_sha = digest(authorization_path)
    next_role = 0
    for row in suffix:
        released = by_job[row["job_id"]]
        record = read_json(released / "planned.json")
        require(
            next_role < len(roles)
            and row["role"] == roles[next_role]
            and row["slurm_start"] >= resource["slurm_end"]
            and record["execution_commit"] == commit
            and record["config_sha256"] == authorization["config_sha256"]
            and record["authorization_sha256"] == authorization_sha,
            "qualification suffix contains an out-of-order or different batch allocation",
        )
        next_role += row["complete"]
    require(
        not require_complete or next_role == len(roles),
        "qualification batch does not have all released roles",
    )
    review = history.parent / predecessor["review_directory"] / "analysis.json"
    require(
        digest(review) == predecessor["analysis_sha256"],
        "qualification predecessor is detached from independent approval",
    )
    replayed, proposal = replay_predecessor(attempt, predecessor)
    require(
        replayed == read_json(review),
        "qualification predecessor raw review differs from independent approval",
    )
    verify_same_condition_report(replayed, proposal, predecessor, resource)
    return {"qualification_batch": batch, "qualification_authorization_sha256": authorization_sha}


def c3_bundle(config_path, history, stages, commit):
    config = read_json(config_path)
    config_sha256 = digest(config_path)
    auth_paths = [stages[role] / "authorization.json" for role in qualification_roles(config)]
    version = qualification_authorization_version(config)
    if version == 1:
        require(
            len({read_json(path)["d2r110_analysis_sha256"] for path in auth_paths}) == 1,
            "qualification roles bind different D2-R110 reviews",
        )
    else:
        require(
            len({digest(path) for path in auth_paths}) == 1,
            "qualification roles bind different batch authorizations",
        )
    prerequisite = verify_split_qualification(
        config_path, history, auth_paths[0], require_complete=True
    )
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
        "schema_version": version,
        "execution_commit": commit,
        **prerequisite,
        "qualification": assessment,
        "components": bindings,
    }


def c3_bundle_bytes(bundle):
    return (
        json.dumps(bundle, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def c3_bundle_digest(bundle):
    return hashlib.sha256(c3_bundle_bytes(bundle)).hexdigest()
