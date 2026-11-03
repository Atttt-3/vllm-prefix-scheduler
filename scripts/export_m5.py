"""Independently recompute M5 evidence, then release a reviewed numerical export."""

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from m5_c3_bundle import c3_bundle, c3_bundle_digest
from m5_seal import git_identity, sealed_leaf, verify_analysis

from vllm_prefix_scheduler.m5_campaign import (
    BASELINE,
    CALIBRATION,
    DIAGNOSTIC_ROLES,
    MODES,
    SCENARIOS,
    SPLIT_C3_ROLES,
    TASK,
    check_authorization,
    check_freeze,
    digest,
    formal_specs,
    is_hash,
    qualification_roles,
    read_json,
    require,
)
from vllm_prefix_scheduler.m5_contract import (
    STATUSES,
    bind_formal_calibration,
    validate_formal_tables,
)
from vllm_prefix_scheduler.m5_resources import (
    RESOURCE_SCHEMA_VERSION,
    TERMINAL,
    attempt_evidence_paths,
    audit_attempt,
    audit_history,
    submitted_attempts,
)

ROOT = Path(__file__).resolve().parents[1]
CRITERIA = ("A", "B", "C", "D", "E", "F")
CONTROL_REJECTIONS = {
    "matrix_positive": None,
    "missing_pair": "M5 evidence rejected: formal matrix missing, duplicate, mislabeled or out of frozen order",
    "wrong_mode": "M5 evidence rejected: formal matrix missing, duplicate, mislabeled or out of frozen order",
    "missing_field": "M5 evidence rejected: request ledger required field missing",
    "failure_denominator": "M5 evidence rejected: request ledger denominator differs",
    "clock_domain": "M5 evidence rejected: clock domain differs",
    "execution_interval": "M5 evidence rejected: runner positions differ from actual scheduled request generation",
    "missing_candidate_decision": "M5 evidence rejected: candidate decision missing for marked batch",
    "duplicate_eviction": "M5 evidence rejected: duplicate physical eviction within allocation",
    "missing_cold_hit_observation": "M5 evidence rejected: all-cold request has a positive or missing client cache observation",
    "source_mismatch": "M5 export rejected: runtime source or job mismatch",
    "resource_residue": "M5 resource gate: allocation process cleanup or timing evidence invalid",
    "missing_boundary": "M5 resource gate: before/during/after GPU checkpoint order incomplete",
    "unknown_terminal": "M5 resource gate: authoritative terminal/allocation identity differs",
}
FREEZE_FIELDS = (
    "schema_version",
    "task_id",
    "execution_commit",
    "config_sha256",
    "calibration_sha256",
    "tables",
    "rates_rps",
    "serial_interval_ms",
)
STRATA = ("all", "hot", "cold")
RATE_FIELDS = ("completion_rate", "timely_first_token_rate")
STRATUM_DISTRIBUTIONS = (
    "ttft_ns",
    "submit_ttft_ns",
    "tpot_ns",
    "enqueue_to_first_dispatch_ns",
    "whole_queue_overtakes",
    "window_overtakes",
    "whole_queue_hot_overtakes",
    "window_hot_overtakes",
)
WORK_FIELDS = (
    "prefill_tokens",
    "decode_tokens",
    "recomputed_prefill_tokens",
    "effective_tokens",
    "padded_tokens",
)
CONTRASTS = ("off-native", "prefix-native", "guarded-native", "guarded-prefix")
CONTRAST_FIELDS = (
    "request_throughput",
    "token_throughput",
    "schedule_p95_ns",
    *WORK_FIELDS,
    *(f"{stratum}.{field}" for stratum in STRATA for field in RATE_FIELDS),
    *(f"{stratum}.{field}.p95" for stratum in STRATA for field in STRATUM_DISTRIBUTIONS),
)
SCHEDULE_SCOPE = {
    "schedule_ns": "inherited_schedule_call",
}


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def numeric(value):
    require(
        value is None or type(value) is int or type(value) is float and math.isfinite(value),
        "public M5 metrics contain a nonnumeric leaf",
    )
    return value


def numbers(row, fields):
    require(type(row) is dict and set(row) == set(fields), "public M5 metric keys differ")
    return {key: numeric(row[key]) for key in fields}


def distribution(row):
    return numbers(row, ("n", "p50", "p95", "p99", "max"))


def private_output(output):
    output = output.resolve()
    require(
        not output.is_relative_to(ROOT) or output.is_relative_to(ROOT / "runs"),
        "M5 original evidence output must be private",
    )
    return output


def control_result(name, returncode, stderr):
    rejection = CONTROL_REJECTIONS[name]
    require(
        type(returncode) is int
        and returncode == (0 if rejection is None else 1)
        and stderr.rstrip("\n") == (rejection or ""),
        f"unexpected M5 release control: {name}",
    )
    return "accepted" if rejection is None else "rejected"


def public_metrics(metrics, measurement_schema):
    strata = {}
    for label in STRATA:
        row = metrics["strata"][label]
        strata[label] = {
            "counts": numbers(
                row["counts"], (*STATUSES, "planned", "submitted", "timely_first_token")
            ),
            **{key: numeric(row[key]) for key in RATE_FIELDS},
            **{key: distribution(row[key]) for key in (*STRATUM_DISTRIBUTIONS, "cached_tokens")},
        }
    public = {
        "strata": strata,
        "throughput": numbers(
            metrics["throughput"],
            (
                "window_ns",
                "completed_requests",
                "generated_tokens",
                "requests_per_s",
                "tokens_per_s",
                "fixed_window_ns",
                "fixed_requests_per_s",
                "fixed_tokens_per_s",
                "last_terminal_requests_per_s",
                "last_terminal_tokens_per_s",
            ),
        ),
        "duration_ns": numbers(
            metrics["duration_ns"],
            (
                "load",
                "warmup",
                "arrival",
                "drain",
                "measure_through_drain",
                "last_terminal_from_start",
                "max_submit_to_end",
            ),
        ),
        "submit_lag_ns": distribution(metrics["submit_lag_ns"]),
        "kv_blocks": numeric(metrics["kv_blocks"]),
        "max_outstanding": numeric(metrics["max_outstanding"]),
        "pressure": numbers(
            metrics["pressure"],
            (
                "competing_windows",
                "protected_windows",
                "cached_evictions",
                "reusable_evictions",
                "preemptions",
            ),
        ),
        "schedule_ns": distribution(metrics["schedule_ns"]),
        "stage_costs_ns": {"schedule": distribution(metrics["stage_costs_ns"]["schedule"])},
        "work": numbers(metrics["work"], WORK_FIELDS),
    }
    if measurement_schema == 2:
        hit = metrics["all_cold_zero_hit"]
        require(
            (hit is True or hit is None) and metrics["schedule_timing_scope"] == SCHEDULE_SCOPE,
            "unexpected public M5 measurement semantics",
        )
        public.update(
            all_cold_zero_hit=hit,
            schedule_envelope_ns=distribution(metrics["schedule_envelope_ns"]),
            schedule_timing_scope=SCHEDULE_SCOPE,
        )
    return public


def public_freeze(freeze, config_path):
    matrix = check_freeze(freeze, config_path, freeze["execution_commit"])
    require(set(freeze) == set(FREEZE_FIELDS), "unexpected public M5 freeze fields")
    return matrix, {key: freeze[key] for key in FREEZE_FIELDS}


def collect(config_path, freeze_path, history, analysis_commit, output):
    output = private_output(output)
    config_path, history = config_path.resolve(), history.resolve()
    freeze = read_json(freeze_path)
    execution = freeze["execution_commit"]
    matrix, _ = public_freeze(freeze, config_path)
    verify_analysis(execution, analysis_commit)
    config = read_json(config_path)
    calibration_roles = qualification_roles(config)
    releases = audit_history(history)
    tables, stages, candidates = None, {}, {}
    private_releases = []
    for path in submitted_attempts(history):
        planned = read_json(path / "planned.json")
        resource = audit_attempt(path)
        resource_paths = attempt_evidence_paths(path)
        allocation_sources = path / "node/allocation-sources.json"
        if allocation_sources.is_file():
            resource_paths = [*resource_paths, allocation_sources]
        private_releases.append(
            {
                "attempt": path.name,
                "role": planned["role"],
                "phase": planned["phase"],
                "state": resource["state"],
                "exit_code": resource["exit_code"],
                "elapsed_seconds": resource["elapsed_seconds"],
                "requested_minutes": 20,
                "allocation_sha256": digest(path / "node/allocation.json"),
                "terminal_sha256": resource["terminal_sha256"],
                "authorization_sha256": digest(path / "authorization.json"),
                "resource_files_sha256": {
                    item.relative_to(path).as_posix(): digest(item) for item in resource_paths
                },
            }
        )
        if not resource["complete"] or planned["execution_commit"] != execution:
            continue
        phase, role = planned["phase"], planned["role"]
        if config.get("measurement_schema_version") == 2 and role not in DIAGNOSTIC_ROLES:
            require(
                planned.get("resource_schema_version") == RESOURCE_SCHEMA_VERSION,
                "qualified M5 evidence requires the current resource schema",
            )
        freeze_arg = freeze_path if phase == "formal" else None
        require(
            role
            in check_authorization(
                read_json(path / "authorization.json"),
                config_path,
                execution,
                phase,
                freeze_path=freeze_arg,
            ),
            "completed role is outside its authorized scope",
        )
        if phase == "calibration":
            if role in DIAGNOSTIC_ROLES:
                continue
            require(role not in stages, "duplicate qualified calibration stage")
            report = path / "node/calibration-result.json"
            require(
                read_json(report)["status"] == "qualified", "calibration report is not qualified"
            )
            if role not in SPLIT_C3_ROLES:
                require(
                    digest(report) == freeze["calibration_sha256"][role],
                    "calibration report is outside the formal freeze",
                )
            stages[role] = path
            if role == "C6":
                tables = path / "node/tables"
        else:
            require(planned["auxiliary_sha256"] == digest(freeze_path), "formal run freeze differs")
            require(role not in candidates, "duplicate completed formal pair")
            candidates[role] = (path, planned, resource["job_id"])
    require(set(stages) == set(calibration_roles) and tables is not None, "calibration incomplete")
    if calibration_roles != CALIBRATION:
        bundle = c3_bundle(config_path, history, stages, execution)
        require(
            c3_bundle_digest(bundle) == freeze["calibration_sha256"]["C3"],
            "split C3 original evidence is outside the formal freeze",
        )
    require(set(candidates) == set(matrix), "formal matrix incomplete")
    bind_formal_calibration(freeze, stages["C2"], stages["C6"])
    validate_formal_tables(tables, config, matrix, freeze)
    output.mkdir(parents=True, exist_ok=False)
    catalog = git_identity(execution)
    inventory = []
    for role in matrix:
        path, planned, job = candidates[role]
        table = tables / f"{role}.json"
        index = read_json(path / "node/index.json")
        specs = formal_specs(config, freeze, role)
        require(
            index["schema_version"] == 1
            and index["stage"] == role
            and len(index["runs"]) == len(specs),
            "allocation matrix index differs",
        )
        for actual, spec in zip(index["runs"], specs, strict=True):
            require(
                actual["pair_id"] == role
                and all(actual[key] == value for key, value in spec.items()),
                "mode order, seed, rate or scenario differs from freeze",
            )
            leaf = sealed_leaf(
                path,
                planned["remote_attempt"],
                actual,
                table,
                config_path,
                config,
                job,
                catalog,
                execution,
            )
            collector = Path(leaf["run"]).parent / "collector"
            inventory.append(
                {
                    **leaf,
                    "audit_sha256": digest(leaf["audit"]),
                    "collector_acceptance_sha256": digest(collector / "acceptance.json"),
                }
            )
    index_path = output / "matrix-index.json"
    report_path = output / "matrix-audit.json"
    write(index_path, {"runs": inventory})
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_m5_trace.py"),
            "matrix",
            "--config",
            str(config_path),
            "--index",
            str(index_path),
            "--output",
            str(report_path),
        ],
        check=True,
    )
    review = {
        "schema_version": 2,
        "task_id": TASK,
        "baseline_commit": BASELINE,
        "execution_commit": execution,
        "analysis_commit": analysis_commit,
        "freeze_sha256": digest(freeze_path),
        "config_sha256": digest(config_path),
        "matrix_index_sha256": digest(index_path),
        "matrix_audit_sha256": digest(report_path),
        "releases": private_releases,
        "elapsed_seconds": sum(row["elapsed_seconds"] for row in releases),
        "requested_minutes": sum(row["requested_minutes"] for row in releases),
    }
    write(output / "review-request.json", review)


def verify_controls(controls_path, request, request_sha):
    controls = read_json(controls_path)
    require(
        controls["schema_version"] == 2
        and controls["review_request_sha256"] == request_sha
        and controls["matrix_index_sha256"] == request["matrix_index_sha256"]
        and controls["analysis_commit"] == request["analysis_commit"]
        and set(controls["controls"]) == set(CONTROL_REJECTIONS),
        "real subprocess control results incomplete",
    )
    controls_dir = controls_path.parent.resolve()
    for name, row in controls["controls"].items():
        folder = controls_dir / name
        require(folder.resolve() == folder, "control directory escaped")
        require(read_json(folder / "outcome.json") == row, "control outcome changed")
        for stream in ("stdout", "stderr"):
            log = folder / f"{stream}.log"
            require(
                log.resolve() == log and digest(log) == row[f"{stream}_sha256"],
                "control log changed",
            )
        observed = control_result(name, row["returncode"], (folder / "stderr.log").read_text())
        require(row["expected"] == row["observed"] == observed, "control outcome differs")
        if observed == "accepted":
            require(
                row["stdout_sha256"] == request["matrix_audit_sha256"],
                "positive matrix control differs from the reviewed raw reconstruction",
            )
        fixtures = row["fixtures"]
        require(
            type(fixtures) is dict and bool(fixtures) == (observed == "rejected"),
            "control fixture inventory incomplete",
        )
        for relative, sha in fixtures.items():
            fixture = folder / relative
            require(
                fixture.resolve() == fixture
                and fixture.is_relative_to(folder)
                and digest(fixture) == sha,
                "control mutation changed",
            )


def public_allocations(request, matrix):
    public = []
    for row in request["releases"]:
        phase, role = row["phase"], row["role"]
        require(
            (
                phase == "calibration"
                and role in (*CALIBRATION, *SPLIT_C3_ROLES, *DIAGNOSTIC_ROLES)
                or phase == "formal"
                and role in matrix
            )
            and row["state"] in TERMINAL
            and re.fullmatch(r"\d+:\d+", row["exit_code"]) is not None
            and is_hash(row["allocation_sha256"])
            and is_hash(row["terminal_sha256"]),
            "unexpected public M5 allocation metadata",
        )
        public.append(
            {
                **{
                    key: row[key]
                    for key in (
                        "phase",
                        "role",
                        "state",
                        "exit_code",
                        "allocation_sha256",
                        "terminal_sha256",
                    )
                },
                **{key: numeric(row[key]) for key in ("elapsed_seconds", "requested_minutes")},
            }
        )
    return public


def public_results(report, matrix, measurement_schema):
    require(report["status"] == "valid", "matrix audit rejected")
    require(
        [pair["pair_id"] for pair in report["pairs"]] == list(matrix),
        "public matrix identities differ",
    )
    pairs = []
    for pair in report["pairs"]:
        pair_id = pair["pair_id"]
        frozen = matrix[pair_id]
        require(pair["scenario"] == frozen["scenario"], "public pair scenario differs")
        modes = {}
        for mode in MODES:
            row = pair["runs"][mode]
            require(
                row["seed"] == frozen["seed"] and is_hash(row["input_sha256"]),
                "public run provenance differs",
            )
            modes[mode] = {
                "seed": frozen["seed"],
                "input_sha256": row["input_sha256"],
                **public_metrics(row["metrics"], measurement_schema),
            }
        pairs.append(
            {
                "pair_id": pair_id,
                "scenario": frozen["scenario"],
                "modes": modes,
                "contrasts": {
                    key: numbers(pair["contrasts"][key], CONTRAST_FIELDS) for key in CONTRASTS
                },
            }
        )
    scenarios = {}
    for scenario in SCENARIOS:
        scenarios[scenario] = {}
        for contrast in CONTRASTS:
            row = report["scenarios"][scenario][contrast]
            require(set(row) == set(CONTRAST_FIELDS), "public M5 contrast keys differ")
            scenarios[scenario][contrast] = {
                key: numbers(
                    row[key], ("n", "median", "min", "max", "positive", "negative", "equal")
                )
                for key in CONTRAST_FIELDS
            }
    return {"pairs": pairs, "scenarios": scenarios}


def release(
    config_path,
    freeze_path,
    history,
    review_dir,
    decision_path,
    approved_decision_sha,
    controls_path,
    output,
):
    output = output.absolute()
    require(
        not output.exists() and not output.is_symlink(), "public export destination already exists"
    )
    review_dir = private_output(review_dir)
    request_path = review_dir / "review-request.json"
    request_sha = digest(request_path)
    request = read_json(request_path)
    require(
        is_hash(approved_decision_sha) and digest(decision_path) == approved_decision_sha,
        "approved independent decision digest differs",
    )
    decision = read_json(decision_path)
    report_file = review_dir / "matrix-audit.json"
    index_file = review_dir / "matrix-index.json"
    require(
        request["schema_version"] == 2
        and request["task_id"] == TASK
        and request["baseline_commit"] == BASELINE
        and digest(freeze_path) == request["freeze_sha256"]
        and digest(config_path) == request["config_sha256"]
        and digest(report_file) == request["matrix_audit_sha256"]
        and digest(index_file) == request["matrix_index_sha256"],
        "review sources changed",
    )
    controls_sha = digest(controls_path)
    require(
        digest(decision_path) == approved_decision_sha
        and decision["review_request_sha256"] == request_sha
        and type(decision["reviewer"]) is str
        and decision["reviewer"].strip()
        and decision["criteria"] == dict.fromkeys(CRITERIA, "passed")
        and decision["controls_sha256"] == controls_sha,
        "complete independent A-F review is required",
    )
    verify_controls(controls_path, request, request_sha)
    require(digest(controls_path) == controls_sha, "control manifest changed")
    config, frozen = read_json(config_path), read_json(freeze_path)
    matrix, freeze = public_freeze(frozen, config_path)
    require(
        digest(config_path) == request["config_sha256"]
        and digest(freeze_path) == request["freeze_sha256"],
        "review sources changed",
    )
    with TemporaryDirectory(prefix=".m5-rebuild-", dir=review_dir.parent) as temp:
        rebuilt = Path(temp) / "review"
        collect(config_path, freeze_path, history, request["analysis_commit"], rebuilt)
        require(
            digest(rebuilt / request_path.name) == request_sha
            and digest(rebuilt / index_file.name) == request["matrix_index_sha256"]
            and digest(rebuilt / report_file.name) == request["matrix_audit_sha256"],
            "fresh reconstruction differs from independent review",
        )
        report = read_json(rebuilt / report_file.name)
    measurement_schema = config.get("measurement_schema_version", 1)
    require(
        type(measurement_schema) is int and measurement_schema in (1, 2),
        "unknown public M5 measurement schema",
    )
    summary = {
        "schema_version": 2,
        "task_id": TASK,
        "measurement_schema_version": measurement_schema,
        "results": public_results(report, matrix, measurement_schema),
        "resource_totals": {
            key: numeric(request[key]) for key in ("elapsed_seconds", "requested_minutes")
        },
        "criteria": dict.fromkeys(CRITERIA, "passed"),
    }
    provenance = {
        **{
            key: request[key]
            for key in (
                "task_id",
                "baseline_commit",
                "execution_commit",
                "analysis_commit",
                "freeze_sha256",
                "config_sha256",
                "matrix_audit_sha256",
            )
        },
        "freeze": freeze,
        "controls_sha256": controls_sha,
        "review_sha256": approved_decision_sha,
        "allocations": public_allocations(request, matrix),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".m5-export-", dir=output.parent) as temp:
        staging = Path(temp) / "public"
        staging.mkdir()
        write(staging / "summary.json", summary)
        write(staging / "provenance.json", provenance)
        require(
            not output.exists() and not output.is_symlink(),
            "public export destination already exists",
        )
        staging.rename(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    review = commands.add_parser("review")
    review.add_argument("--history", required=True, type=Path)
    review.add_argument("--analysis-commit", required=True)
    publish = commands.add_parser("release")
    publish.add_argument("--history", required=True, type=Path)
    publish.add_argument("--review-dir", required=True, type=Path)
    publish.add_argument("--decision", required=True, type=Path)
    publish.add_argument("--approved-decision-sha256", required=True)
    publish.add_argument("--controls", required=True, type=Path)
    for sub in (review, publish):
        sub.add_argument("--config", required=True, type=Path)
        sub.add_argument("--freeze", required=True, type=Path)
        sub.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    try:
        if args.command == "review":
            collect(args.config, args.freeze, args.history, args.analysis_commit, args.output)
        else:
            release(
                args.config,
                args.freeze,
                args.history,
                args.review_dir,
                args.decision,
                args.approved_decision_sha256,
                args.controls,
                args.output,
            )
    except (KeyError, TypeError, ValueError, OSError, subprocess.SubprocessError) as error:
        p.exit(1, f"M5 export rejected: {error}\n")


if __name__ == "__main__":
    main()
