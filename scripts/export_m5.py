"""Independently recompute M5 evidence, then release a reviewed numerical export."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from m5_seal import git_identity, sealed_leaf, verify_analysis

from vllm_prefix_scheduler.m5_campaign import (
    BASELINE,
    CALIBRATION,
    MODES,
    TASK,
    check_authorization,
    check_freeze,
    digest,
    formal_specs,
    read_json,
    require,
)
from vllm_prefix_scheduler.m5_resources import audit_attempt, audit_history

ROOT = Path(__file__).resolve().parents[1]
CRITERIA = ("A", "B", "C", "D", "E", "F")
METRICS = (
    "strata",
    "throughput",
    "duration_ns",
    "submit_lag_ns",
    "kv_blocks",
    "max_outstanding",
    "pressure",
    "schedule_ns",
    "stage_costs_ns",
    "work",
)
CONTROLS = {
    "matrix_positive",
    "missing_pair",
    "wrong_mode",
    "missing_field",
    "failure_denominator",
    "source_mismatch",
    "resource_residue",
    "missing_boundary",
    "unknown_terminal",
}


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def numeric(value):
    if value is None or type(value) is int:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    require(type(value) is dict, "public M5 metrics contain a nonnumeric leaf")
    return {key: numeric(item) for key, item in value.items()}


def collect(config_path, freeze_path, history, analysis_commit, output):
    freeze = read_json(freeze_path)
    execution = freeze["execution_commit"]
    matrix = check_freeze(freeze, config_path, execution)
    verify_analysis(execution, analysis_commit)
    config = read_json(config_path)
    releases = audit_history(history)
    tables, stages, candidates = None, {}, {}
    private_releases = []
    for path in sorted(history.iterdir()):
        if not path.is_dir():
            continue
        planned = read_json(path / "planned.json")
        resource = audit_attempt(path)
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
                "terminal_sha256": digest(path / "watcher/slurm-final.json"),
                "authorization_sha256": digest(path / "authorization.json"),
            }
        )
        if not resource["complete"] or planned["execution_commit"] != execution:
            continue
        phase, role = planned["phase"], planned["role"]
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
            require(role not in stages, "duplicate qualified calibration stage")
            report = path / "node/calibration-result.json"
            require(
                digest(report) == freeze["calibration_sha256"][role]
                and read_json(report)["status"] == "qualified",
                "calibration report is outside the formal freeze",
            )
            stages[role] = report
            if role == "C6":
                tables = path / "node/tables"
        else:
            require(planned["auxiliary_sha256"] == digest(freeze_path), "formal run freeze differs")
            require(role not in candidates, "duplicate completed formal pair")
            candidates[role] = (path, planned, resource["job_id"])
    require(set(stages) == set(CALIBRATION) and tables is not None, "calibration incomplete")
    require(set(candidates) == set(matrix), "formal matrix incomplete")
    output.mkdir(parents=True, exist_ok=False)
    catalog = git_identity(execution)
    inventory = []
    for role in matrix:
        path, planned, job = candidates[role]
        table = tables / f"{role}.json"
        require(digest(table) == freeze["tables"][role], "frozen table was modified")
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
            inventory.append(
                sealed_leaf(
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


def release(config_path, freeze_path, review_dir, decision_path, controls_path, output):
    request = read_json(review_dir / "review-request.json")
    decision = read_json(decision_path)
    controls = read_json(controls_path)
    report_file = review_dir / "matrix-audit.json"
    index_file = review_dir / "matrix-index.json"
    require(
        request["task_id"] == TASK
        and request["baseline_commit"] == BASELINE
        and digest(freeze_path) == request["freeze_sha256"]
        and digest(config_path) == request["config_sha256"]
        and digest(report_file) == request["matrix_audit_sha256"]
        and digest(index_file) == request["matrix_index_sha256"],
        "review sources changed",
    )
    verify_analysis(request["execution_commit"], request["analysis_commit"])
    require(
        decision["review_request_sha256"] == digest(review_dir / "review-request.json")
        and type(decision["reviewer"]) is str
        and decision["reviewer"].strip()
        and decision["criteria"] == dict.fromkeys(CRITERIA, "passed")
        and decision["controls_sha256"] == digest(controls_path),
        "complete independent A-F review is required",
    )
    require(
        controls["matrix_index_sha256"] == request["matrix_index_sha256"]
        and controls["analysis_commit"] == request["analysis_commit"]
        and set(controls["controls"]) == CONTROLS
        and all(row["observed"] == row["expected"] for row in controls["controls"].values()),
        "real subprocess control results incomplete",
    )
    for bound in read_json(index_file)["runs"]:
        require(digest(bound["inputs"]) == bound["input_sha256"], "raw input changed since review")
        require(
            all(
                digest(Path(bound["run"]) / name) == value
                for name, value in bound["raw_hashes"].items()
            ),
            "raw trace changed since review",
        )
    report = read_json(report_file)
    require(report["status"] == "valid", "matrix audit rejected")
    pairs = []
    for pair in report["pairs"]:
        modes = {}
        for mode in MODES:
            original = pair["runs"][mode]
            modes[mode] = {
                "seed": original["seed"],
                "input_sha256": original["input_sha256"],
                **{key: numeric(original["metrics"][key]) for key in METRICS},
            }
        pairs.append(
            {
                "pair_id": pair["pair_id"],
                "scenario": pair["scenario"],
                "modes": modes,
                "contrasts": numeric(pair["contrasts"]),
            }
        )
    output.mkdir(parents=True, exist_ok=False)
    write(
        output / "summary.json",
        {
            "schema_version": 1,
            "task_id": TASK,
            "results": {"pairs": pairs, "scenarios": numeric(report["scenarios"])},
            "resource_totals": {
                key: request[key] for key in ("elapsed_seconds", "requested_minutes")
            },
            "criteria": decision["criteria"],
        },
    )
    write(
        output / "provenance.json",
        {
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
            "freeze": read_json(freeze_path),
            "controls_sha256": digest(controls_path),
            "review_sha256": digest(decision_path),
            "allocations": [
                {
                    key: row[key]
                    for key in (
                        "phase",
                        "role",
                        "state",
                        "exit_code",
                        "elapsed_seconds",
                        "requested_minutes",
                        "allocation_sha256",
                        "terminal_sha256",
                    )
                }
                for row in request["releases"]
            ],
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    review = commands.add_parser("review")
    review.add_argument("--history", required=True, type=Path)
    review.add_argument("--analysis-commit", required=True)
    publish = commands.add_parser("release")
    publish.add_argument("--review-dir", required=True, type=Path)
    publish.add_argument("--decision", required=True, type=Path)
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
                args.config, args.freeze, args.review_dir, args.decision, args.controls, args.output
            )
    except (KeyError, TypeError, ValueError, OSError, subprocess.SubprocessError) as error:
        p.exit(1, f"M5 export rejected: {error}\n")


if __name__ == "__main__":
    main()
