"""Independently re-audit completed calibration and freeze the formal arrival tables."""

import argparse
import json
from pathlib import Path

from m5_c3_bundle import c3_bundle, c3_bundle_bytes, c3_bundle_digest
from m5_seal import git_identity, sealed_leaf, verify_analysis
from plan_m5_calibration import evaluate_stage

from vllm_prefix_scheduler.m5_campaign import (  # noqa: E402
    DIAGNOSTIC_ROLES,
    TASK,
    check_freeze,
    config_matrix,
    digest,
    qualification_roles,
    read_json,
    require,
)
from vllm_prefix_scheduler.m5_contract import audit_run, validate_formal_tables  # noqa: E402
from vllm_prefix_scheduler.m5_resources import (  # noqa: E402
    audit_attempt,
    audit_history,
    submitted_attempts,
)


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def freeze_campaign(config_path, history, commit, output):
    verify_analysis(commit, commit)
    config = read_json(config_path)
    matrix = config_matrix(config)
    roles = qualification_roles(config)
    audit_history(history)
    stages = {}
    for directory in submitted_attempts(history):
        planned = read_json(directory / "planned.json")
        if (
            planned["phase"] != "calibration"
            or planned["execution_commit"] != commit
            or planned["role"] in DIAGNOSTIC_ROLES
        ):
            continue
        resource = audit_attempt(directory)
        if resource["complete"]:
            require(
                planned.get("resource_schema_version", 1)
                == config.get("resource_schema_version", 1),
                "completed calibration resource schema differs from freeze",
            )
            stage = planned["role"]
            require(stage not in stages, "duplicate completed calibration stage")
            stages[stage] = (directory, planned, resource["job_id"])
    require(set(stages) == set(roles), "complete qualified M5 calibration roles required")
    output.mkdir(parents=True, exist_ok=False)
    catalog = git_identity(commit)
    reports, report_hashes = {}, {}
    config_hash = digest(config_path)
    for stage in roles:
        directory, planned, job = stages[stage]
        node = directory / "node"
        index = read_json(node / "index.json")
        private = output / stage
        private.mkdir()
        for row in index["runs"]:
            table = node / row["id"] / "input.json"
            bound = sealed_leaf(
                directory,
                planned["remote_attempt"],
                row,
                table,
                config_path,
                config,
                job,
                catalog,
                commit,
            )
            report = audit_run(
                config_path,
                table,
                Path(bound["run"]),
                pair_id=row["id"],
                mode=row["mode"],
                phase="diagnostic" if stage == "C1" else "calibration",
            )
            require(
                report == read_json(bound["audit"]), "on-node calibration audit differs from replay"
            )
            row.update(inputs=bound["inputs"], run=bound["run"], audit=bound["audit"])
        write(private / "index.json", index)
        rates = reports["C2"]["selected_rates"] if stage not in ("C1", "C2") else None
        dress = [reports["C4"], reports["C5"]] if stage == "C6" else ()
        recomputed = evaluate_stage(
            config, stage, private / "index.json", rates, dress, config_sha256=config_hash
        )
        recomputed["config_sha256"] = config_hash
        result = node / "calibration-result.json"
        require(
            recomputed["status"] == "qualified" and recomputed == read_json(result),
            f"independent calibration result differs for {stage}",
        )
        write(private / "reviewed-result.json", recomputed)
        reports[stage], report_hashes[stage] = recomputed, digest(result)
    if "C3" not in report_hashes:
        bound = c3_bundle(
            config_path, history, {stage: row[0] for stage, row in stages.items()}, commit
        )
        with (output / "c3-bundle.json").open("xb") as stream:
            stream.write(c3_bundle_bytes(bound))
        for stage in ("C3-burst", "C3-mixed"):
            del report_hashes[stage]
        report_hashes["C3"] = c3_bundle_digest(bound)
    c6 = stages["C6"][0] / "node"
    hashes = read_json(c6 / "allocation.json")["formal_table_sha256"]
    require(type(hashes) is dict, "C6 formal table hashes missing")
    rates = {**reports["C2"]["selected_rates"], **reports["C6"]["selected_rates"]}
    interval = reports["C6"]["serial_interval_ms"]
    freeze = {
        "schema_version": 1,
        "task_id": TASK,
        "execution_commit": commit,
        "config_sha256": config_hash,
        "calibration_sha256": report_hashes,
        "tables": hashes,
        "rates_rps": rates,
        "serial_interval_ms": interval,
    }
    check_freeze(freeze, config_path, commit)
    validate_formal_tables(c6 / "tables", config, matrix, freeze)
    write(output / "formal-freeze.json", freeze)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        freeze_campaign(args.config, args.history, args.execution_commit, args.output)
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.exit(1, f"M5 freeze rejected: {error}\n")


if __name__ == "__main__":
    main()
