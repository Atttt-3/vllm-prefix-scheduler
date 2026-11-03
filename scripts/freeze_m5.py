"""Independently re-audit completed calibration and freeze the formal arrival tables."""

import argparse
import json
from pathlib import Path

from m5_seal import git_identity, sealed_leaf, verify_analysis
from plan_m5_calibration import evaluate_stage

from vllm_prefix_scheduler.m5_campaign import (  # noqa: E402
    CALIBRATION,
    DIAGNOSTIC_ROLES,
    TASK,
    check_freeze,
    config_matrix,
    digest,
    read_json,
    require,
)
from vllm_prefix_scheduler.m5_contract import audit_run  # noqa: E402
from vllm_prefix_scheduler.m5_resources import audit_attempt, audit_history  # noqa: E402


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def freeze_campaign(config_path, history, commit, output):
    verify_analysis(commit, commit)
    config = read_json(config_path)
    matrix = config_matrix(config)
    audit_history(history)
    stages = {}
    for directory in history.iterdir():
        if not directory.is_dir():
            continue
        planned = read_json(directory / "planned.json")
        if (
            planned["phase"] != "calibration"
            or planned["execution_commit"] != commit
            or planned["role"] in DIAGNOSTIC_ROLES
        ):
            continue
        resource = audit_attempt(directory)
        if resource["complete"]:
            stage = planned["role"]
            require(stage not in stages, "duplicate completed calibration stage")
            stages[stage] = (directory, planned, resource["job_id"])
    require(set(stages) == set(CALIBRATION), "all six qualified M5 calibration stages required")
    output.mkdir(parents=True, exist_ok=False)
    catalog = git_identity(commit)
    reports, report_hashes = {}, {}
    config_hash = digest(config_path)
    for stage in CALIBRATION:
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
        rates = reports["C2"]["selected_rates"] if stage in CALIBRATION[2:] else None
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
    table_dir = stages["C6"][0] / "node/tables"
    require(
        {file.name for file in table_dir.iterdir()} == {f"{pair}.json" for pair in matrix},
        "C6 formal input inventory differs",
    )
    rates = {**reports["C2"]["selected_rates"], **reports["C6"]["selected_rates"]}
    interval = reports["C6"]["serial_interval_ms"]
    hashes = {}
    for identity, pair in matrix.items():
        path = table_dir / f"{identity}.json"
        table = read_json(path)
        scenario = pair["scenario"]
        require(
            table["scenario"] == scenario
            and table["seed"] == pair["seed"]
            and len(table["requests"]) == config["request_count"]
            and table["rate_rps"] == rates.get(scenario)
            and table["interval_ms"] == (interval if scenario == "serial" else None),
            f"C6 formal arrival table differs for {identity}",
        )
        hashes[identity] = digest(path)
    require(
        read_json(stages["C6"][0] / "node/allocation.json")["formal_table_sha256"] == hashes,
        "formal tables detached from the C6 allocation",
    )
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
