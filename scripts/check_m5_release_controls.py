"""Run positive and negative subprocess controls over the sealed formal originals."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from export_m5 import CONTROL_REJECTIONS, control_result, private_output
from m5_seal import ROOT

from vllm_prefix_scheduler.m5_campaign import digest, read_json, require


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def replace_copy(path, value):
    path.unlink()
    write(path, value)


def run_controls(args):
    index_path = args.review_dir / "matrix-index.json"
    index = read_json(index_path)
    request_path = args.review_dir / "review-request.json"
    request = read_json(request_path)
    require(
        request["schema_version"] == 2
        and digest(index_path) == request["matrix_index_sha256"]
        and digest(args.config) == request["config_sha256"]
        and digest(args.freeze) == request["freeze_sha256"]
        and read_json(args.config).get("measurement_schema_version") == 2,
        "review sources changed",
    )
    output = private_output(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {}

    def child(name, script, *arguments, fixtures=()):
        path = output / name
        path.mkdir(exist_ok=True)
        command = [sys.executable, str(ROOT / "scripts" / script), *map(str, arguments)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        (path / "stdout.log").write_text(result.stdout)
        (path / "stderr.log").write_text(result.stderr)
        outcome = control_result(name, result.returncode, result.stderr)
        evidence = {
            "expected": "accepted" if CONTROL_REJECTIONS[name] is None else "rejected",
            "observed": outcome,
            "returncode": result.returncode,
            "argv": command,
            "stdout_sha256": digest(path / "stdout.log"),
            "stderr_sha256": digest(path / "stderr.log"),
            "fixtures": {
                fixture.relative_to(path).as_posix(): digest(fixture) for fixture in fixtures
            },
        }
        write(path / "outcome.json", evidence)
        report[name] = evidence
        if name == "matrix_positive":
            require(
                evidence["stdout_sha256"] == request["matrix_audit_sha256"],
                "positive matrix control differs from the reviewed raw reconstruction",
            )

    def matrix(name, catalog=None):
        source = index_path
        if catalog is not None:
            path = output / name
            path.mkdir()
            source = path / "index.json"
            write(source, catalog)
        child(
            name,
            "check_m5_trace.py",
            "matrix",
            "--config",
            args.config,
            "--index",
            source,
            fixtures=() if catalog is None else (source,),
        )

    matrix("matrix_positive")
    runs = index["runs"]
    matrix("missing_pair", {"runs": [row for row in runs if row["pair_id"] != runs[0]["pair_id"]]})
    matrix("wrong_mode", {"runs": [{**runs[0], "mode": runs[1]["mode"]}, *runs[1:]]})

    first = runs[0]
    guarded = next(row for row in runs if row["mode"] == "guarded")
    pressure = next(
        row for row in runs if row["pair_id"].startswith("burst-") and row["mode"] == "native"
    )
    cold = next(row for row in runs if row["pair_id"].startswith("all-cold-"))
    for label, source, filename in (
        ("missing_field", first, "client.json"),
        ("failure_denominator", first, "client.json"),
        ("clock_domain", first, "engine.json"),
        ("execution_interval", first, "engine.json"),
        ("missing_candidate_decision", guarded, "engine.json"),
        ("duplicate_eviction", pressure, "engine.json"),
        ("missing_cold_hit_observation", cold, "client.json"),
    ):
        folder = output / label
        folder.mkdir()
        original = Path(source["run"])
        trace = read_json(original / filename)
        if label == "missing_field":
            del trace["requests"][0]["first_token_ns"]
        elif label == "failure_denominator":
            trace["requests"].pop()
        elif label == "clock_domain":
            trace["clock"]["boot_id"] += "-negative-control"
        elif label == "execution_interval":
            runner = next((row["runner"] for row in trace["batches"] if row["runner"]), None)
            require(
                runner is not None and runner["rows"], "positive control lacks a runner interval"
            )
            runner["rows"][0]["end"] += 1
        elif label == "missing_candidate_decision":
            require(trace["decisions"], "positive control lacks a candidate decision")
            trace["decisions"].pop()
        elif label == "duplicate_eviction":
            evictions = trace["evictions"]
            position = next(
                (i for i, row in enumerate(evictions) if row["stage"] == "measure"), None
            )
            require(position is not None, "positive control lacks a physical eviction")
            evictions.insert(position, evictions[position])
        else:
            completed = next(
                (row for row in trace["requests"] if row["status"] == "completed"), None
            )
            require(completed is not None, "positive all-cold control lacks a completed request")
            completed["cached_tokens"] = None
        for name in {"client.json", "engine.json", "sources.json"} - {filename}:
            (folder / name).symlink_to(original / name)
        write(folder / filename, trace)
        child(
            label,
            "check_m5_trace.py",
            "run",
            "--config",
            args.config,
            "--inputs",
            source["inputs"],
            "--run",
            folder,
            "--pair",
            source["pair_id"],
            "--mode",
            source["mode"],
            fixtures=(folder / filename,),
        )

    owner = Path(first["run"]).parents[2]
    require(owner.parent.resolve() == args.history.resolve(), "resource source outside history")
    for label in ("resource_residue", "missing_boundary", "unknown_terminal"):
        folder = output / label / owner.name
        shutil.copytree(owner, folder, copy_function=os.link)
        if label == "resource_residue":
            path = folder / "node/allocation.json"
            allocation = read_json(path)
            allocation["remaining_processes"] = ["test-residue"]
            replace_copy(path, allocation)
            fixtures = (path,)
        elif label == "missing_boundary":
            path = folder / "node/gpu-during.jsonl"
            path.unlink()
            path.write_text("")
            fixtures = [path]
            binding = folder / "node/allocation.json"
            allocation = read_json(binding)
            if "gpu_during_sha256" in allocation:
                allocation["gpu_during_sha256"] = digest(path)
                replace_copy(binding, allocation)
                fixtures.append(binding)
        else:
            terminal = next(
                (
                    path
                    for path in (
                        folder / "recovery/slurm-final.json",
                        folder / "watcher/slurm-final.json",
                    )
                    if path.is_file()
                ),
                None,
            )
            require(terminal is not None, "positive control lacks a Slurm terminal record")
            old = read_json(terminal)
            updated = {
                **old,
                "stdout": " ".join(
                    "JobState=RUNNING" if value.startswith("JobState=") else value
                    for value in old["stdout"].split()
                ),
            }
            require(updated != old, "positive control lacks a Slurm terminal state")
            replace_copy(terminal, updated)
            fixtures = [terminal]
            recovery = terminal.parent / "release.json"
            if recovery.is_file():
                recovered = read_json(recovery)
                recovered["terminal_sha256"] = digest(terminal)
                replace_copy(recovery, recovered)
                fixtures.append(recovery)
            journal = terminal.parent / "watch.jsonl"
            if journal.is_file():
                records = journal.read_text().splitlines()
                require(json.loads(records[-1]) == old, "positive control terminal journal differs")
                records[-1] = json.dumps(updated)
                journal.unlink()
                journal.write_text("\n".join(records) + "\n")
                fixtures.append(journal)
        child(label, "check_m5_resources.py", folder, fixtures=fixtures)

    source_history = output / "source_mismatch" / "attempts"
    shutil.copytree(args.history, source_history, copy_function=os.link)
    raw_sources = (
        source_history / owner.name / "node" / Path(first["run"]).parent.name / "raw/sources.json"
    )
    wrong = read_json(raw_sources)
    wrong["upstream_commit"] = "0" * 40
    replace_copy(raw_sources, wrong)
    allocation_sources = source_history / owner.name / "node/allocation-sources.json"
    replace_copy(allocation_sources, wrong)
    collector = raw_sources.parents[1] / "collector"
    manifest_path = collector / "manifest.json"
    manifest = read_json(manifest_path)
    entry = next(
        row
        for row in manifest["files"]
        if row["path"].startswith("artifacts/") and row["source"].endswith("/sources.json")
    )
    archived = collector / entry["path"]
    archived.unlink()
    shutil.copyfile(raw_sources, archived)
    entry.update(sha256=digest(archived), bytes=archived.stat().st_size)
    replace_copy(manifest_path, manifest)
    acceptance_path = collector / "acceptance.json"
    acceptance = read_json(acceptance_path)
    acceptance["manifest_sha256"] = digest(manifest_path)
    replace_copy(acceptance_path, acceptance)
    child(
        "source_mismatch",
        "export_m5.py",
        "review",
        "--config",
        args.config,
        "--freeze",
        args.freeze,
        "--history",
        source_history,
        "--analysis-commit",
        request["analysis_commit"],
        "--output",
        output / "source_mismatch/private",
        fixtures=(raw_sources, allocation_sources, archived, manifest_path, acceptance_path),
    )
    require(set(report) == set(CONTROL_REJECTIONS), "release control matrix incomplete")
    write(
        output / "controls.json",
        {
            "schema_version": 2,
            "review_request_sha256": digest(request_path),
            "analysis_commit": request["analysis_commit"],
            "matrix_index_sha256": digest(index_path),
            "controls": report,
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "freeze", "history", "review-dir", "output"):
        p.add_argument(f"--{key}", type=Path, required=True)
    args = p.parse_args()
    try:
        run_controls(args)
    except (KeyError, TypeError, ValueError, OSError, subprocess.SubprocessError) as error:
        p.exit(1, f"M5 release controls rejected: {error}\n")


if __name__ == "__main__":
    main()
