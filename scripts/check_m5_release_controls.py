"""Run positive and negative subprocess controls over the sealed formal originals."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from export_m5 import CONTROLS
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
    request = read_json(args.review_dir / "review-request.json")
    require(digest(index_path) == request["matrix_index_sha256"], "review index changed")
    output = args.output.resolve()
    require(
        not output.is_relative_to(ROOT) or output.is_relative_to(ROOT / "runs"),
        "controls are private",
    )
    output.mkdir(parents=True, exist_ok=False)
    report = {}

    def child(name, expected, script, *arguments, reason=None, derived=None):
        path = output / name
        path.mkdir(exist_ok=True)
        command = [sys.executable, str(ROOT / "scripts" / script), *map(str, arguments)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        (path / "stdout.log").write_text(result.stdout)
        (path / "stderr.log").write_text(result.stderr)
        outcome = "accepted" if result.returncode == 0 else "rejected"
        evidence = {
            "expected": expected,
            "observed": outcome,
            "returncode": result.returncode,
            "argv": command,
            "stdout_sha256": digest(path / "stdout.log"),
            "stderr_sha256": digest(path / "stderr.log"),
            "derived_sha256": digest(derived) if derived is not None and derived.exists() else None,
        }
        write(path / "outcome.json", evidence)
        report[name] = evidence
        require(
            expected == outcome and (reason is None or reason in result.stderr),
            f"unexpected M5 release control: {name}",
        )

    def matrix(name, catalog=None, reason=None):
        source = index_path
        if catalog is not None:
            path = output / name
            path.mkdir()
            source = path / "index.json"
            write(source, catalog)
        child(
            name,
            "accepted" if catalog is None else "rejected",
            "check_m5_trace.py",
            "matrix",
            "--config",
            args.config,
            "--index",
            source,
            reason=reason,
            derived=source,
        )

    matrix("matrix_positive")
    runs = index["runs"]
    matrix("missing_pair", {"runs": [row for row in runs if row["pair_id"] != runs[0]["pair_id"]]})
    matrix("wrong_mode", {"runs": [{**runs[0], "mode": runs[1]["mode"]}, *runs[1:]]})

    first = runs[0]
    for label in ("missing_field", "failure_denominator"):
        folder = output / label
        folder.mkdir()
        original = Path(first["run"])
        client = read_json(original / "client.json")
        if label == "missing_field":
            del client["requests"][0]["first_token_ns"]
        else:
            client["requests"].pop()
        for name in ("engine.json", "sources.json"):
            (folder / name).symlink_to(original / name)
        write(folder / "client.json", client)
        child(
            label,
            "rejected",
            "check_m5_trace.py",
            "run",
            "--config",
            args.config,
            "--inputs",
            first["inputs"],
            "--run",
            folder,
            "--pair",
            first["pair_id"],
            "--mode",
            first["mode"],
            derived=folder / "client.json",
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
        elif label == "missing_boundary":
            path = folder / "node/gpu-during.jsonl"
            path.unlink()
        else:
            path = folder / "watcher/slurm-final.json"
            path.unlink()
        child(label, "rejected", "check_m5_resources.py", folder, derived=path)

    source_history = output / "source_mismatch" / "attempts"
    shutil.copytree(args.history, source_history, copy_function=os.link)
    raw_sources = (
        source_history / owner.name / "node" / Path(first["run"]).parent.name / "raw/sources.json"
    )
    wrong = read_json(raw_sources)
    wrong["upstream_commit"] = "0" * 40
    replace_copy(raw_sources, wrong)
    replace_copy(source_history / owner.name / "node/allocation-sources.json", wrong)
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
        "rejected",
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
        reason="runtime source or job mismatch",
        derived=raw_sources,
    )
    require(set(report) == CONTROLS, "release control matrix incomplete")
    write(
        output / "controls.json",
        {
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
