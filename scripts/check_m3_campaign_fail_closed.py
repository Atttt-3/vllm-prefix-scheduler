"""Real subprocess controls for the campaign layer, additional to legacy M3 controls.

A successful unmodified full campaign is mandatory before mutations. Negative
controls must fail at their named gate, not merely exit nonzero. Original files
are never modified; mutated JSON/reference files live only under --output.
"""

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from check_m3_campaign import ROOT, arguments

sys.path.insert(0, str(ROOT / "src"))
from vllm_prefix_scheduler.m3_campaign_contract import require  # noqa: E402
from vllm_prefix_scheduler.m3_contract import require_assertions, sha256  # noqa: E402

CONTROLS = {
    "optimized": "requires assertions",
    "missing_campaign_envelope": "GPU event campaign identity differs",
    "mixed_gpu_job": "GPU event job identity missing/mixed",
    "history_corruption": "history index hash differs",
    "expanded_history_exception": "historical exception widened",
    "missing_new_terminal": "new allocation terminal evidence missing",
    "resource_campaign": "resources campaign identity differs",
    "zero_allocations": "new allocation budget missing/exceeded",
    "fifth_allocation": "new allocation budget missing/exceeded",
    "historical_job_reuse": "new allocation reused historical/duplicate job",
    "omitted_submission": "submission catalog omits or duplicates campaign attempt",
    "monitor_started_late": "monitor/submission/terminal clock order differs",
    "detached_terminal": "terminal declaration is not bound to original",
    "detached_checkpoint": "checkpoint job differs",
    "failed_selected_launcher": "selected campaign attempt did not complete all layers",
    "wrong_launch_binding": "attempt launch binding differs",
    "detached_active_query": "active query declaration is not bound to original",
    "stale_native_commit": "native selected job/commit differs",
    "corrupt_native_runtime": "compat runtime artifact hash differs",
    "wrong_native_inputs": "compat native frozen input differs",
    "undeclared_m1_substitutions": "m1 native substitutions differ from fixed replay v1 declaration",
    "changed_m1_substitutions": "m1 native substitutions differ from fixed replay v1 declaration",
    "monitor_missing_terminal": "terminal absent from actual monitor trace",
    "monitor_not_exited": "monitor did not exit cleanly after terminal",
    "stale_active_query": "final active query is not a later full-queue observation",
    "ten_gpu_allocation": "raw Slurm single GPU allocation missing",
    "gpu_collector_unstable": "GPU schema2 collector identity/stability differs",
    "unregistered_accepted_receipt": "submission receipt catalog differs from recorded attempts",
}


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def mutate(label, original, directory):
    args = copy.copy(original)
    flags = []
    resources, native = load(args["resources"]), load(args["native_index"])
    selected = str(native["job_id"])
    row = next(r for r in resources["allocations"] if str(r["job_id"]) == selected)
    if label == "optimized":
        return args, ["-O"]
    if label in {"missing_campaign_envelope", "mixed_gpu_job"}:
        path = Path(args["trace_dir"])
        if path.is_dir():
            path = path / "trace.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        if label == "missing_campaign_envelope":
            events[0].pop("campaign_id")
        else:
            events[0]["campaign_job_id"] = "999999999"
        changed = directory / "trace.jsonl"
        changed.write_text("".join(json.dumps(e) + "\n" for e in events))
        args["trace_dir"] = changed
        return args, flags
    if label == "expanded_history_exception":
        changed = directory / "campaign-config.json"
        value = load(args["campaign_config"])
        value["history"]["unverified_terminal_count"] = 2
        write(changed, value)
        args["campaign_config"] = changed
        return args, flags
    if label == "history_corruption":
        changed = directory / "history-index.json"
        value = load(args["history_index"])
        value["allocations"].pop()
        write(changed, value)
        args["history_index"] = changed
        return args, flags
    if label == "missing_new_terminal":
        row["evidence"].pop("terminal")
    elif label == "resource_campaign":
        resources["campaign_id"] = "wrong-campaign"
    elif label == "zero_allocations":
        resources["allocations"] = []
    elif label == "fifth_allocation":
        resources["allocations"] = [copy.deepcopy(row) for _ in range(5)]
    elif label == "historical_job_reuse":
        row["job_id"] = load(args["history_index"])["allocations"][2]["job_id"]
    elif label in {
        "omitted_submission",
        "monitor_started_late",
        "monitor_missing_terminal",
        "monitor_not_exited",
    }:
        ledger = load(resources["submission_ledger"]["path"])
        if label == "omitted_submission":
            ledger["attempts"].pop()
        else:
            attempt = next(a for a in ledger["attempts"] if str(a["job_id"]) == selected)
            if label == "monitor_missing_terminal":
                path = directory / "watch.jsonl"
                path.write_text("")
                attempt["monitor_trace"] = {"path": str(path.resolve()), "sha256": sha256(path)}
            elif label == "monitor_not_exited":
                monitor_exit = load(attempt["monitor_exit"]["path"])
                monitor_exit["owned_pids_remaining"] = [999999]
                attempt["monitor_exit"] = write(directory / "monitor-exit.json", monitor_exit)
            else:
                monitor = load(attempt["monitor_ready"]["path"])
                receipt = load(attempt["submission"]["path"])
                monitor["observed_monotonic_ns"] = receipt["finished_monotonic_ns"] + 1
                attempt["monitor_ready"] = write(directory / "monitor-ready.json", monitor)
        resources["submission_ledger"] = write(directory / "submission-ledger.json", ledger)
    elif label == "detached_terminal":
        row["slurm_final"]["stderr"] = "changed without matching raw original"
    elif label in {"detached_checkpoint", "failed_selected_launcher", "wrong_launch_binding"}:
        role = {
            "detached_checkpoint": "checkpoints",
            "failed_selected_launcher": "exit",
            "wrong_launch_binding": "binding",
        }[label]
        changed = load(row["evidence"][role]["path"])
        if label == "detached_checkpoint":
            changed["job_id"] = "999999999"
        elif label == "failed_selected_launcher":
            changed["native"] = 1
        else:
            changed["project_commit"] = "0" * 40
        row["evidence"][role] = write(directory / f"{role}.json", changed)
    elif label == "detached_active_query":
        resources["active_query"]["stderr"] = "changed without matching raw original"
    elif label == "unregistered_accepted_receipt":
        catalog = directory / "submission-catalog"
        shutil.copytree(resources["submission_catalog_dir"], catalog, copy_function=os.link)
        ledger = load(resources["submission_ledger"]["path"])
        for attempt in ledger["attempts"]:
            original_path = Path(attempt["submission"]["path"])
            copied_path = catalog / original_path.parent.name / "submission.json"
            ref = {"path": str(copied_path.resolve()), "sha256": sha256(copied_path)}
            attempt["submission"] = ref
            next(a for a in resources["allocations"] if str(a["job_id"]) == str(attempt["job_id"]))[
                "evidence"
            ]["submission"] = ref
        receipt = load(row["evidence"]["submission"]["path"])
        receipt["stdout"] = "999999999\n"
        write(catalog / "omitted-attempt/submission.json", receipt)
        resources["submission_catalog_dir"] = str(catalog.resolve())
        resources["submission_ledger"] = write(directory / "submission-ledger.json", ledger)
    elif label == "stale_active_query":
        resources["active_query"]["observed_monotonic_ns"] = 1
        resources["active_query_evidence"] = write(
            directory / "active.json", resources["active_query"]
        )
    elif label == "ten_gpu_allocation":
        terminal = load(row["evidence"]["terminal"]["path"])
        terminal["stdout"] = terminal["stdout"].replace("gres/gpu=1", "gres/gpu=10")
        row["slurm_final"] = terminal
        row["evidence"]["terminal"] = write(directory / "terminal.json", terminal)
        ledger = load(resources["submission_ledger"]["path"])
        attempt = next(a for a in ledger["attempts"] if str(a["job_id"]) == selected)
        watch = [
            json.loads(line)
            for line in Path(attempt["monitor_trace"]["path"]).read_text().splitlines()
        ]
        watch[-1] = terminal
        path = directory / "watch.jsonl"
        path.write_text("".join(json.dumps(e) + "\n" for e in watch))
        attempt["monitor_trace"] = {"path": str(path.resolve()), "sha256": sha256(path)}
        resources["submission_ledger"] = write(directory / "submissions.json", ledger)
    elif label == "gpu_collector_unstable":
        cloned = directory / "gpu-collector"
        shutil.copytree(native["gpu"]["collector_dir"], cloned, copy_function=os.link)
        manifest_path = cloned / "manifest.json"
        manifest = load(manifest_path)
        manifest["source_changed_during_run"] = True
        manifest_path.unlink()  # break the hardlink before writing; originals stay immutable
        write(manifest_path, manifest)
        native["gpu"]["collector_dir"] = str(cloned.resolve())
    elif label in {"undeclared_m1_substitutions", "changed_m1_substitutions"}:
        entry = native["m1"]
        original_runtime_hash = entry["runtime"]["sha256"]
        runtime = load(entry["runtime"]["path"])
        if label == "undeclared_m1_substitutions":
            runtime.pop("substitutions")
        else:
            runtime["substitutions"] = ["Different and unapproved test substitution"]
        cloned = directory / "m1-collector"
        shutil.copytree(entry["collector_dir"], cloned, copy_function=os.link)
        manifest_path = cloned / "manifest.json"
        manifest = load(manifest_path)
        matching = [
            record
            for record in manifest["files"]
            if record["path"].startswith("artifacts/") and record["sha256"] == original_runtime_hash
        ]
        require(len(matching) == 1, "m1 runtime mutation needs one bound collector artifact")
        record = matching[0]
        runtime_path = cloned / record["path"]
        runtime_path.unlink()  # break every mutated hardlink; originals remain read-only
        entry["runtime"] = write(runtime_path, runtime)
        record.update(sha256=entry["runtime"]["sha256"], bytes=runtime_path.stat().st_size)
        manifest_path.unlink()
        manifest_reference = write(manifest_path, manifest)
        acceptance_path = cloned / "acceptance.json"
        acceptance = load(acceptance_path)
        acceptance["manifest_sha256"] = manifest_reference["sha256"]
        acceptance_path.unlink()
        write(acceptance_path, acceptance)
        entry["collector_dir"] = str(cloned.resolve())
    elif label == "stale_native_commit":
        native["project_commit"] = "403b7838e46ecf17a97cc14de0a1f048938dfbd8"
    elif label == "corrupt_native_runtime":
        native["compat"]["runtime"]["sha256"] = "0" * 64
    elif label == "wrong_native_inputs":
        native["compat"]["inputs"] = copy.deepcopy(native["compat"]["runtime"])
    else:
        raise ValueError(f"unknown negative control: {label}")
    if label in {
        "stale_native_commit",
        "corrupt_native_runtime",
        "wrong_native_inputs",
        "gpu_collector_unstable",
        "undeclared_m1_substitutions",
        "changed_m1_substitutions",
    }:
        args["native_index"] = directory / "native-index.json"
        write(args["native_index"], native)
    else:
        args["resources"] = directory / "resources.json"
        write(args["resources"], resources)
    return args, flags


def run(command, directory):
    env = dict(os.environ)
    env.pop("PYTHONOPTIMIZE", None)
    result = subprocess.run(
        command, capture_output=True, text=True, env=env, timeout=900, check=False
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "stdout.log").write_text(result.stdout)
    (directory / "stderr.log").write_text(result.stderr)
    write(directory / "command.json", {"command": command, "returncode": result.returncode})
    return result


def command(arguments, output, flags=()):
    result = [sys.executable, *flags, str(ROOT / "scripts/check_m3_campaign.py")]
    for name, path in arguments.items():
        result += ["--" + name.replace("_", "-"), str(path)]
    return result + ["--output", str(output)]


def main():
    require_assertions()
    parser = arguments(argparse.ArgumentParser(description=__doc__))
    args = vars(parser.parse_args())
    output = args.pop("output")
    require(output is not None and not output.exists(), "fresh --output directory required")
    output.mkdir(parents=True)
    positive = run(command(args, output / "positive/audit.json"), output / "positive")
    require(positive.returncode == 0, "unmodified full campaign positive control failed")
    report = {
        "schema_version": 1,
        "scope": "campaign_extra_subprocess_controls",
        "positive_passed": True,
        "controls": [],
    }
    for label, fragment in CONTROLS.items():
        directory = output / label
        directory.mkdir()
        modified, flags = mutate(label, args, directory)
        result = run(command(modified, directory / "unexpected-audit.json", flags), directory)
        passed = result.returncode != 0 and fragment in result.stderr
        report["controls"].append(
            {
                "name": label,
                "expected_error": fragment,
                "returncode": result.returncode,
                "rejected_at_expected_gate": passed,
            }
        )
        write(output / "result.json", report)
        require(passed, f"negative control {label} did not reject at expected gate")
    report["negative_control_count"] = len(CONTROLS)
    report["passed"] = True
    write(output / "result.json", report)
    print(
        json.dumps(
            {"positive_passed": True, "negative_control_count": len(CONTROLS), "passed": True}
        )
    )


if __name__ == "__main__":
    main()
