"""Independent replacement-campaign gate; the historical M3 gate stays blocked.

No production strategy is imported. The unchanged legacy auditor evaluates only
new campaign GPU evidence; this layer binds history, authorization, every new
submission, role-specific raw resources, and same-freeze native replay evidence.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath

from . import m3_contract as legacy
from .run_evidence import check_run

SCOPE = "replacement_campaign_with_retained_history_exception"
OLD_COMMIT = "403b7838e46ecf17a97cc14de0a1f048938dfbd8"
SOURCE_ROOTS = ("src", "scripts", "configs", "tests", "Makefile")
# M1 replay v1 boundary, recorded by native_replay.runtime_record (frozen source).
# These declared native-test substitutions never apply to the real-model GPU run.
M1_NATIVE_SUBSTITUTIONS_V1 = (
    "Synthetic ModelRunnerOutput only; no model runner, model weights, tokenizer, or KV tensors",
    "One-layer FullAttentionSpec metadata fixture (not model-derived KV geometry)",
    "One recorded waiting-queue permutation only in C,B test fixture",
)
HISTORY_ROWS = (
    ("calibration-01", "FAILED", "1:0"),
    ("formal-01", "COMPLETED", "0:0"),
    ("formal-02", "UNVERIFIED_PURGED", None),
    ("formal-03", "COMPLETED", "0:0"),
)


def require(condition, message):
    if not condition:
        raise ValueError(f"M3 campaign: {message}")


def load(path):
    value = json.loads(Path(path).read_text())
    require(type(value) is dict, "JSON object required")
    return value


def reference(record, label):
    require(type(record) is dict and type(record.get("path")) is str, f"{label} reference missing")
    path = Path(record["path"])
    require(
        path.is_absolute() and path.is_file() and legacy.is_hash(record.get("sha256")),
        f"{label} artifact missing",
    )
    require(legacy.sha256(path) == record["sha256"], f"{label} artifact hash differs")
    if "bytes" in record:
        require(
            type(record["bytes"]) is int and path.stat().st_size == record["bytes"],
            f"{label} size differs",
        )
    return path


def identity(value, campaign, digest, label):
    require(
        value.get("campaign_id") == campaign and value.get("campaign_sha256") == digest,
        f"{label} campaign identity differs",
    )


def authorization_gate(root, plan, authorization, config, authorization_path):
    require(
        plan.get("schema_version") == 1
        and plan.get("task_id") == legacy.TASK_ID
        and plan.get("scope") == SCOPE,
        "campaign config scope/schema differs",
    )
    require(
        type(plan.get("campaign_id")) is str
        and re.fullmatch(r"m3-campaign-[a-z0-9-]+", plan["campaign_id"]),
        "campaign id invalid",
    )
    require(
        plan.get("budget")
        == {
            "new_allocations": 4,
            "cumulative_allocations": 8,
            "gpus": 1,
            "max_minutes": 20,
            "serial": True,
        },
        "campaign budget differs",
    )
    require(
        plan.get("required")
        == {"gpu_cases": 20, "repeats": 3, "compat_cases": 432, "m1_cases": 12},
        "campaign matrix differs",
    )
    for key, path in (("runtime_config", config), ("authorization", authorization_path)):
        ref = plan.get(key, {})
        require(
            Path(path).resolve() == (root / ref.get("path", "")).resolve()
            and legacy.sha256(path) == ref.get("sha256"),
            f"{key} binding differs",
        )
    expected = {
        "schema_version": 1,
        "task_id": legacy.TASK_ID,
        "prior_gpu_native_tested_commit": OLD_COMMIT,
        "record_type": "replacement_campaign_authorization",
        "acceptance_scope": SCOPE,
        "original_allocation_limit": 4,
        "additional_allocation_limit": 4,
        "cumulative_allocation_limit": 8,
        "max_minutes_per_allocation": 20,
        "gpus_per_allocation": 1,
        "serial_execution_required": True,
        "failed_allocations_and_retries_count": True,
        "historical_allocations_must_be_retained": True,
        "historical_terminal_status": "unverifiable",
        "original_full_history_gate": "blocked",
        "new_campaign_terminal_evidence_required": True,
        "old_unknown_cannot_cover_new_allocation": True,
        "administrator_recovery_is_prerequisite": False,
        "original_runtime_config": plan["runtime_config"]["path"],
        "original_runtime_config_sha256": plan["runtime_config"]["sha256"],
    }
    require(
        all(
            type(authorization.get(k)) is type(v) and authorization.get(k) == v
            for k, v in expected.items()
        ),
        "replacement campaign authorization differs",
    )
    prior = authorization.get("previous_authorization", {})
    require(
        legacy.sha256(root / prior.get("path", "")) == prior.get("sha256"),
        "previous authorization lineage differs",
    )


def history_gate(plan, path):
    expected = plan.get("history", {})
    require(legacy.sha256(path) == expected.get("index_sha256"), "history index hash differs")
    history = load(path)
    require(
        history.get("schema_version") == 1
        and history.get("scope") == "retained_historical_evidence_not_campaign_acceptance",
        "history scope differs",
    )
    require(
        expected.get("allocation_count") == 4
        and expected.get("unverified_run") == "formal-02"
        and expected.get("unverified_terminal_count") == 1
        and expected.get("original_full_history_gate") == "blocked",
        "historical exception widened",
    )
    rows = history.get("allocations", [])
    catalog = tuple(
        (r.get("run"), str(r.get("job_id")), r.get("state"), r.get("exit_code")) for r in rows
    )
    require(
        tuple((r[0], r[2], r[3]) for r in catalog) == HISTORY_ROWS
        and all(r[1].isdigit() for r in catalog)
        and len({r[1] for r in catalog}) == 4,
        "historical allocation catalog differs",
    )
    ledger_path = reference(history.get("ledger"), "historical ledger")
    require(
        legacy.sha256(ledger_path) == expected.get("ledger_sha256"),
        "historical ledger binding differs",
    )
    ledger = load(ledger_path)
    require(
        tuple(
            (r.get("run"), str(r.get("job_id")), r.get("state"), r.get("exit_code"))
            for r in ledger.get("allocations", [])
        )
        == catalog,
        "historical ledger catalog differs",
    )
    artifacts = history.get("artifacts")
    require(type(artifacts) is list and artifacts, "historical artifact catalog missing")
    seen = set()
    for item in artifacts:
        source = reference(item, "historical raw")
        require(str(source.resolve()) not in seen, "historical artifact duplicated")
        seen.add(str(source.resolve()))
    require(str(ledger_path.resolve()) in seen, "historical ledger omitted from raw catalog")
    for item in ledger.get("raw_artifacts", []):
        require(
            str(reference(item, "historical ledger raw").resolve()) in seen,
            "historical raw omitted from index",
        )
    # These checks retain resource evidence without declaring the missing terminal valid.
    for row in ledger["allocations"]:
        require(
            row.get("gpus") == 1
            and type(row.get("limit_minutes")) is int
            and 0 < row["limit_minutes"] <= 20,
            "historical budget differs",
        )
        require(row.get("owned_pids_remaining") == [], "historical task processes remain")
        for key in ("gpu_before", "gpu_during", "gpu_after"):
            samples = row.get(key)
            samples = [samples] if type(samples) is dict else samples
            require(type(samples) is list and samples, "historical GPU observations missing")
            for sample in samples:
                legacy.audit_gpu_snapshot(sample)
    return history, ledger


def slurm_fields(raw):
    return dict(re.findall(r"(\w+)=([^\s]+)", raw.get("stdout", "")))


def resources_gate(
    resources, plan, campaign_hash, history, selected_job, commit, events, root=None
):
    identity(resources, plan["campaign_id"], campaign_hash, "resources")
    allocations = resources.get("allocations")
    require(
        type(allocations) is list and 1 <= len(allocations) <= 4,
        "new allocation budget missing/exceeded",
    )
    jobs = [str(a.get("job_id")) for a in allocations]
    require(
        len(set(jobs)) == len(jobs)
        and not set(jobs) & {str(r["job_id"]) for r in history["allocations"]},
        "new allocation reused historical/duplicate job",
    )
    require(selected_job in jobs, "selected GPU/native job absent from campaign resources")
    ledger = load(reference(resources.get("submission_ledger"), "submission ledger"))
    identity(ledger, plan["campaign_id"], campaign_hash, "submission ledger")
    require(ledger.get("schema_version") == 1, "submission ledger schema differs")
    attempts = ledger.get("attempts")
    require(
        type(attempts) is list
        and len(attempts) == len(jobs)
        and {str(a.get("job_id")) for a in attempts} == set(jobs),
        "submission catalog omits or duplicates campaign attempt",
    )
    require(
        [a.get("attempt_id") for a in attempts]
        == [f"attempt-{i:02d}" for i in range(1, len(attempts) + 1)],
        "attempt catalog order differs",
    )
    catalog_path = Path(resources.get("submission_catalog_dir", ""))
    require(
        catalog_path.is_absolute() and catalog_path.is_dir() and not catalog_path.is_symlink(),
        "submission receipt catalog directory missing/unsafe",
    )
    accepted_receipts = set()
    receipt_files = sorted(catalog_path.glob("*/submission.json"))
    require(receipt_files, "submission receipt catalog empty")
    for path in receipt_files:
        require(
            not path.is_symlink() and not path.parent.is_symlink() and path.is_file(),
            "submission receipt catalog contains symlink",
        )
        raw = load(path)
        require(type(raw.get("returncode")) is int, "submission receipt status ambiguous")
        if raw["returncode"] == 0:
            require(
                type(raw.get("stdout")) is str and re.fullmatch(r"[0-9]+\n?", raw["stdout"]),
                "successful submission receipt identity ambiguous",
            )
            accepted_receipts.add((str(path.resolve()), legacy.sha256(path), raw["stdout"].strip()))
    declared_receipts = {
        (str(Path(a["submission"]["path"]).resolve()), a["submission"]["sha256"], str(a["job_id"]))
        for a in attempts
    }
    require(
        accepted_receipts == declared_receipts,
        "submission receipt catalog differs from recorded attempts",
    )
    by_job = {str(a["job_id"]): a for a in attempts}
    raw_roles = set()
    terminal_clocks = []
    monitor_exit_clocks = []
    for row in allocations:
        job = str(row["job_id"])
        attempt = by_job[job]
        attempt_commit = attempt.get("project_commit")
        require(
            legacy.is_hash(attempt_commit, 40) and attempt_commit != OLD_COMMIT,
            "new attempt frozen commit differs",
        )
        if attempt_commit != commit:
            require(root is not None, "attempt source repository missing")
            exists = subprocess.run(
                ["git", "-C", str(root), "cat-file", "-e", f"{attempt_commit}^{{commit}}"],
                capture_output=True,
                check=False,
            )
            require(exists.returncode == 0, "attempt source commit unavailable")
        evidence = row.get("evidence", {})
        require(type(evidence.get("terminal")) is dict, "new allocation terminal evidence missing")
        require(
            set(evidence) == {"submission", "terminal", "checkpoints", "exit", "binding"},
            "allocation raw evidence roles differ",
        )
        paths = {role: reference(ref, f"{role} raw") for role, ref in evidence.items()}
        raw_roles.update((str(p.resolve()), legacy.sha256(p)) for p in paths.values())
        require(
            evidence["submission"] == attempt.get("submission"),
            "submission receipt role binding differs",
        )
        receipt, terminal, checkpoints, exits = (
            load(paths[k]) for k in ("submission", "terminal", "checkpoints", "exit")
        )
        require(
            receipt.get("returncode") == 0
            and type(receipt.get("returncode")) is int
            and receipt.get("stdout", "").strip() == job
            and re.fullmatch(r"[0-9]+\n?", receipt["stdout"]),
            "accepted sbatch receipt differs",
        )
        command = receipt.get("command")
        require(
            type(command) is list
            and command
            and all(type(x) is str for x in command)
            and any("sbatch" in x for x in command),
            "sbatch command receipt missing",
        )
        monitor = load(reference(attempt.get("monitor_ready"), "monitor ready"))
        identity(monitor, plan["campaign_id"], campaign_hash, "monitor ready")
        require(
            monitor.get("ready") is True
            and monitor.get("attempt_id") == attempt["attempt_id"]
            and monitor.get("project_commit") == attempt_commit,
            "monitor readiness binding differs",
        )
        domain = monitor.get("clock_domain")
        clocks = [
            monitor.get("observed_monotonic_ns"),
            receipt.get("started_monotonic_ns"),
            receipt.get("finished_monotonic_ns"),
            terminal.get("observed_monotonic_ns"),
        ]
        require(
            type(domain) is str
            and domain.startswith("monotonic_ns:controller:")
            and receipt.get("clock_domain") == terminal.get("clock_domain") == domain
            and all(type(x) is int and x > 0 for x in clocks)
            and clocks[0] < clocks[1] <= clocks[2] < clocks[3],
            "monitor/submission/terminal clock order differs",
        )
        terminal_clocks.append((domain, clocks[-1]))
        require(
            terminal.get("command", [])[-1:] == [f"scontrol show job {job}"],
            "terminal query command differs",
        )
        watch_path = reference(attempt.get("monitor_trace"), "monitor trace")
        watch = [json.loads(line) for line in watch_path.read_text().splitlines() if line.strip()]
        require(
            watch
            and watch[-1] == terminal
            and all(
                e.get("clock_domain") == domain
                and e.get("command", [])[-1:] == [f"scontrol show job {job}"]
                and type(e.get("observed_monotonic_ns")) is int
                for e in watch
            ),
            "terminal absent from actual monitor trace",
        )
        watch_clocks = [e["observed_monotonic_ns"] for e in watch]
        require(
            watch_clocks == sorted(set(watch_clocks)) and watch_clocks[0] > clocks[2],
            "monitor trace query chronology differs",
        )
        monitor_exit = load(reference(attempt.get("monitor_exit"), "monitor exit"))
        require(
            type(monitor.get("pid")) is int
            and monitor_exit.get("pid") == monitor["pid"]
            and type(monitor_exit.get("returncode")) is int
            and monitor_exit["returncode"] == 0
            and monitor_exit.get("owned_pids_remaining") == []
            and monitor_exit.get("clock_domain") == domain
            and type(monitor_exit.get("observed_monotonic_ns")) is int
            and monitor_exit["observed_monotonic_ns"] > clocks[-1],
            "monitor did not exit cleanly after terminal",
        )
        monitor_exit_clocks.append(monitor_exit["observed_monotonic_ns"])
        require(terminal == row.get("slurm_final"), "terminal declaration is not bound to original")
        require(str(checkpoints.get("job_id")) == job, "checkpoint job differs")
        for key in ("gpu_before", "gpu_during", "gpu_after", "owned_pids_remaining"):
            require(checkpoints.get(key) == row.get(key), f"checkpoint {key} differs from original")
        require(
            str(exits.get("job_id")) == job
            and exits.get("campaign_id") == plan["campaign_id"]
            and exits.get("project_commit") == attempt_commit,
            "launcher exit identity differs",
        )
        require(
            all(
                type(exits.get(k)) is int and exits[k] >= 0
                for k in ("gpu", "native", "gpu_after_query")
            ),
            "launcher exit codes missing",
        )
        binding = load(paths["binding"])
        identity(binding, plan["campaign_id"], campaign_hash, "attempt launch binding")
        require(
            binding.get("project_commit") == attempt_commit
            and str(binding.get("job_id")) == job
            and binding.get("runtime_config_sha256") == plan["runtime_config"]["sha256"]
            and binding.get("history_index_sha256") == plan["history"]["index_sha256"],
            "attempt launch binding differs",
        )
        require(
            checkpoints.get("campaign_binding") == binding,
            "checkpoint launch binding differs from original",
        )
        identity(exits, plan["campaign_id"], campaign_hash, "launcher exit")
        require(exits.get("owned_pids_remaining") == [], "launcher task processes remain")
        fields = slurm_fields(terminal)
        require(
            "gres/gpu=1" in fields.get("AllocTRES", "").split(",")
            and all(
                token.rsplit("=", 1)[-1] == "1"
                for token in fields.get("AllocTRES", "").split(",")
                if token.startswith("gres/gpu")
            ),
            "raw Slurm single GPU allocation missing",
        )
        require(
            fields.get("TimeLimit") == f"00:{row.get('limit_minutes', 0):02d}:00",
            "Slurm declared time limit differs",
        )
        for key in ("gpu_before", "gpu_during", "gpu_after"):
            samples = row[key] if type(row[key]) is list else [row[key]]
            require(
                all(len(s.get("stdout", "").strip().splitlines()) == 1 for s in samples),
                "new allocation observed multiple GPUs",
            )
        if job == selected_job:
            require(attempt_commit == commit, "selected attempt commit differs from GPU")
            require(
                row.get("state") == "COMPLETED"
                and row.get("exit_code") == "0:0"
                and all(exits[k] == 0 for k in ("gpu", "native", "gpu_after_query")),
                "selected campaign attempt did not complete all layers",
            )
            observed = checkpoints.get("engines", [])
            exited = [e for e in events if e["event"] == "process_exit"]
            require(
                len(observed) == len(exited) == 8
                and {e.get("engine_id") for e in observed} == {e["engine_id"] for e in exited},
                "selected process checkpoint engine catalog differs",
            )
            for record in observed:
                event = next(e for e in exited if e["engine_id"] == record["engine_id"])
                require(
                    all(event.get(k) == v for k, v in record.items()),
                    "process exit differs from resource original",
                )
    active_path = reference(resources.get("active_query_evidence"), "active query")
    require(
        load(active_path) == resources.get("active_query"),
        "active query declaration is not bound to original",
    )
    active = resources["active_query"]
    require(
        active.get("command", [])[-1:] == ['squeue -h -o "%i|%T|%j"']
        and type(active.get("observed_monotonic_ns")) is int
        and all(
            active.get("clock_domain") == d and active["observed_monotonic_ns"] > t
            for d, t in terminal_clocks
        ),
        "final active query is not a later full-queue observation",
    )
    require(
        active["observed_monotonic_ns"] < max(monitor_exit_clocks),
        "final watcher exited before its active query",
    )
    raw_roles.add((str(active_path.resolve()), legacy.sha256(active_path)))
    registered = {
        (str(Path(a["path"]).resolve()), a["sha256"]) for a in resources.get("raw_artifacts", [])
    }
    require(raw_roles <= registered, "resource role originals absent from raw artifact catalog")
    require(
        not (
            {str(r["job_id"]) for r in history["allocations"]}
            & set(re.findall(r"\b[0-9]+\b", resources["active_query"].get("stdout", "")))
        ),
        "historical job remains in final active query",
    )
    result = legacy.audit_resources(resources)
    # Server-domain times only: new jobs are later than every retained known terminal.
    old_ends = [
        datetime.fromisoformat(slurm_fields(r["slurm_final"])["EndTime"])
        for r in history["allocations"]
        if r["state"] != "UNVERIFIED_PURGED"
    ]
    new_starts = [
        datetime.fromisoformat(slurm_fields(r["slurm_final"])["StartTime"]) for r in allocations
    ]
    require(
        max(old_ends) <= min(new_starts), "campaign overlaps retained historical terminal interval"
    )
    return result


def frozen_sources(root, sources):
    commit = sources.get("project_commit")
    require(
        legacy.is_hash(commit, 40)
        and commit != OLD_COMMIT
        and sources.get("project_dirty") is False,
        "campaign requires a new clean frozen commit",
    )
    tree = subprocess.check_output(
        ["git", "-C", str(root), "ls-tree", "-r", "--name-only", commit, "--", *SOURCE_ROOTS],
        text=True,
    ).splitlines()
    hashes = sources.get("source_sha256")
    require(
        type(hashes) is dict and set(hashes) == set(tree) and hashes,
        "campaign source catalog differs from frozen Git tree",
    )
    for name in tree:
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts, "source path outside checkout")
        blob = subprocess.check_output(
            ["git", "-C", str(root), "cat-file", "blob", f"{commit}:{name}"]
        )
        require(
            hashlib.sha256(blob).hexdigest() == hashes[name],
            "campaign source differs from frozen Git blob",
        )
        require(
            (root / name).is_file() and legacy.sha256(root / name) == hashes[name],
            "current audit checkout differs from frozen source",
        )
    unchanged = [
        f"src/vllm_prefix_scheduler/{name}.py"
        for name in ("scheduler", "config", "probe", "wait_guard", "m3_contract")
    ] + [
        "scripts/check_m3_trace.py",
        "scripts/check_m3_fail_closed.py",
        "scripts/export_m3.py",
        "tests/native/m2b_trace_check.py",
        "tests/native/m2a_trace_check.py",
        "tests/native/test_native_replay.py",
        "configs/native-m2b.json",
        "configs/native-bc-replay.json",
    ]
    for path in unchanged:
        original = subprocess.check_output(
            ["git", "-C", str(root), "cat-file", "blob", f"{OLD_COMMIT}:{path}"]
        )
        require(
            hashlib.sha256(original).hexdigest() == hashes.get(path),
            "legacy strategy/auditor changed; replacement campaign contract insufficient",
        )
    return hashes


def native_raw_audit(root, kind, trace):
    script, cfg = (
        ("tests/native/m2b_trace_check.py", "configs/native-m2b.json")
        if kind == "compat"
        else ("tests/native/test_native_replay.py", "configs/native-bc-replay.json")
    )
    command = [
        sys.executable,
        str(root / script),
        "--config",
        str(root / cfg),
        "--trace",
        str(trace),
    ]
    if kind == "compat":
        command += ["--suite", "compat"]
    env = dict(os.environ)
    env.pop("PYTHONOPTIMIZE", None)
    process = subprocess.run(
        command, capture_output=True, text=True, env=env, timeout=600, check=False
    )
    require(
        process.returncode == 0,
        f"{kind} independent raw native audit failed: {process.stderr[-1000:]}",
    )
    result = json.loads(process.stdout)
    require(
        (
            result.get("matrix_verified") is True
            and result.get("suite") == "compat"
            and result.get("case_count") == 432
        )
        if kind == "compat"
        else (
            result.get("audited_native_cases") == 12 and result.get("causal_trace_verified") is True
        ),
        f"{kind} raw native matrix incomplete",
    )
    return {
        "cases": 432 if kind == "compat" else 12,
        "raw_audit_sha256": hashlib.sha256(process.stdout.encode()).hexdigest(),
    }


def native_gate(root, index, plan, campaign_hash, commit, selected_job, sources):
    identity(index, plan["campaign_id"], campaign_hash, "native index")
    require(
        index.get("schema_version") == 1
        and index.get("project_commit") == commit
        and str(index.get("job_id")) == selected_job,
        "native selected job/commit differs",
    )
    binding = load(reference(index.get("binding"), "campaign launch binding"))
    identity(binding, plan["campaign_id"], campaign_hash, "campaign launch binding")
    require(
        binding.get("project_commit") == commit
        and str(binding.get("job_id")) == selected_job
        and binding.get("runtime_config_sha256") == plan["runtime_config"]["sha256"]
        and binding.get("history_index_sha256") == plan["history"]["index_sha256"],
        "campaign launch binding differs",
    )
    model_config_hash = next(
        item["sha256"] for item in sources["model"]["files"] if item["name"] == "config.json"
    )
    result = {"compat_cases": 432, "m1_cases": 12}
    for kind in ("compat", "m1"):
        entry = index.get(kind, {})
        directory = Path(entry.get("collector_dir", ""))
        paths = {
            key: reference(entry.get(key), f"{kind} {key}")
            for key in ("runtime", "inputs", "trace")
        }
        manifest = load(directory / "manifest.json")
        require(
            manifest.get("schema_version") == 2
            and manifest.get("kind") == "native"
            and manifest.get("source_changed_during_run") is False,
            f"{kind} native schema2 stable collection missing",
        )
        require(
            manifest.get("project", {}).get("commit") == commit
            and manifest["project"].get("dirty") is False
            and str(manifest.get("allocation", {}).get("slurm_job_id")) == selected_job,
            f"{kind} collector commit/job differs",
        )
        check_run(directory, require_success=True)
        before, after = (load(directory / f"source-{when}.json") for when in ("before", "after"))
        require(
            before == after
            and before.get("commit") == commit
            and before.get("dirty") is False
            and all(
                before.get("files", {}).get(n) == h for n, h in sources["source_sha256"].items()
            ),
            f"{kind} source snapshots differ from GPU freeze",
        )
        records = manifest.get("files", [])
        for key, path in paths.items():
            require(
                any(
                    r.get("path", "").startswith("artifacts/")
                    and r.get("sha256") == legacy.sha256(path)
                    and r.get("bytes") == path.stat().st_size
                    for r in records
                ),
                f"{kind} {key} not bound to schema2 artifact",
            )
        runtime, inputs = load(paths["runtime"]), load(paths["inputs"])
        require(
            (
                runtime.get("model_config_sha256")
                if kind == "compat"
                else runtime.get("model_config", {}).get("sha256")
            )
            == model_config_hash,
            f"{kind} model metadata differs from real GPU model",
        )
        require(
            runtime.get("project_commit") == commit
            and str(runtime.get("environment", {}).get("SLURM_JOB_ID")) == selected_job,
            f"{kind} runtime commit/job differs",
        )
        require(
            runtime.get("python_optimize") == 0
            and type(runtime.get("python_optimize")) is int
            and runtime.get("python_version", "").startswith("3.12.16 ")
            and runtime.get("vllm_version") == "0.32.0"
            and runtime.get("upstream_commit") == sources["upstream_commit"],
            f"{kind} native runtime differs",
        )
        require(
            runtime.get("environment", {}).get("VLLM_USE_V2_MODEL_RUNNER") == "1",
            f"{kind} V2 runtime missing",
        )
        config_path = root / (
            "configs/native-m2b.json" if kind == "compat" else "configs/native-bc-replay.json"
        )
        config = load(config_path)
        require(
            inputs == ({"suite": "compat", "matrix": config} if kind == "compat" else config),
            f"{kind} native frozen input differs",
        )
        modules = runtime.get("modules", {})
        require(
            set(modules) == set(config["source_sha256"])
            and all(modules[n].get("sha256") == h for n, h in config["source_sha256"].items()),
            f"{kind} native module identity differs",
        )
        if kind == "compat":
            require(
                runtime.get("suite") == "compat" and runtime.get("core_native_replacements") == [],
                "compat native replacement/suite differs",
            )
            require(
                runtime.get("custom_sources")
                == {
                    f"{n}.py": sources["source_sha256"][f"src/vllm_prefix_scheduler/{n}.py"]
                    for n in ("config", "scheduler", "probe", "wait_guard", "m2b_contract")
                },
                "compat production source differs",
            )
            validation = runtime.get("validation_sources", {})
            require(
                validation
                and all(sources["source_sha256"].get(n) == h for n, h in validation.items()),
                "compat validator source differs",
            )
        else:
            require(
                runtime.get("python_assertions_enabled") is True
                and runtime.get("mocks_or_stubs") == [],
                "m1 native assertion/mock contract differs",
            )
            require(
                runtime.get("substitutions") == list(M1_NATIVE_SUBSTITUTIONS_V1),
                "m1 native substitutions differ from fixed replay v1 declaration",
            )
        result[kind] = {
            **native_raw_audit(root, kind, paths["trace"]),
            "runtime_sha256": legacy.sha256(paths["runtime"]),
            "inputs_sha256": legacy.sha256(paths["inputs"]),
            "trace_sha256": legacy.sha256(paths["trace"]),
            "manifest_sha256": legacy.sha256(directory / "manifest.json"),
            "source_sha256": legacy.sha256(directory / "source-before.json"),
        }
    return result


def gpu_collector_gate(index, plan, campaign_hash, sources, selected_job, inputs_path, trace_path):
    entry = index.get("gpu", {})
    directory = Path(entry.get("collector_dir", ""))
    paths = {
        key: reference(entry.get(key), f"GPU collector {key}")
        for key in ("sources", "inputs", "trace")
    }
    actual_trace = Path(trace_path)
    if actual_trace.is_dir():
        actual_trace = actual_trace / "trace.jsonl"
    require(
        legacy.sha256(paths["trace"]) == legacy.sha256(actual_trace)
        and legacy.sha256(paths["inputs"]) == legacy.sha256(inputs_path)
        and load(paths["sources"]) == sources,
        "GPU collector inputs/trace/sources differ from actual audit",
    )
    manifest = load(directory / "manifest.json")
    commit = sources["project_commit"]
    require(
        manifest.get("schema_version") == 2
        and manifest.get("kind") == "gpu"
        and manifest.get("source_changed_during_run") is False
        and manifest.get("project", {}).get("commit") == commit
        and manifest["project"].get("dirty") is False
        and str(manifest.get("allocation", {}).get("slurm_job_id")) == selected_job,
        "GPU schema2 collector identity/stability differs",
    )
    check_run(directory, require_success=True)
    before, after = (load(directory / f"source-{when}.json") for when in ("before", "after"))
    require(
        before == after
        and before.get("commit") == commit
        and before.get("dirty") is False
        and all(before.get("files", {}).get(n) == h for n, h in sources["source_sha256"].items()),
        "GPU source snapshots differ from actual freeze",
    )
    for key, path in paths.items():
        prefix = "configs/" if key == "inputs" else "artifacts/"
        require(
            any(
                r.get("path", "").startswith(prefix)
                and r.get("sha256") == legacy.sha256(path)
                and r.get("bytes") == path.stat().st_size
                for r in manifest.get("files", [])
            ),
            f"GPU {key} not bound to schema2 collection",
        )
    return {
        "manifest_sha256": legacy.sha256(directory / "manifest.json"),
        "source_sha256": legacy.sha256(directory / "source-before.json"),
    }


def artifact_inventory(paths):
    """Snapshot direct and role-bound indirect originals, including collector files."""
    records, visited, expanded = {}, set(), set()

    def visit(path, recurse=False):
        path = Path(path).resolve()
        if path in visited and (not recurse or path in expanded):
            return
        visited.add(path)
        require(path.is_file(), "bound evidence inventory file missing")
        records[str(path)] = {
            "path": str(path),
            "sha256": legacy.sha256(path),
            "bytes": path.stat().st_size,
        }
        if not recurse or path.suffix != ".json":
            return
        expanded.add(path)
        value = json.loads(path.read_text())

        def walk(item):
            if type(item) is dict:
                if type(item.get("path")) is str and legacy.is_hash(item.get("sha256")):
                    candidate = Path(item["path"])
                    if candidate.is_absolute():
                        visit(candidate)
                if type(item.get("submission_catalog_dir")) is str:
                    for receipt_path in sorted(
                        Path(item["submission_catalog_dir"]).glob("*/submission.json")
                    ):
                        visit(receipt_path)
                if type(item.get("collector_dir")) is str:
                    directory = Path(item["collector_dir"])
                    visit(directory / "manifest.json")
                    visit(directory / "acceptance.json")
                    manifest = load(directory / "manifest.json")
                    for record in manifest.get("files", []):
                        visit(directory / record["path"])
                for key, child in item.items():
                    if (
                        key in {"ledger", "submission_ledger"}
                        and type(child) is dict
                        and "path" in child
                    ):
                        ledger_path = Path(child["path"]).resolve()
                        visit(ledger_path, recurse=True)
                    else:
                        walk(child)
            elif type(item) is list:
                for child in item:
                    walk(child)

        walk(value)

    for path in paths:
        visit(path, recurse=True)
    return [records[key] for key in sorted(records)]


def audit_campaign(
    *,
    root,
    campaign_config,
    authorization,
    history_index,
    config,
    inputs,
    trace_dir,
    resources,
    native_index,
):
    legacy.require_assertions()
    root = Path(root).resolve()
    plan, auth = load(campaign_config), load(authorization)
    authorization_gate(root, plan, auth, config, authorization)
    history, historical_ledger = history_gate(plan, history_index)
    canonical_trace = (
        Path(trace_dir) if Path(trace_dir).is_file() else Path(trace_dir) / "trace.jsonl"
    )
    inventory_paths = [
        history_index,
        resources,
        native_index,
        campaign_config,
        authorization,
        config,
        inputs,
        canonical_trace,
    ]
    inventory = artifact_inventory(inventory_paths)
    campaign_hash = legacy.sha256(campaign_config)
    events = legacy.read_events(trace_dir)
    require(events, "campaign trace empty")
    for event in events:
        identity(event, plan["campaign_id"], campaign_hash, "GPU event")
    jobs = {str(e.get("campaign_job_id")) for e in events}
    require(len(jobs) == 1 and next(iter(jobs)).isdigit(), "GPU event job identity missing/mixed")
    selected_job = next(iter(jobs))
    ready = [e for e in events if e.get("event") == "engine_ready"]
    require(len(ready) == 8, "campaign GPU engine catalog incomplete")
    sources = ready[0]["sources"]
    commit = sources.get("project_commit")
    hashes = frozen_sources(root, sources)
    for path in (campaign_config, authorization, config):
        name = Path(path).resolve().relative_to(root).as_posix()
        require(
            hashes.get(name) == legacy.sha256(path),
            "campaign/config/authorization not bound to GPU sources",
        )
    resource_data = load(resources)
    resources_gate(
        resource_data,
        plan,
        campaign_hash,
        historical_ledger,
        selected_job,
        commit,
        events,
        root=root,
    )
    native_data = load(native_index)
    gpu_collection = gpu_collector_gate(
        native_data, plan, campaign_hash, sources, selected_job, inputs, trace_dir
    )
    gpu = legacy.audit_trace(
        events,
        load(config),
        resources=resource_data,
        config_sha256=legacy.sha256(config),
        frozen_inputs=load(inputs),
        inputs_sha256=legacy.sha256(inputs),
    )
    require(
        gpu.get("accepted") is True
        and gpu.get("scope") == "complete"
        and gpu.get("case_count") == 20,
        "new campaign GPU matrix incomplete",
    )
    native = native_gate(root, native_data, plan, campaign_hash, commit, selected_job, sources)
    paths = {
        "campaign_config": campaign_config,
        "authorization": authorization,
        "history_index": history_index,
        "config": config,
        "inputs": inputs,
        "resources": resources,
        "native_index": native_index,
    }
    trace_path = Path(trace_dir)
    trace_files = (
        [trace_path]
        if trace_path.is_file()
        else (
            [trace_path / "trace.jsonl"]
            if (trace_path / "trace.jsonl").is_file()
            else sorted(trace_path.rglob("*.jsonl"))
        )
    )
    require(
        inventory == artifact_inventory(inventory_paths),
        "bound evidence changed during campaign audit",
    )
    count = len(resource_data["allocations"])
    return {
        "schema_version": 1,
        "scope": SCOPE,
        "campaign_id": plan["campaign_id"],
        "campaign_acceptance": "passed",
        "campaign_resource_acceptance": "passed",
        "historical_resource_status": "exception_unverified_terminal",
        "original_full_history_gate": "blocked",
        "history_terminal_unverifiable": True,
        "original_full_history_gate_blocked": True,
        "task_budget_status": "within_8",
        "project_commit": commit,
        "gpu_audit": gpu,
        "history": {
            "allocation_count": len(history["allocations"]),
            "unverified_terminal_count": 1,
            "index_sha256": legacy.sha256(history_index),
            "ledger_sha256": plan["history"]["ledger_sha256"],
        },
        "budget": {
            "historical_allocations": 4,
            "new_allocations": count,
            "total_allocations": 4 + count,
            "cumulative_limit": 8,
        },
        "native": native,
        "provenance": {
            **{f"{key}_sha256": legacy.sha256(path) for key, path in paths.items()},
            "trace_sha256": [legacy.sha256(path) for path in trace_files],
            "artifact_inventory": inventory,
            "gpu_collector": gpu_collection,
        },
    }
