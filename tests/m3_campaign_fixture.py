"""SYNTHETIC combination fixture; native raw checker and Git are explicit test seams.

The production CLI has no fixture switch. These objects never prove GPU/native
execution. Real campaign subprocess controls use unmodified archived evidence.
"""

import copy
import json
from pathlib import Path

from m3_trace_fixture import build_fixture

from vllm_prefix_scheduler.m3_campaign_contract import HISTORY_ROWS
from vllm_prefix_scheduler.m3_contract import canonical, sha256

ROOT = Path(__file__).resolve().parents[1]


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical(data) + "\n")
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def build_campaign_fixture(directory):
    directory = Path(directory).resolve()
    fixture = build_fixture(directory / "gpu")
    root = directory / "repo"
    root.mkdir()
    paths = {"root": root, "inputs": fixture["inputs_path"], "trace_dir": fixture["trace_path"]}
    for name in (
        "m3-real-model.json",
        "m3-campaign-v1.json",
        "m3-campaign-authorization.json",
        "m3-closeout-authorization.json",
        "native-m2b.json",
        "native-bc-replay.json",
    ):
        path = root / "configs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / "configs" / name).read_bytes())
    paths.update(
        config=root / "configs/m3-real-model.json",
        campaign_config=root / "configs/m3-campaign-v1.json",
        authorization=root / "configs/m3-campaign-authorization.json",
    )
    plan = json.loads(paths["campaign_config"].read_text())
    gpu = fixture["resources"]["allocations"][0]["gpu_before"]
    historical = []
    for i, (run, state, code) in enumerate(HISTORY_ROWS):
        job = str(120001 + i)
        row = copy.deepcopy(fixture["resources"]["allocations"][0])
        row.update(run=run, job_id=job, state=state, exit_code=code)
        row["slurm_final"]["stdout"] = (
            f"JobId={job} JobState={state} ExitCode={code} TimeLimit=00:20:00 StartTime=2026-11-02T10:{i * 2:02d}:00 EndTime=2026-11-02T10:{i * 2 + 1:02d}:00"
        )
        historical.append(row)
    oldraw = write(directory / "history/raw.json", {"synthetic": True})
    ledger = {"allocations": historical, "raw_artifacts": [oldraw]}
    ledger_ref = write(directory / "history/ledger.json", ledger)
    index = {
        "schema_version": 1,
        "scope": "retained_historical_evidence_not_campaign_acceptance",
        "ledger": ledger_ref,
        "allocations": [
            {k: r[k] for k in ("run", "job_id", "state", "exit_code")} for r in historical
        ],
        "artifacts": [ledger_ref, oldraw],
    }
    paths["history_index"] = directory / "history/index.json"
    history_ref = write(paths["history_index"], index)
    plan["history"].update(index_sha256=history_ref["sha256"], ledger_sha256=ledger_ref["sha256"])
    write(paths["campaign_config"], plan)
    campaign_hash = sha256(paths["campaign_config"])
    common = {"campaign_id": plan["campaign_id"], "campaign_sha256": campaign_hash}
    source = fixture["sources"]["source_sha256"]
    source.update(
        {
            "configs/m3-campaign-v1.json": campaign_hash,
            "configs/m3-campaign-authorization.json": sha256(paths["authorization"]),
            "scripts/replay_m2b.py": "9" * 64,
            "src/vllm_prefix_scheduler/m2b_contract.py": "8" * 64,
        }
    )
    commit, job = fixture["sources"]["project_commit"], "150001"
    for event in fixture["events"]:
        event.update(**common, campaign_job_id=job)
        if event["event"] == "engine_start":
            event["source_sha256"] = copy.deepcopy(source)
        if event["event"] == "engine_ready":
            event["sources"] = copy.deepcopy(fixture["sources"])
    fixture["trace_path"].write_text("".join(canonical(e) + "\n" for e in fixture["events"]))
    domain = "monotonic_ns:controller:synthetic"
    receipt = write(
        directory / "control/attempt-01/submission.json",
        {
            "command": ["sbatch", "--parsable", "fixture.sbatch"],
            "returncode": 0,
            "stdout": job + "\n",
            "stderr": "",
            "clock_domain": domain,
            "started_monotonic_ns": 2,
            "finished_monotonic_ns": 3,
        },
    )
    ready = write(
        directory / "new/monitor.json",
        {
            **common,
            "ready": True,
            "attempt_id": "attempt-01",
            "project_commit": commit,
            "clock_domain": domain,
            "observed_monotonic_ns": 1,
            "observed_unix_ns": 1,
            "pid": 99,
        },
    )
    terminal = {
        "returncode": 0,
        "command": ["ssh", "synthetic", f"scontrol show job {job}"],
        "stdout": f"JobId={job} JobState=COMPLETED ExitCode=0:0 TimeLimit=00:20:00 AllocTRES=cpu=4,gres/gpu=1 StartTime=2026-11-03T10:00:00 EndTime=2026-11-03T10:05:00",
        "clock_domain": domain,
        "observed_monotonic_ns": 4,
        "observed_unix_ns": 4,
    }
    terminal_ref = write(directory / "new/terminal.json", terminal)
    engines = [
        {
            k: e[k]
            for k in (
                "engine_id",
                "exitcode",
                "timed_out",
                "owned_pids_before_cleanup",
                "owned_pids_remaining",
            )
        }
        for e in fixture["events"]
        if e["event"] == "process_exit"
    ]
    checkpoints = {
        "job_id": job,
        "gpu_before": gpu,
        "gpu_during": [gpu],
        "gpu_after": gpu,
        "owned_pids_remaining": [],
        "engines": engines,
    }
    checkpoints_ref = write(directory / "new/checkpoints.json", checkpoints)
    exit_ref = write(
        directory / "new/exit.json",
        {
            "gpu": 0,
            "native": 0,
            "gpu_after_query": 0,
            "job_id": job,
            "campaign_id": plan["campaign_id"],
            "project_commit": commit,
            "campaign_sha256": campaign_hash,
            "owned_pids_remaining": [],
        },
    )
    binding = {
        **common,
        "project_commit": commit,
        "job_id": job,
        "runtime_config_sha256": plan["runtime_config"]["sha256"],
        "history_index_sha256": plan["history"]["index_sha256"],
    }
    binding_ref = write(directory / "new/binding.json", binding)
    checkpoints["campaign_binding"] = binding
    checkpoints_ref = write(directory / "new/checkpoints.json", checkpoints)
    watch_path = directory / "new/watch.jsonl"
    watch_path.write_text(canonical(terminal) + "\n")
    watch_ref = {"path": str(watch_path), "sha256": sha256(watch_path)}
    monitor_exit = write(
        directory / "new/watcher-exit.json",
        {
            "pid": 99,
            "returncode": 0,
            "owned_pids_remaining": [],
            "clock_domain": domain,
            "observed_monotonic_ns": 6,
        },
    )
    submissions = write(
        directory / "new/submissions.json",
        {
            "schema_version": 1,
            **common,
            "attempts": [
                {
                    "attempt_id": "attempt-01",
                    "job_id": job,
                    "project_commit": commit,
                    "submission": receipt,
                    "monitor_ready": ready,
                    "monitor_trace": watch_ref,
                    "monitor_exit": monitor_exit,
                }
            ],
        },
    )
    active = {
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "command": ["ssh", "synthetic", 'squeue -h -o "%i|%T|%j"'],
        "clock_domain": domain,
        "observed_monotonic_ns": 5,
    }
    active_ref = write(directory / "new/active.json", active)
    roles = {
        "submission": receipt,
        "terminal": terminal_ref,
        "checkpoints": checkpoints_ref,
        "exit": exit_ref,
        "binding": binding_ref,
    }
    row = {
        "job_id": job,
        "limit_minutes": 20,
        "gpus": 1,
        "state": "COMPLETED",
        "exit_code": "0:0",
        "slurm_final": terminal,
        "evidence": roles,
        **{
            k: checkpoints[k]
            for k in ("gpu_before", "gpu_during", "gpu_after", "owned_pids_remaining")
        },
    }
    resources = {
        **common,
        "allocations": [row],
        "active_query_success": True,
        "active_owned_jobs": [],
        "active_query": active,
        "active_query_evidence": active_ref,
        "submission_ledger": submissions,
        "submission_catalog_dir": str(directory / "control"),
        "raw_artifacts": [*roles.values(), active_ref, ready, submissions],
    }
    paths["resources"] = directory / "resources.json"
    write(paths["resources"], resources)
    native = {
        "schema_version": 1,
        **common,
        "project_commit": commit,
        "job_id": job,
        "binding": binding_ref,
    }
    for kind in ("compat", "m1"):
        cfg = json.loads(
            (
                root
                / (
                    "configs/native-m2b.json"
                    if kind == "compat"
                    else "configs/native-bc-replay.json"
                )
            ).read_text()
        )
        runtime = {
            "project_commit": commit,
            "environment": {"SLURM_JOB_ID": job, "VLLM_USE_V2_MODEL_RUNNER": "1"},
            "python_optimize": 0,
            "python_version": "3.12.16 synthetic",
            "vllm_version": "0.32.0",
            "upstream_commit": fixture["sources"]["upstream_commit"],
            "modules": {n: {"sha256": h} for n, h in cfg["source_sha256"].items()},
        }
        model_hash = next(
            f["sha256"] for f in fixture["sources"]["model"]["files"] if f["name"] == "config.json"
        )
        if kind == "compat":
            runtime.update(
                suite="compat",
                core_native_replacements=[],
                model_config_sha256=model_hash,
                custom_sources={
                    f"{n}.py": source[f"src/vllm_prefix_scheduler/{n}.py"]
                    for n in ("config", "scheduler", "probe", "wait_guard", "m2b_contract")
                },
                validation_sources={"scripts/replay_m2b.py": source["scripts/replay_m2b.py"]},
            )
        else:
            runtime.update(
                python_assertions_enabled=True,
                mocks_or_stubs=[],
                substitutions=[
                    "Synthetic ModelRunnerOutput only; no model runner, model weights, tokenizer, or KV tensors",
                    "One-layer FullAttentionSpec metadata fixture (not model-derived KV geometry)",
                    "One recorded waiting-queue permutation only in C,B test fixture",
                ],
                model_config={"sha256": model_hash},
            )
        collector = directory / f"native/{kind}"
        records = []

        def add(name, value, collector=collector, records=records):
            ref = write(collector / name, value)
            record = {
                "path": name,
                "sha256": ref["sha256"],
                "bytes": (collector / name).stat().st_size,
            }
            records.append(record)
            return ref

        runtime_ref = add("artifacts/runtime.json", runtime)
        inputs_ref = add(
            "artifacts/inputs.json", {"suite": "compat", "matrix": cfg} if kind == "compat" else cfg
        )
        trace_ref = add("artifacts/trace.json", {"SYNTHETIC": "explicit raw-audit test seam only"})
        state = {"commit": commit, "dirty": False, "files": source}
        add("source-before.json", state)
        add("source-after.json", state)
        add("stdout.log", {})
        add("stderr.log", {})
        cfgref = add("configs/input.json", cfg)
        records[-1]["source"] = "synthetic-config"
        criteria = [{"id": "C1", "description": "synthetic schema2 fixture"}]
        manifest = {
            "schema_version": 2,
            "kind": "native",
            "command_status": "succeeded",
            "returncode": 0,
            "project": {"commit": commit, "dirty": False},
            "allocation": {"slurm_job_id": job},
            "source_changed_during_run": False,
            "collection_errors": [],
            "files": records,
            "criteria": criteria,
            "config_inputs_stable": True,
            "config_checks": [
                {
                    "snapshot_path": "configs/input.json",
                    "source": "synthetic-config",
                    "expected_sha256": cfgref["sha256"],
                    "actual_sha256": cfgref["sha256"],
                    "status": "unchanged",
                }
            ],
        }
        manifest_ref = write(collector / "manifest.json", manifest)
        write(
            collector / "acceptance.json",
            {
                "schema_version": 1,
                "manifest_sha256": manifest_ref["sha256"],
                "decision": "pending",
                "criteria": [{**criteria[0], "status": "pending"}],
            },
        )
        native[kind] = {
            "collector_dir": str(collector),
            "runtime": runtime_ref,
            "inputs": inputs_ref,
            "trace": trace_ref,
        }
    collector = directory / "gpu-collector"
    records = []

    def copy_record(name, source):
        path = collector / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(source).read_bytes())
        ref = {"path": str(path), "sha256": sha256(path)}
        records.append({"path": name, "sha256": ref["sha256"], "bytes": path.stat().st_size})
        return ref

    source_path = directory / "gpu/sources.json"
    write(source_path, fixture["sources"])
    gpu_sources = copy_record("artifacts/sources.json", source_path)
    gpu_trace = copy_record("artifacts/trace.jsonl", paths["trace_dir"])
    gpu_inputs = copy_record("configs/inputs.json", paths["inputs"])
    records[-1]["source"] = "synthetic-gpu-inputs"
    for name, data in (
        ("source-before.json", state),
        ("source-after.json", state),
        ("stdout.log", {}),
        ("stderr.log", {}),
    ):
        ref = write(collector / name, data)
        records.append(
            {"path": name, "sha256": ref["sha256"], "bytes": (collector / name).stat().st_size}
        )
    manifest = {
        "schema_version": 2,
        "kind": "gpu",
        "command_status": "succeeded",
        "returncode": 0,
        "project": {"commit": commit, "dirty": False},
        "allocation": {"slurm_job_id": job},
        "source_changed_during_run": False,
        "collection_errors": [],
        "files": records,
        "criteria": criteria,
        "config_inputs_stable": True,
        "config_checks": [
            {
                "snapshot_path": "configs/inputs.json",
                "source": "synthetic-gpu-inputs",
                "expected_sha256": gpu_inputs["sha256"],
                "actual_sha256": gpu_inputs["sha256"],
                "status": "unchanged",
            }
        ],
    }
    manifest_ref = write(collector / "manifest.json", manifest)
    write(
        collector / "acceptance.json",
        {
            "schema_version": 1,
            "manifest_sha256": manifest_ref["sha256"],
            "decision": "pending",
            "criteria": [{**criteria[0], "status": "pending"}],
        },
    )
    native["gpu"] = {
        "collector_dir": str(collector),
        "sources": gpu_sources,
        "trace": gpu_trace,
        "inputs": gpu_inputs,
    }
    paths["native_index"] = directory / "native-index.json"
    write(paths["native_index"], native)
    return {
        "paths": paths,
        "fixture": fixture,
        "plan": plan,
        "history": ledger,
        "resources": resources,
        "native": native,
    }
