"""Local contract controls. These tests do not run vLLM or a GPU."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from m4_trace_fixture import Trace, gpu_fixture

from vllm_prefix_scheduler.m4_contract import (
    EXECUTION_GROUPS,
    audit,
    audit_lifecycle,
    audit_probe,
    expected_cases,
    matrix_gate,
    sha256,
    source_gate,
)

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "configs/m4-stability.json").read_text())


def replace_reference(reference, value):
    path = Path(reference["path"])
    path.write_text(json.dumps(value) + "\n")
    reference["sha256"] = sha256(path)


def declare_execution_group(resources, group, index=0):
    evidence = resources["allocations"][index]["evidence"]
    declaration = {
        "execution_group": group,
        "declared_steps": EXECUTION_GROUPS[group],
        "phase": "formal",
    }
    binding = json.loads(Path(evidence["binding"]["path"]).read_text())
    binding.update(declaration)
    replace_reference(evidence["binding"], binding)
    exits = json.loads(Path(evidence["exit"]["path"]).read_text())
    exits.update(declaration)
    exits["steps"] = [
        {"name": name, "returncode": 0, "status": "exited", "remaining": []}
        for name in EXECUTION_GROUPS[group]
    ]
    replace_reference(evidence["exit"], exits)
    checks = json.loads(Path(evidence["checkpoints"]["path"]).read_text())
    checks["launcher_steps"] = exits["steps"]
    replace_reference(evidence["checkpoints"], checks)
    ledger = json.loads(Path(resources["submission_ledger"]["path"]).read_text())
    ledger["attempts"][index].update(declaration)
    replace_reference(resources["submission_ledger"], ledger)


def two_job_resource_fixture(directory):
    """Real tiny local originals for two synthetic serial allocations, no Slurm."""
    from m4_trace_fixture import resource_fixture

    resources = resource_fixture(directory)
    ledger = json.loads(Path(resources["submission_ledger"]["path"]).read_text())
    original = ledger["attempts"][0]
    second = copy.deepcopy(original)
    second.update(attempt_id="attempt-02", job_id="12346")
    allocation = copy.deepcopy(resources["allocations"][0])
    allocation["job_id"] = "12346"
    destination = Path(resources["submission_catalog_dir"]) / "attempt-02"
    destination.mkdir()

    def duplicate(reference):
        value = json.loads(
            Path(reference["path"])
            .read_text()
            .replace("12345", "12346")
            .replace("attempt-01", "attempt-02")
            .replace("T01:00:00", "T01:20:00")
            .replace("T01:10:00", "T01:30:00")
        )
        for key in ("observed_monotonic_ns", "started_monotonic_ns", "finished_monotonic_ns"):
            if key in value:
                value[key] += 1000
        updated = {"path": str(destination / Path(reference["path"]).name)}
        replace_reference(updated, value)
        return updated

    for name, reference in allocation["evidence"].items():
        allocation["evidence"][name] = duplicate(reference)
    second["submission"] = allocation["evidence"]["submission"]
    for name in ("monitor_ready", "monitor_exit", "monitor_trace"):
        second[name] = duplicate(second[name])
    ledger["attempts"].append(second)
    replace_reference(resources["submission_ledger"], ledger)
    resources["allocations"].append(allocation)
    resources["active_query"]["observed_monotonic_ns"] = 1450
    replace_reference(resources["active_query_evidence"], resources["active_query"])
    declare_execution_group(resources, "metadata", 0)
    declare_execution_group(resources, "gpu", 1)
    return resources


def run_two_job_resource_child(tmp_path, resources, label):
    path = tmp_path / f"{label}-resources.json"
    path.write_text(json.dumps(resources))
    command = [
        sys.executable,
        "-c",
        "import json,sys; from vllm_prefix_scheduler.m4_contract import audit_resources; "
        "print(json.dumps(audit_resources(json.load(open(sys.argv[1])), "
        "json.load(open(sys.argv[2])), selected_commit='a'*40, "
        "selected_jobs={'metadata':'12345','gpu':'12346'})))",
        str(path),
        str(ROOT / "configs/m4-stability.json"),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=10,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    (tmp_path / f"{label}-command.json").write_text(
        json.dumps({"command": command, "returncode": result.returncode})
    )
    (tmp_path / f"{label}-stdout.log").write_text(result.stdout)
    (tmp_path / f"{label}-stderr.log").write_text(result.stderr)
    return result


def test_frozen_matrix_cannot_be_shrunk():
    c = config()
    assert {layer: len(expected_cases(c, layer)) for layer in ("native", "gpu", "cpu")} == {
        "native": 105,
        "gpu": 60,
        "cpu": 540,
    }
    for key, value in (
        ("repeats", 1),
        ("native_rounds_per_seed", 99),
        ("seeds", [17]),
        ("window", True),
    ):
        changed = copy.deepcopy(c)
        changed[key] = value
        with pytest.raises(ValueError, match="frozen configuration"):
            expected_cases(changed, "gpu")
    changed = copy.deepcopy(c)
    changed["gpu_matrix"].pop()
    with pytest.raises(ValueError, match="GPU matrix"):
        expected_cases(changed, "gpu")


def test_complete_synthetic_gpu_bridge_is_only_contract_evidence():
    c = config()
    result = audit(c, gpu_fixture(c), layer="gpu", contract_fixture=True)
    assert result["case_count"] == 60
    assert result["scope"] == "contract_fixture"
    assert result["accepted"] is False
    assert result["complete_m4_acceptance"] is False


@pytest.mark.parametrize("mode", ["native", "off", "prefix", "guarded"])
@pytest.mark.parametrize("label", ["boundary", "before_shutdown"])
def test_gpu_resources_outside_cases_fail_closed(mode, label):
    c = config()
    events = gpu_fixture(c)
    checkpoint = copy.deepcopy(
        next(e for e in events if e["event"] == "m4_generation_drained" and e["mode"] == mode)
    )
    stream = [e for e in events if e["engine_id"] == checkpoint["engine_id"]]
    checkpoint.update(
        case_id=None,
        label=label,
        seq=0 if label == "boundary" else max(e["seq"] for e in stream) + 1,
        monotonic_ns=0 if label == "boundary" else max(e["monotonic_ns"] for e in stream) + 1000,
    )
    events.insert(0 if label == "boundary" else len(events), checkpoint)

    result = audit(c, events, layer="gpu", contract_fixture=True)
    assert result["case_count"] == 60
    assert result["scope"] == "contract_fixture"
    assert result["accepted"] is False
    assert result["complete_m4_acceptance"] is False

    checkpoint["resources"]["non_null_ref_sum"] = 1
    with pytest.raises(ValueError, match="native resource not zero: non_null_ref_sum"):
        audit(c, events, layer="gpu", contract_fixture=True)


@pytest.mark.parametrize("operation", ["empty", "missing", "duplicate", "mislabeled"])
def test_matrix_boundaries_fail_closed(operation):
    c = config()
    events = gpu_fixture(c)
    if operation == "empty":
        events = []
    elif operation == "missing":
        case = events[0]["case_id"]
        events = [e for e in events if e["case_id"] != case]
    elif operation == "duplicate":
        events.append(copy.deepcopy(events[0]))
    else:
        events[0]["mode"] = "guarded"
    with pytest.raises(ValueError):
        matrix_gate(c, events, "gpu")


def test_safe_reuse_has_independent_first_dispatch_and_age():
    meta = {
        "mode": "guarded",
        "scenario": "long_cycle",
        "variant": "complete_cancel_reuse",
        "repeat": 0,
        "seed": 17,
    }
    trace = Trace(meta, "safe")
    trace.complete("same-id")
    trace.complete("intervening-id")
    trace.complete("same-id")
    result = audit_lifecycle(trace.events, "guarded")
    assert result["guard_registrations"] == 3
    assert len(result["bindings"]) == 3
    assert all(r["repeated_prompt_tokens"] == 0 for r in result["dispatch"].values())
    bad = [e for e in copy.deepcopy(trace.events) if e["event"] != "m4_generation_drained"]
    with pytest.raises(ValueError, match="ID reused before"):
        audit_lifecycle(bad, "guarded")


def test_probe_equality_is_recomputed_from_raw_storage():
    events = [
        {
            "event": kind,
            "seq": seq,
            "probe_id": "probe",
            "score": 0,
            "flags": {"full": True, "events": True, "stats": True, "metrics": True},
            "snapshot": {"free": [1, 2], "stats": {"hits": 3}},
        }
        for seq, kind in enumerate(("m4_probe_before", "m4_probe_after"), 1)
    ]
    audit_probe(events)
    disabled = copy.deepcopy(events)
    disabled[0]["flags"]["events"] = False
    with pytest.raises(ValueError, match="probe profile not active"):
        audit_probe(disabled)
    events[1]["snapshot"]["stats"]["hits"] += 1
    with pytest.raises(ValueError, match="probe changed"):
        audit_probe(events)


def test_missing_runtime_rejected_without_gpu_dependency_import():
    with pytest.raises(ValueError, match="source/runtime"):
        source_gate(None, config(), "gpu")
    assert "vllm" not in sys.modules
    assert "vllm_prefix_scheduler.scheduler" not in sys.modules


@pytest.mark.parametrize(
    "args,environment", [(["-O"], {}), (["-OO"], {}), ([], {"PYTHONOPTIMIZE": "1"})]
)
def test_optimized_child_refuses_before_argument_or_dependency_work(args, environment):
    child = subprocess.run(
        [sys.executable, *args, str(ROOT / "scripts/check_m4_trace.py")],
        text=True,
        capture_output=True,
        env={**os.environ, **environment},
    )
    assert child.returncode != 0
    assert "assertions enabled" in child.stderr


def test_resource_audit_requires_each_original_and_full_terminal(tmp_path):
    from m4_trace_fixture import resource_fixture

    from vllm_prefix_scheduler.m4_contract import audit_resources

    resources = resource_fixture(tmp_path / "originals")
    declare_execution_group(resources, "all")
    result = audit_resources(resources, config(), selected_commit="a" * 40, selected_job="12345")
    assert result["allocations"] == 1 and result["requested_minutes"] == 20
    bad = copy.deepcopy(resources)
    bad["allocations"][0]["evidence"].pop("terminal")
    with pytest.raises(ValueError, match="terminal/checkpoint"):
        audit_resources(bad, config())
    bad = copy.deepcopy(resources)
    bad["allocations"][0]["limit_minutes"] = 21
    with pytest.raises(ValueError, match="20 minute"):
        audit_resources(bad, config())
    bad = copy.deepcopy(resources)
    bad["allocations"] *= 4
    with pytest.raises(ValueError, match="budget"):
        audit_resources(bad, config())
    receipt = Path(resources["allocations"][0]["evidence"]["terminal"]["path"])
    receipt.write_text("{}")
    with pytest.raises(ValueError, match="original missing/changed"):
        audit_resources(resources, config())


def test_unregistered_failed_submission_is_not_dropped(tmp_path):
    from m4_trace_fixture import resource_fixture

    from vllm_prefix_scheduler.m4_contract import audit_resources

    resources = resource_fixture(tmp_path / "originals")
    extra = Path(resources["submission_catalog_dir"]) / "attempt-02"
    extra.mkdir()
    (extra / "submission.json").write_text(json.dumps({"returncode": 2, "stdout": ""}))
    with pytest.raises(ValueError, match="omitted/unregistered"):
        audit_resources(resources, config())


@pytest.mark.parametrize("mode", ["prefix", "guarded"])
def test_window_tail_requires_actual_entry_of_original_outside_objects(mode):
    from vllm_prefix_scheduler.m4_contract import audit_barrier

    requests = [
        {"request_id": f"q{i}", "object_id": f"object:{i}", "generation": i} for i in range(18)
    ]
    records = (
        [{**request, "enqueued_ns": i + 1, "sequence": i} for i, request in enumerate(requests)]
        if mode == "guarded"
        else []
    )
    tail_records = records[16:] if records else []
    events = [
        {
            "event": "m4_barrier_check",
            "seq": 1,
            "before": requests,
            "after": requests,
            "candidates": requests[:16],
            "guard_before": records,
            "guard_after": records,
        },
        {"event": "m4_window", "seq": 2, "waiting": requests[16:], "candidates": requests[16:]},
        {
            "event": "m4_tail_entered",
            "seq": 3,
            "records_before": tail_records,
            "records_after": tail_records,
        },
    ]
    audit_barrier(events, "window_tail_age")
    with pytest.raises(ValueError, match="exactly one m4_tail_entered"):
        audit_barrier(events[:-1], "window_tail_age")
    wrong = copy.deepcopy(events)
    wrong[1]["waiting"] = requests[:2]
    wrong[1]["candidates"] = requests[:2]
    with pytest.raises(ValueError, match="earlier outside-window objects"):
        audit_barrier(wrong, "window_tail_age")


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("cleanup_errors", "allocation cleanup errors remain"),
        ("steps_mismatch", "allocation step originals differ"),
        ("step_remaining", "allocation step owned processes remain"),
    ],
)
def test_failed_nonselected_allocation_requires_full_cleanup_in_real_child(
    tmp_path, mutation, error
):
    """Local resource contract subprocesses; no allocation/native/GPU execution."""
    from m4_trace_fixture import resource_fixture

    from vllm_prefix_scheduler.m4_contract import sha256

    resources = resource_fixture(tmp_path / "originals")
    allocation = resources["allocations"][0]
    evidence = allocation["evidence"]

    def replace(reference, value):
        path = Path(reference["path"])
        path.write_text(json.dumps(value) + "\n")
        reference["sha256"] = sha256(path)

    terminal = json.loads(Path(evidence["terminal"]["path"]).read_text())
    terminal["stdout"] = (
        terminal["stdout"]
        .replace("JobState=COMPLETED", "JobState=FAILED")
        .replace("ExitCode=0:0", "ExitCode=1:0")
    )
    replace(evidence["terminal"], terminal)
    allocation.update(state="FAILED", exit_code="1:0")
    ledger_ref = resources["submission_ledger"]
    ledger = json.loads(Path(ledger_ref["path"]).read_text())
    watch_ref = ledger["attempts"][0]["monitor_trace"]
    replace(watch_ref, terminal)  # A one-line JSON object is also the complete watch JSONL.
    replace(ledger_ref, ledger)
    failed_steps = [{"name": "native", "returncode": 1, "status": "exited", "remaining": []}]
    exits = json.loads(Path(evidence["exit"]["path"]).read_text())
    exits.update(returncode=1, steps=failed_steps)
    replace(evidence["exit"], exits)
    checks = json.loads(Path(evidence["checkpoints"]["path"]).read_text())
    checks.update(launcher_steps=copy.deepcopy(failed_steps), cleanup_errors=[])
    replace(evidence["checkpoints"], checks)
    index = tmp_path / "resources.json"
    index.write_text(json.dumps(resources))
    command = [
        sys.executable,
        str(ROOT / "scripts/check_m4_resources.py"),
        "--config",
        str(ROOT / "configs/m4-stability.json"),
        "--resources",
        str(index),
    ]
    positive = subprocess.run(command, text=True, capture_output=True, timeout=10)
    (tmp_path / "positive-stdout.log").write_text(positive.stdout)
    (tmp_path / "positive-stderr.log").write_text(positive.stderr)
    assert positive.returncode == 0, positive.stderr
    assert json.loads(positive.stdout)["allocations"] == 1

    # Keep this failed historical/calibration job nonselected. Every modified
    # original reference is rebound, so rejection must reach the cleanup gate.
    if mutation == "cleanup_errors":
        checks["cleanup_errors"] = ["local synthetic cleanup failure"]
    elif mutation == "steps_mismatch":
        checks["launcher_steps"][0]["returncode"] = 2
    else:
        checks["launcher_steps"][0]["remaining"] = [123]
        exits["steps"] = copy.deepcopy(checks["launcher_steps"])
        replace(evidence["exit"], exits)
    replace(evidence["checkpoints"], checks)
    index.write_text(json.dumps(resources))
    negative = subprocess.run(command, text=True, capture_output=True, timeout=10)
    (tmp_path / "negative-stdout.log").write_text(negative.stdout)
    (tmp_path / "negative-stderr.log").write_text(negative.stderr)
    assert negative.returncode != 0
    assert error in negative.stderr


@pytest.mark.parametrize("signal_exit", [-1, -15])
def test_signal_interrupted_submission_is_ambiguous_not_known_failure(tmp_path, signal_exit):
    from m4_trace_fixture import resource_fixture

    from vllm_prefix_scheduler.m4_contract import audit_resources, sha256

    resources = resource_fixture(tmp_path / "originals")
    ledger_path = Path(resources["submission_ledger"]["path"])
    ledger = json.loads(ledger_path.read_text())
    second = Path(resources["submission_catalog_dir"]) / "attempt-02"
    second.mkdir()
    domain = "monotonic_ns:controller:fixture"

    def write(path, value):
        path.write_text(json.dumps(value) + "\n")
        return {"path": str(path), "sha256": sha256(path)}

    receipt = {
        "returncode": 2,
        "stdout": "",
        "command": ["sbatch", "local-fixture.sbatch"],
        "clock_domain": domain,
        "started_monotonic_ns": 700,
        "finished_monotonic_ns": 800,
    }
    attempt = {
        "attempt_id": "attempt-02",
        "project_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "status": "submission_failed",
        "job_id": None,
        "submission": write(second / "submission.json", receipt),
        "monitor_ready": write(
            second / "ready.json",
            {
                "ready": True,
                "campaign_id": "M4-scheduler-stability",
                "campaign_sha256": "b" * 64,
                "attempt_id": "attempt-02",
                "project_commit": "a" * 40,
                "clock_domain": domain,
                "observed_monotonic_ns": 600,
                "pid": 2,
            },
        ),
        "monitor_exit": write(
            second / "watcher-exit.json",
            {
                "returncode": 3,
                "pid": 2,
                "owned_pids_remaining": [],
                "clock_domain": domain,
                "observed_monotonic_ns": 900,
            },
        ),
    }
    ledger["attempts"].append(attempt)
    resources["submission_ledger"] = write(ledger_path, ledger)
    assert audit_resources(resources, config())["attempts"] == 2
    receipt["returncode"] = signal_exit
    attempt["submission"] = write(second / "submission.json", receipt)
    resources["submission_ledger"] = write(ledger_path, ledger)
    with pytest.raises(ValueError, match="ambiguous submission cannot be treated"):
        audit_resources(resources, config())


def test_two_serial_same_freeze_jobs_cover_exact_final_groups_in_real_child(tmp_path):
    resources = two_job_resource_fixture(tmp_path / "originals")
    result = run_two_job_resource_child(tmp_path, resources, "positive")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["requested_minutes"] == 40 and report["allocations"] == 2
    assert report["selected_jobs"] == {
        "12345": {
            "execution_group": "metadata",
            "declared_steps": ["native", "cpu", "regressions"],
        },
        "12346": {"execution_group": "gpu", "declared_steps": ["inputs", "gpu"]},
    }


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("missing_layer", "selected allocation lacks its exact execution group"),
        ("failed_selected", "selected frozen allocation did not complete"),
        ("cross_freeze", "selected raw job commit differs"),
        ("cross_config", "selected raw job config differs"),
        ("wrong_stage", "selected allocation lacks its exact execution group"),
        ("calibration", "selected allocation is not a formal run"),
        ("undeclared_group", "allocation execution group/declaration differs"),
    ],
)
def test_two_job_selection_rejects_rebound_invalid_originals_in_real_child(
    tmp_path, mutation, error
):
    resources = two_job_resource_fixture(tmp_path / "originals")
    positive = run_two_job_resource_child(tmp_path, resources, "positive")
    assert positive.returncode == 0, positive.stderr
    evidence = resources["allocations"][0]["evidence"]
    exits = json.loads(Path(evidence["exit"]["path"]).read_text())
    checks = json.loads(Path(evidence["checkpoints"]["path"]).read_text())
    binding = json.loads(Path(evidence["binding"]["path"]).read_text())
    ledger = json.loads(Path(resources["submission_ledger"]["path"]).read_text())
    attempt = ledger["attempts"][0]
    if mutation == "missing_layer":
        exits["steps"].pop()  # The selected metadata job never executed regressions.
        checks["launcher_steps"] = exits["steps"]
    elif mutation == "failed_selected":
        exits["returncode"] = exits["steps"][0]["returncode"] = 1
        checks["launcher_steps"] = exits["steps"]
        terminal = json.loads(Path(evidence["terminal"]["path"]).read_text())
        terminal["stdout"] = (
            terminal["stdout"]
            .replace("JobState=COMPLETED", "JobState=FAILED")
            .replace("ExitCode=0:0", "ExitCode=1:0")
        )
        replace_reference(evidence["terminal"], terminal)
        replace_reference(attempt["monitor_trace"], terminal)
        resources["allocations"][0].update(state="FAILED", exit_code="1:0")
    elif mutation == "cross_freeze":
        attempt["project_commit"] = binding["project_commit"] = exits["project_commit"] = "c" * 40
        ready = json.loads(Path(attempt["monitor_ready"]["path"]).read_text())
        ready["project_commit"] = "c" * 40
        replace_reference(attempt["monitor_ready"], ready)
    elif mutation == "cross_config":
        attempt["config_sha256"] = binding["config_sha256"] = "c" * 64
        ready = json.loads(Path(attempt["monitor_ready"]["path"]).read_text())
        ready["campaign_sha256"] = "c" * 64
        replace_reference(attempt["monitor_ready"], ready)
    elif mutation == "wrong_stage":
        binding["execution_group"] = exits["execution_group"] = "gpu"
        binding["declared_steps"] = exits["declared_steps"] = ["inputs", "gpu"]
        exits["steps"] = [
            {"name": name, "returncode": 0, "status": "exited", "remaining": []}
            for name in ("inputs", "gpu")
        ]
        checks["launcher_steps"] = exits["steps"]
    elif mutation == "calibration":
        attempt["phase"] = binding["phase"] = exits["phase"] = "calibration"
    else:
        binding.pop("execution_group")
        binding.pop("declared_steps")
        exits.pop("execution_group")
        exits.pop("declared_steps")
    replace_reference(evidence["exit"], exits)
    replace_reference(evidence["checkpoints"], checks)
    replace_reference(evidence["binding"], binding)
    replace_reference(resources["submission_ledger"], ledger)
    result = run_two_job_resource_child(tmp_path, resources, "negative")
    assert result.returncode != 0
    assert error in result.stderr


@pytest.mark.parametrize(
    "selection",
    [
        {"metadata": "12345"},
        {"gpu": "12346"},
        {"metadata": "12345", "gpu": "12345"},
        {"metadata": "12345", "gpu": "12346", "extra": "12347"},
        {"metadata": "12345", "gpu": "missing"},
    ],
)
def test_final_resource_role_selection_is_not_arbitrary(tmp_path, selection):
    from vllm_prefix_scheduler.m4_contract import audit_resources

    resources = two_job_resource_fixture(tmp_path / "originals")
    with pytest.raises(ValueError, match="two distinct metadata/gpu jobs"):
        audit_resources(resources, config(), selected_commit="a" * 40, selected_jobs=selection)


def test_gpu_layer_selects_exact_gpu_group_and_default_still_requires_all_five(tmp_path):
    from vllm_prefix_scheduler.m4_contract import audit_resources

    resources = two_job_resource_fixture(tmp_path / "originals")
    report = audit_resources(
        resources,
        config(),
        selected_commit="a" * 40,
        selected_job="12346",
        required_steps=["inputs", "gpu"],
    )
    assert set(report["selected_jobs"]) == {"12346"}
    with pytest.raises(ValueError, match="selected allocation lacks its exact execution group"):
        audit_resources(resources, config(), selected_commit="a" * 40, selected_job="12346")
    with pytest.raises(ValueError, match="not a fixed group"):
        audit_resources(
            resources,
            config(),
            selected_commit="a" * 40,
            selected_job="12346",
            required_steps=["gpu"],
        )


def test_execution_group_catalog_cannot_be_removed_or_shrunk():
    from vllm_prefix_scheduler.m4_contract import config_gate

    original = config()
    config_gate(original)
    for group in (None, "metadata", "gpu"):
        changed = copy.deepcopy(original)
        if group is None:
            changed.pop("execution_groups")
        else:
            changed["execution_groups"][group].pop()
        with pytest.raises(ValueError, match="execution_groups"):
            config_gate(changed)
