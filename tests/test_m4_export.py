"""Independent local export binding tests; no native or GPU execution claims.

Collectors and Git blobs are real small local fixtures. Only the external
subprocess boundary of fresh-control dispatch is replaced in the dispatch tests.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from vllm_prefix_scheduler.run_evidence import check_run, collect_run

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("m4_export_under_test", ROOT / "scripts/export_m4.py")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


@pytest.fixture
def frozen_repository(tmp_path):
    root = tmp_path / "source-fixture"
    root.mkdir()
    files = {
        ".gitignore": "/runs\n",
        "configs/upstream.json": '{"vllm":{"commit":"local-contract-fixture"}}\n',
        "configs/m4-stability.json": '{"scope":"local-export-test-only"}\n',
        "scripts/export_m4.py": "# Frozen local exporter fixture.\n",
        "scripts/check_m4_trace.py": "# Frozen local checker fixture.\n",
        "src/vllm_prefix_scheduler/m4_contract.py": "# Frozen local auditor fixture.\n",
        "tests/test_m4_export.py": "# Frozen local test fixture.\n",
        "Makefile": "check:\n\t@true\n",
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=M4 local export fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "freeze local fixture",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    return root, commit


@pytest.fixture
def sealed_collector(frozen_repository, tmp_path, monkeypatch):
    root, commit = frozen_repository
    raw = tmp_path / "actual-input" / "trace.jsonl"
    raw.parent.mkdir()
    raw.write_text('{"scope":"local-original-only","value":1}\n')
    # This is explicitly kind=local. The fabricated environment ID only exercises
    # identity binding; no Slurm submission, hardware query or GPU code is called.
    job = "12345"
    monkeypatch.setenv("SLURM_JOB_ID", job)
    directory = root / "runs" / "local-collector"
    manifest, code = collect_run(
        root,
        directory,
        [sys.executable, "-c", "print('local collector binding fixture')"],
        task_id="M4-local-export-contract",
        kind="local",
        timeout=5,
        configs=[],
        artifacts=[raw],
        criteria=["Retain one actual local artifact"],
    )
    assert code == 0 and manifest["schema_version"] == 2 and manifest["kind"] == "local"
    assert check_run(directory, require_success=True)["integrity"] == "valid"
    return directory, commit, job, raw


def test_collector_binds_actual_sealed_raw_bytes(sealed_collector):
    directory, commit, job, raw = sealed_collector
    bound = exporter.bind_collector(directory, commit, job, [raw])
    assert bound[str(raw)] == exporter.sha256(raw)
    assert bound[str(directory / "manifest.json")] == exporter.sha256(directory / "manifest.json")


def test_valid_collector_cannot_back_different_same_named_raw_bytes(sealed_collector, tmp_path):
    directory, commit, job, raw = sealed_collector
    changed = tmp_path / "other-input" / raw.name
    changed.parent.mkdir()
    changed.write_text('{"scope":"different-local-input","value":999}\n')
    sealed_manifest_before = (directory / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="not the collector sealed artifact"):
        exporter.bind_collector(directory, commit, job, [changed])
    # The rejection is about detachment, not an invalid or corrupted collector.
    assert check_run(directory, require_success=True)["integrity"] == "valid"
    assert (directory / "manifest.json").read_bytes() == sealed_manifest_before
    assert raw.read_text() == '{"scope":"local-original-only","value":1}\n'


@pytest.mark.parametrize("mismatch", ["commit", "job"])
def test_valid_collector_rejects_different_freeze_or_job(sealed_collector, mismatch):
    directory, commit, job, raw = sealed_collector
    requested_commit = "f" * 40 if mismatch == "commit" else commit
    requested_job = "99999" if mismatch == "job" else job
    with pytest.raises(ValueError, match="collector freeze/job/stability differs"):
        exporter.bind_collector(directory, requested_commit, requested_job, [raw])
    assert check_run(directory, require_success=True)["integrity"] == "valid"


def test_execution_binding_reads_actual_tested_git_blobs(frozen_repository):
    root, commit = frozen_repository
    bound = exporter.bind_execution(commit, root=root)
    expected = {
        str(root / name)
        for name in (
            "configs/upstream.json",
            "configs/m4-stability.json",
            "scripts/export_m4.py",
            "scripts/check_m4_trace.py",
            "src/vllm_prefix_scheduler/m4_contract.py",
            "tests/test_m4_export.py",
            "Makefile",
        )
    }
    assert set(bound) == expected
    assert all(exporter.sha256(Path(path)) == checksum for path, checksum in bound.items())


@pytest.mark.parametrize(
    "name", ["scripts/export_m4.py", "src/vllm_prefix_scheduler/m4_contract.py"]
)
def test_execution_binding_rejects_current_exporter_or_auditor_drift(frozen_repository, name):
    root, commit = frozen_repository
    exporter.bind_execution(commit, root=root)
    (root / name).write_text("# Changed after the tested commit.\n")
    with pytest.raises(ValueError, match="currently executing source differs from tested freeze"):
        exporter.bind_execution(commit, root=root)


def test_existing_control_reports_never_replace_fresh_execution(tmp_path, monkeypatch):
    destination = tmp_path / "already-exists"
    old = destination / "native" / "results.json"
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps({"all_controls_passed": True, "controls": [{}] * 17}))
    original = old.read_bytes()

    def unexpected_subprocess(*_args, **_kwargs):
        pytest.fail("Existing control destination should be rejected before executing anything")

    monkeypatch.setattr(exporter.subprocess, "run", unexpected_subprocess)
    with pytest.raises(FileExistsError):
        exporter.rerun_controls(
            tmp_path / "config.json", tmp_path / "data", tmp_path / "resources.json", destination
        )
    assert old.read_bytes() == original


def test_fresh_controls_invoke_fixed_tested_checkers_and_retain_commands(tmp_path, monkeypatch):
    config, data, resources = (
        tmp_path / "config.json",
        tmp_path / "data",
        tmp_path / "resources.json",
    )
    destination = tmp_path / "fresh-controls"
    calls = []
    expected_counts = {"native": 17, "cpu": 5, "gpu": 18, "resources": 9}

    def external_control_process(command, **kwargs):
        # Only dispatch is under test here; these reports are synthetic responses
        # from the replaced external command, not accepted control executions.
        assert kwargs == {"capture_output": True, "text": True, "timeout": 1800}
        output = Path(command[command.index("--output") + 1])
        layer = output.name
        checker = (
            "check_m4_resources_fail_closed.py"
            if layer == "resources"
            else "check_m4_fail_closed.py"
        )
        assert command[:2] == [sys.executable, str(ROOT / "scripts" / checker)]
        assert command[command.index("--config") + 1] == str(config)
        assert "--contract-fixture" not in command and "--partial" not in command
        if layer == "resources":
            assert command[command.index("--resources") + 1] == str(resources)
        else:
            assert command[command.index("--layer") + 1] == layer
            assert command[command.index("--trace") + 1] == str(data / f"{layer}-data/trace.jsonl")
            assert command[command.index("--sources") + 1] == str(
                data / f"{layer}-data" / ("sources.json" if layer == "gpu" else "runtime.json")
            )
            if layer == "gpu":
                assert command[command.index("--inputs") + 1] == str(data / "gpu-inputs.json")
                assert command[command.index("--resources") + 1] == str(resources)
        output.mkdir(parents=True, exist_ok=False)
        (output / "results.json").write_text(
            json.dumps(
                {
                    "scope": "local-dispatch-test-only",
                    "all_controls_passed": True,
                    "controls": [{"local_test_only": True}] * expected_counts[layer],
                }
            )
        )
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="local dispatch fixture\n", stderr="")

    monkeypatch.setattr(exporter.subprocess, "run", external_control_process)
    counts, bound = exporter.rerun_controls(config, data, resources, destination)
    assert counts == expected_counts and len(calls) == 4
    for layer in expected_counts:
        record = json.loads((destination / f"{layer}-command.json").read_text())
        assert record["returncode"] == 0 and record["command"] in calls
        assert str(destination / layer / "results.json") in bound
    assert all(exporter.sha256(Path(path)) == checksum for path, checksum in bound.items())


def test_fresh_control_process_failure_is_not_accepted(tmp_path, monkeypatch):
    calls = []

    def failed_process(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 7, stdout="", stderr="local forced child failure\n"
        )

    monkeypatch.setattr(exporter.subprocess, "run", failed_process)
    destination = tmp_path / "failed-fresh-controls"
    with pytest.raises(ValueError, match="fresh fixed-source subprocess controls failed: native"):
        exporter.rerun_controls(
            tmp_path / "config.json", tmp_path / "data", tmp_path / "resources.json", destination
        )
    assert len(calls) == 1
    assert (destination / "native-stderr.log").read_text() == "local forced child failure\n"
    assert not (destination / "native" / "results.json").exists()


def test_regression_job_uses_existing_m1_m2b_runtime_environment_schema():
    # Field layout matches all three actual attempt01 regression runtime originals.
    runtime = {
        "project_commit": "a" * 40,
        "environment": {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HUB_OFFLINE": "1",
            "PYTHONHASHSEED": "42",
            "SLURM_JOB_ID": "12345",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
        },
    }
    assert exporter.bind_regression_job(runtime, "a" * 40) == "12345"
    for changed in (
        {**runtime, "project_commit": "b" * 40},
        {"project_commit": "a" * 40, "job_id": "12345"},
        {**runtime, "environment": {}},
        {**runtime, "environment": {"SLURM_JOB_ID": "unknown"}},
        {**runtime, "job_id": "99999"},
    ):
        with pytest.raises(ValueError):
            exporter.bind_regression_job(changed, "a" * 40)


def test_cpu_public_costs_recompute_fixed_numeric_allowlist_without_private_extras():
    secret = "PRIVATE_HOST_PATH_JOB_ID_MUST_NOT_LEAK"
    events = [
        {
            "event": "m4_cpu_environment",
            "machine": secret,
            "unit": secret,
            "processor": secret,
            "clock": {"implementation": secret},
            "empty_wrapper_ns": {
                **{
                    key: [1] * 500 + [100] * 500
                    for key in ("bare_call_ns", "timed_wrapper_call_ns", "clock_pair_ns")
                },
                "private_extra": secret,
            },
        }
    ]
    for mode in ("native", "off", "prefix", "guarded"):
        for workload in ("cold", "tied_hot", "mixed"):
            for window in (8, 16, 32):
                for outside in (0, 128, 4096):
                    meta = {
                        "mode": mode,
                        "workload": workload,
                        "window": window,
                        "outside_length": outside,
                        "job_id": secret,
                        "path": secret,
                        "extra": secret,
                    }
                    normal = {
                        **meta,
                        "event": "m4_cpu_sample",
                        "variant": "normal",
                        "calls": {**dict.fromkeys(exporter.CPU_CALLS, 2), "private": secret},
                        "visits": {**dict.fromkeys(exporter.CPU_VISITS, 3), "private": secret},
                    }
                    # This is a numerical export unit fixture, not a native CPU run.
                    events.extend([normal] * 500)
                    events.append(
                        {**meta, "event": "m4_cpu_unwrapped", "samples_ns": [1] * 50 + [100] * 50}
                    )
                    if mode not in {"prefix", "guarded"} or workload != "mixed":
                        continue
                    for duration in (5, 1, 9, 3, 7):
                        events.append(
                            {
                                **meta,
                                "event": "m4_cpu_sample",
                                "variant": "prepend_restore",
                                "duration_ns": {
                                    **dict.fromkeys(exporter.CPU_PHASES, duration),
                                    "private": secret,
                                },
                            }
                        )
                        events.append(
                            {
                                **meta,
                                "event": "m4_cpu_restore_counter",
                                "counts": {
                                    "popleft_calls": 1,
                                    "popleft_entries": 1,
                                    "extendleft_calls": 1,
                                    "extendleft_entries": window,
                                    "private": secret,
                                },
                            }
                        )
    public = exporter.public_cpu_costs(events)
    assert secret not in json.dumps(public)
    assert set(public["empty_wrapper"]) == {
        "bare_call_ns",
        "timed_wrapper_call_ns",
        "clock_pair_ns",
    }
    assert all(
        item == {"n": 1000, "p50_ns": 1, "p95_ns": 100} for item in public["empty_wrapper"].values()
    )
    assert len(public["unwrapped"]) == len(public["normal_operation_counts"]) == 108
    assert len(public["prepend_restore"]) == len(public["restore_operation_counts"]) == 18
    assert public["restore_counter_samples"] == 90
    assert public["unwrapped"][0]["p50_ns"] == 1  # Nearest rank, not the mean of two middle values.
    assert public["unwrapped"][0]["n"] == 100
    assert public["normal_operation_counts"][0]["n"] == 500
    assert set(public["normal_operation_counts"][0]["calls"]) == set(exporter.CPU_CALLS)
    assert set(public["normal_operation_counts"][0]["visits"]) == set(exporter.CPU_VISITS)
    assert public["normal_operation_counts"][0]["calls"]["score"] == {"min": 2, "max": 2}
    assert set(public["prepend_restore"][0]["phases"]) == set(exporter.CPU_PHASES)
    assert public["prepend_restore"][0]["phases"]["total"] == {"n": 5, "p50_ns": 5, "p95_ns": 9}
    assert set(public["restore_operation_counts"][0]["counts"]) == set(exporter.CPU_RESTORE_COUNTS)
    assert public["duration_unit"] == "nanoseconds"
    assert public["measurement_units"] == {
        "native_off": "one PAUSED_ALL native schedule",
        "prefix_guarded": "one waiting-window transaction",
    }


@pytest.fixture
def layer_job_evidence():
    # Ordinary JSON-compatible records exercise provenance binding only. These
    # records make no claim that a native scheduler or GPU has been executed.
    commit = "a" * 40
    return {
        "commit": commit,
        "metadata_job": "12345",
        "gpu_job": "12346",
        "sources": {"project_commit": commit, "job_id": "12345"},
        "events": [
            {"event": "local_binding_begin", "job_id": "12345", "project_commit": commit},
            {"event": "local_binding_end", "job_id": "12345", "project_commit": commit},
        ],
    }


@pytest.mark.parametrize("layer", ["native", "cpu"])
def test_layer_job_binds_every_metadata_record(layer_job_evidence, layer):
    evidence = layer_job_evidence
    assert (
        exporter.bind_layer_job(layer, evidence["events"], evidence["sources"], evidence["commit"])
        == evidence["metadata_job"]
    )


@pytest.mark.parametrize("include_source_job", [False, True])
def test_gpu_layer_job_binds_one_campaign_job(layer_job_evidence, include_source_job):
    evidence = layer_job_evidence
    sources = {"project_commit": evidence["commit"]}
    if include_source_job:
        sources["job_id"] = evidence["gpu_job"]
    events = [
        {"event": "local_engine_start", "campaign_job_id": evidence["gpu_job"]},
        {"event": "local_worker_detail", "campaign_job_id": evidence["gpu_job"]},
        {"event": "local_engine_end", "campaign_job_id": evidence["gpu_job"]},
    ]
    assert exporter.bind_layer_job("gpu", events, sources, evidence["commit"]) == "12346"


@pytest.mark.parametrize("layer", ["native", "cpu"])
@pytest.mark.parametrize("invalid_job", [None, "", "not-a-job", "99999"])
def test_metadata_layer_rejects_any_unbound_or_foreign_row(layer_job_evidence, layer, invalid_job):
    evidence = layer_job_evidence
    if invalid_job is None:
        evidence["events"][1].pop("job_id")
    else:
        evidence["events"][1]["job_id"] = invalid_job
    with pytest.raises(ValueError):
        exporter.bind_layer_job(layer, evidence["events"], evidence["sources"], evidence["commit"])


@pytest.mark.parametrize("layer", ["native", "cpu"])
@pytest.mark.parametrize("row_commit", [None, "b" * 40])
def test_metadata_layer_rejects_any_missing_or_different_row_freeze(
    layer_job_evidence, layer, row_commit
):
    evidence = layer_job_evidence
    if row_commit is None:
        evidence["events"][1].pop("project_commit")
    else:
        evidence["events"][1]["project_commit"] = row_commit
    with pytest.raises(ValueError):
        exporter.bind_layer_job(layer, evidence["events"], evidence["sources"], evidence["commit"])


@pytest.mark.parametrize("layer", ["native", "cpu", "gpu"])
@pytest.mark.parametrize("source_commit", [None, "b" * 40])
def test_layer_job_rejects_missing_or_different_source_freeze(
    layer_job_evidence, layer, source_commit
):
    evidence = layer_job_evidence
    sources = {"job_id": evidence["metadata_job"]}
    if source_commit is not None:
        sources["project_commit"] = source_commit
    events = evidence["events"]
    if layer == "gpu":
        events = [{"campaign_job_id": evidence["metadata_job"]}]
    with pytest.raises(ValueError):
        exporter.bind_layer_job(layer, events, sources, evidence["commit"])


@pytest.mark.parametrize("layer", ["native", "cpu"])
@pytest.mark.parametrize("source_job", [None, "", "not-a-job", "99999"])
def test_metadata_layer_rejects_missing_invalid_or_foreign_source_job(
    layer_job_evidence, layer, source_job
):
    evidence = layer_job_evidence
    sources = {"project_commit": evidence["commit"]}
    if source_job is not None:
        sources["job_id"] = source_job
    with pytest.raises(ValueError):
        exporter.bind_layer_job(layer, evidence["events"], sources, evidence["commit"])


@pytest.mark.parametrize(
    "events",
    [
        [],
        [{"event": "local_record_without_campaign_job"}],
        [{"campaign_job_id": "12346"}, {"event": "local_row_missing_campaign_job"}],
        [{"campaign_job_id": ""}],
        [{"campaign_job_id": "not-a-job"}],
        [{"campaign_job_id": "12346"}, {"campaign_job_id": "99999"}],
    ],
)
def test_gpu_layer_rejects_missing_invalid_or_multiple_campaign_jobs(layer_job_evidence, events):
    evidence = layer_job_evidence
    with pytest.raises(ValueError):
        exporter.bind_layer_job(
            "gpu", events, {"project_commit": evidence["commit"]}, evidence["commit"]
        )


def test_gpu_layer_rejects_source_job_detached_from_campaign(layer_job_evidence):
    evidence = layer_job_evidence
    with pytest.raises(ValueError):
        exporter.bind_layer_job(
            "gpu",
            [{"campaign_job_id": evidence["gpu_job"]}],
            evidence["sources"],
            evidence["commit"],
        )


@pytest.mark.parametrize("layer", ["native", "cpu"])
def test_metadata_layer_rejects_empty_raw_evidence(layer_job_evidence, layer):
    evidence = layer_job_evidence
    with pytest.raises(ValueError):
        exporter.bind_layer_job(layer, [], evidence["sources"], evidence["commit"])


@pytest.fixture
def complete_job_binding():
    return (
        {"native": "12345", "cpu": "12345", "gpu": "12346"},
        {"compat": "12345", "m1": "12345", "full": "12345"},
    )


def test_final_jobs_bind_complete_metadata_and_distinct_gpu_allocations(complete_job_binding):
    layers, regressions = complete_job_binding
    assert exporter.bind_final_jobs(layers, regressions) == {
        "metadata": "12345",
        "gpu": "12346",
    }


@pytest.mark.parametrize("layer", ["native", "cpu", "gpu"])
def test_final_jobs_reject_any_missing_layer(complete_job_binding, layer):
    layers, regressions = complete_job_binding
    layers.pop(layer)
    with pytest.raises(ValueError):
        exporter.bind_final_jobs(layers, regressions)


@pytest.mark.parametrize("regression", ["compat", "m1", "full"])
def test_final_jobs_reject_any_missing_regression(complete_job_binding, regression):
    layers, regressions = complete_job_binding
    regressions.pop(regression)
    with pytest.raises(ValueError):
        exporter.bind_final_jobs(layers, regressions)


@pytest.mark.parametrize("target", ["layers", "regressions"])
def test_final_jobs_reject_extra_layer_or_regression(complete_job_binding, target):
    layers, regressions = complete_job_binding
    (layers if target == "layers" else regressions)["unexpected"] = "12345"
    with pytest.raises(ValueError):
        exporter.bind_final_jobs(layers, regressions)


def test_final_jobs_reject_cpu_detached_from_native_allocation(complete_job_binding):
    layers, regressions = complete_job_binding
    layers["cpu"] = "99999"
    with pytest.raises(ValueError):
        exporter.bind_final_jobs(layers, regressions)


@pytest.mark.parametrize("regression", ["compat", "m1", "full"])
def test_final_jobs_reject_regression_detached_from_metadata_allocation(
    complete_job_binding, regression
):
    layers, regressions = complete_job_binding
    regressions[regression] = "99999"
    with pytest.raises(ValueError):
        exporter.bind_final_jobs(layers, regressions)


def test_final_jobs_reject_reusing_metadata_job_as_gpu_job(complete_job_binding):
    layers, regressions = complete_job_binding
    layers["gpu"] = layers["native"]
    with pytest.raises(ValueError):
        exporter.bind_final_jobs(layers, regressions)


@pytest.mark.parametrize("job", ["", "not-a-job"])
def test_final_jobs_reject_nonnumeric_metadata_job_even_when_all_layers_agree(job):
    with pytest.raises(ValueError):
        exporter.bind_final_jobs(
            {"native": job, "cpu": job, "gpu": "12346"},
            {"compat": job, "m1": job, "full": job},
        )


@pytest.mark.parametrize("job", ["", "not-a-job"])
def test_final_jobs_reject_nonnumeric_gpu_job(complete_job_binding, job):
    layers, regressions = complete_job_binding
    layers["gpu"] = job
    with pytest.raises(ValueError):
        exporter.bind_final_jobs(layers, regressions)


AUDIT_PATHS = {
    "scripts/export_m4.py",
    "src/vllm_prefix_scheduler/m4_contract.py",
    "tests/test_m4_contract.py",
    "tests/test_m4_export.py",
}


def fixture_git(root, *arguments):
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def commit_fixture_changes(root):
    fixture_git(root, "add", ".")
    fixture_git(
        root,
        "-c",
        "user.name=M4 local export fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "local audit binding fixture",
    )
    return fixture_git(root, "rev-parse", "HEAD")


@pytest.fixture
def audit_repository(frozen_repository, monkeypatch):
    root, _original_commit = frozen_repository
    (root / "tests/test_m4_contract.py").write_text("# Frozen local contract test fixture.\n")
    (root / "scripts/m4_engine_fixture.py").write_text("# Frozen local producer fixture.\n")
    (root / "docs").mkdir()
    (root / "docs/review.md").write_text("Frozen local documentation fixture.\n")
    tested = commit_fixture_changes(root)
    monkeypatch.setattr(exporter, "EXECUTION_FREEZE", tested)
    assert set(exporter.AUDIT_ONLY_PATHS) == AUDIT_PATHS
    return root, tested


def commit_allowed_audit_changes(root):
    for name in sorted(AUDIT_PATHS):
        path = root / name
        path.write_text(path.read_text() + "# Committed local audit-only revision.\n")
    return commit_fixture_changes(root)


def test_explicit_analysis_binds_all_four_allowed_files_to_actual_git_blobs(audit_repository):
    root, tested = audit_repository
    tested_binding = exporter.bind_execution(tested, root=root)
    analysis = commit_allowed_audit_changes(root)
    bound = exporter.bind_execution(tested, root=root, analysis_commit=analysis)
    assert set(bound) == set(tested_binding)
    assert {
        str(Path(path).relative_to(root)) for path in bound if bound[path] != tested_binding[path]
    } == AUDIT_PATHS
    assert all(exporter.sha256(Path(path)) == digest for path, digest in bound.items())
    assert fixture_git(root, "cat-file", "-t", analysis) == "commit"


def test_default_execution_binding_still_rejects_committed_audit_changes(audit_repository):
    root, tested = audit_repository
    commit_allowed_audit_changes(root)
    with pytest.raises(ValueError, match="currently executing source differs from tested freeze"):
        exporter.bind_execution(tested, root=root)


def test_explicit_analysis_allows_documentation_only_change(audit_repository):
    root, tested = audit_repository
    original = exporter.bind_execution(tested, root=root)
    (root / "docs/review.md").write_text("Documented local analysis result.\n")
    analysis = commit_fixture_changes(root)
    assert exporter.bind_execution(tested, root=root, analysis_commit=analysis) == original
    assert exporter.bind_execution(tested, root=root) == original


@pytest.mark.parametrize(
    "reference", ["head", "abbreviated", "uppercase", "tree", "blob", "missing"]
)
def test_explicit_analysis_requires_full_lowercase_existing_commit(audit_repository, reference):
    root, tested = audit_repository
    analysis = commit_allowed_audit_changes(root)
    references = {
        "head": "HEAD",
        "abbreviated": analysis[:12],
        "uppercase": analysis.upper(),
        "tree": fixture_git(root, "rev-parse", f"{analysis}^{{tree}}"),
        "blob": fixture_git(root, "rev-parse", f"{analysis}:scripts/export_m4.py"),
        "missing": "f" * 40,
    }
    with pytest.raises(ValueError):
        exporter.bind_execution(tested, root=root, analysis_commit=references[reference])


def test_explicit_analysis_rejects_other_tested_commit_even_with_matching_bytes(audit_repository):
    root, tested = audit_repository
    analysis = commit_allowed_audit_changes(root)
    assert analysis != tested
    with pytest.raises(ValueError):
        exporter.bind_execution(analysis, root=root, analysis_commit=analysis)


@pytest.mark.parametrize(
    "name",
    [
        "scripts/m4_engine_fixture.py",
        "scripts/check_m4_trace.py",
        "configs/m4-stability.json",
        "Makefile",
    ],
)
def test_explicit_analysis_rejects_changes_outside_fixed_audit_paths(audit_repository, name):
    root, tested = audit_repository
    path = root / name
    path.write_text(path.read_text() + "\n# Changed outside the fixed audit allowlist.\n")
    analysis = commit_fixture_changes(root)
    with pytest.raises(ValueError):
        exporter.bind_execution(tested, root=root, analysis_commit=analysis)


@pytest.mark.parametrize("change", ["add", "delete"])
def test_explicit_analysis_rejects_committed_source_catalog_change(audit_repository, change):
    root, tested = audit_repository
    if change == "add":
        (root / "scripts/new_producer.py").write_text("# New tracked executable fixture.\n")
    else:
        (root / "tests/test_m4_export.py").unlink()
    analysis = commit_fixture_changes(root)
    with pytest.raises(ValueError):
        exporter.bind_execution(tested, root=root, analysis_commit=analysis)


@pytest.mark.parametrize("change", ["add", "delete"])
def test_explicit_analysis_rejects_current_index_catalog_change(audit_repository, change):
    root, tested = audit_repository
    analysis = commit_allowed_audit_changes(root)
    if change == "add":
        path = root / "scripts/index_only_producer.py"
        path.write_text("# Tracked after the selected analysis commit.\n")
        fixture_git(root, "add", str(path))
    else:
        # Leave the original bytes present so only the tracked catalog differs.
        fixture_git(root, "rm", "--cached", "tests/test_m4_export.py")
        assert (root / "tests/test_m4_export.py").is_file()
    with pytest.raises(ValueError):
        exporter.bind_execution(tested, root=root, analysis_commit=analysis)


@pytest.mark.parametrize("name", ["scripts/export_m4.py", "configs/m4-stability.json"])
def test_explicit_analysis_rejects_dirty_bytes_after_selected_analysis(audit_repository, name):
    root, tested = audit_repository
    analysis = commit_allowed_audit_changes(root)
    path = root / name
    path.write_text(path.read_text() + "# Uncommitted bytes after the analysis freeze.\n")
    with pytest.raises(ValueError):
        exporter.bind_execution(tested, root=root, analysis_commit=analysis)


def test_explicit_analysis_rejects_worktree_matching_tested_instead_of_analysis(audit_repository):
    root, tested = audit_repository
    path = root / "scripts/export_m4.py"
    original = path.read_bytes()
    analysis = commit_allowed_audit_changes(root)
    path.write_bytes(original)
    with pytest.raises(ValueError):
        exporter.bind_execution(tested, root=root, analysis_commit=analysis)
