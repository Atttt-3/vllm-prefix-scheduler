"""Offline release controls and synthetic review bundles. No formal results are published."""

import json
import shutil
import sys
from pathlib import Path

import pytest
from test_m5_contract import call_run, put, setup_capture, setup_capture_v2

from vllm_prefix_scheduler.m5_campaign import (
    BASELINE,
    CALIBRATION,
    MODES,
    SCENARIOS,
    TASK,
    config_matrix,
    digest,
    qualification_roles,
    read_json,
)
from vllm_prefix_scheduler.m5_contract import audit_run, comparable, paired_summary

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import export_m5  # noqa: E402
import m5_c3_bundle  # noqa: E402


def test_split_c3_bundle_binds_leaf_bytes_and_resource_releases(tmp_path, monkeypatch):
    config = ROOT / "configs/m5-performance.json"
    commit = "a" * 40
    roles = qualification_roles(read_json(config))
    stages = {role: tmp_path / role for role in roles}
    for attempt in stages.values():
        attempt.mkdir()
        put(attempt / "authorization.json", {"d2r110_analysis_sha256": "d" * 64})
    for i, stage in enumerate(("C2", "C3-burst", "C3-mixed")):
        attempt = tmp_path / stage
        node = attempt / "node"
        leaf = node / "leaf"
        (leaf / "raw").mkdir(parents=True)
        (leaf / "collector").mkdir()
        stages[stage] = attempt
        put(
            attempt / "planned.json",
            {
                "role": stage,
                "phase": "calibration",
                "execution_commit": commit,
                "config_sha256": digest(config),
            },
        )
        put(node / "allocation-sources.json", {"execution_commit": commit})
        put(attempt / "terminal.json", {"job": i, "state": "COMPLETED"})
        audit = put(leaf / "audit.json", {})
        table = put(leaf / "input.json", {})
        put(
            node / "index.json",
            {
                "stage": stage,
                "runs": [{"id": "leaf", "audit_sha256": digest(audit), "elapsed_seconds": 1}],
            },
        )
        put(
            node / "calibration-result.json",
            {
                "stage": stage,
                "status": "qualified",
                "config_sha256": digest(config),
                "selected_rates": {"burst": 3, "mixed": 8},
                "runs": {
                    "leaf": {
                        "audit_sha256": digest(audit),
                        "input_sha256": digest(table),
                        "elapsed_seconds": 1,
                    }
                },
            },
        )
        for filename in (
            "raw/client.json",
            "raw/engine.json",
            "raw/sources.json",
            "collector/manifest.json",
            "collector/acceptance.json",
        ):
            put(leaf / filename, {})

    def release(attempt, *, require_completed):
        assert require_completed is True
        terminal = attempt / "terminal.json"
        return {"job_id": read_json(terminal)["job"], "terminal_sha256": digest(terminal)}

    monkeypatch.setattr(m5_c3_bundle, "audit_attempt", release)
    monkeypatch.setattr(m5_c3_bundle, "verify_d2r110_qualification", lambda *_args: "d" * 64)
    monkeypatch.setattr(
        m5_c3_bundle, "attempt_evidence_paths", lambda attempt: [attempt / "terminal.json"]
    )
    monkeypatch.setattr(
        m5_c3_bundle, "aggregate_c3", lambda *_args, **_kwargs: {"status": "qualified"}
    )

    def binding():
        return m5_c3_bundle.c3_bundle_digest(
            m5_c3_bundle.c3_bundle(config, tmp_path, stages, commit)
        )

    original = binding()
    target = stages["C3-burst"] / "node/leaf/raw/client.json"
    put(target, {"different": True})
    assert binding() != original
    put(target, {})
    assert binding() == original
    auth = stages["C4"] / "authorization.json"
    put(auth, {"d2r110_analysis_sha256": "e" * 64})
    with pytest.raises(ValueError, match="bind different D2-R110 reviews"):
        binding()
    put(auth, {"d2r110_analysis_sha256": "d" * 64})
    put(stages["C3-burst"] / "terminal.json", {"job": 2, "state": "COMPLETED"})
    with pytest.raises(ValueError, match="share an allocation identity"):
        binding()


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    config_path = ROOT / "configs/m5-performance.json"
    matrix = config_matrix(read_json(config_path))
    freeze = {
        "schema_version": 1,
        "task_id": TASK,
        "execution_commit": "a" * 40,
        "config_sha256": digest(config_path),
        "calibration_sha256": dict.fromkeys(CALIBRATION, "c" * 64),
        "tables": {pair: f"{index:064x}" for index, pair in enumerate(matrix)},
        "rates_rps": dict.fromkeys(SCENARIOS.keys() - {"serial"}, 3),
        "serial_interval_ms": 200,
    }
    freeze_path = put(tmp_path / "freeze.json", freeze)
    origin = tmp_path / "origin"
    origin.mkdir()
    config, inputs, _, _ = setup_capture(origin)
    audit = audit_run(
        config, inputs, origin, pair_id="mixed-01", mode="prefix", phase="calibration"
    )
    metrics = audit["metrics"]
    measurement = read_json(config_path).get("measurement_schema_version", 1)
    if measurement == 2:
        audit["measurement_schema_version"] = 2
        metrics.update(
            all_cold_zero_hit=None,
            schedule_envelope_ns=metrics["schedule_ns"],
            schedule_timing_scope=export_m5.SCHEDULE_SCOPE,
        )
    contrasts = dict.fromkeys(export_m5.CONTRASTS, dict.fromkeys(comparable(audit), 0))
    report = {
        "status": "valid",
        "pairs": [
            {
                "pair_id": pair,
                "scenario": row["scenario"],
                "runs": {
                    mode: {
                        "seed": row["seed"],
                        "input_sha256": freeze["tables"][pair],
                        "metrics": {
                            **metrics,
                            **(
                                {"all_cold_zero_hit": True}
                                if measurement == 2 and row["scenario"] == "all-cold"
                                else {}
                            ),
                            "/private/dropped metric key": 3,
                        },
                    }
                    for mode in MODES
                },
                "contrasts": contrasts,
            }
            for pair, row in matrix.items()
        ],
        "scenarios": {
            scenario: {name: paired_summary([values]) for name, values in contrasts.items()}
            for scenario in SCENARIOS
        },
    }
    report_path = put(origin / "matrix-audit.json", report)
    index_path = put(origin / "matrix-index.json", {"private_path": "/private/control evidence"})
    request = {
        "schema_version": 2,
        "task_id": TASK,
        "baseline_commit": BASELINE,
        "analysis_commit": freeze["execution_commit"],
        "execution_commit": freeze["execution_commit"],
        "config_sha256": freeze["config_sha256"],
        "freeze_sha256": digest(freeze_path),
        "matrix_index_sha256": digest(index_path),
        "matrix_audit_sha256": digest(report_path),
        "elapsed_seconds": 5.0,
        "requested_minutes": 20,
        "releases": [
            {
                "phase": "calibration",
                "role": "C1",
                "state": "COMPLETED",
                "exit_code": "0:0",
                "elapsed_seconds": 5.0,
                "requested_minutes": 20,
                "allocation_sha256": "1" * 64,
                "terminal_sha256": "2" * 64,
                "resource_files_sha256": {"watcher/watch.jsonl": "3" * 64},
            }
        ],
    }
    request_path = put(origin / "review-request.json", request)
    review = tmp_path / "review"
    review.mkdir()
    files = (report_path, index_path, request_path)

    def rebuild(_config, _freeze, _history, analysis, output):
        assert analysis == request["analysis_commit"]
        output.mkdir()
        for path in files:
            shutil.copyfile(path, output / path.name)

    rebuild(config_path, freeze_path, origin, request["analysis_commit"], review / "rebuild")
    review = review / "rebuild"
    monkeypatch.setattr(export_m5, "collect", rebuild)

    controls = tmp_path / "controls"
    controls.mkdir()
    records = {}
    for name, reason in export_m5.CONTROL_REJECTIONS.items():
        folder = controls / name
        folder.mkdir()
        stdout, stderr = folder / "stdout.log", folder / "stderr.log"
        stdout.write_bytes(report_path.read_bytes() if reason is None else b"")
        stderr.write_text(f"{reason}\n" if reason is not None else "")
        status = "accepted" if reason is None else "rejected"
        mutation = put(folder / "fixture.json", {"name": name}) if reason is not None else None
        row = {
            "expected": status,
            "observed": status,
            "returncode": int(reason is not None),
            "stdout_sha256": digest(stdout),
            "stderr_sha256": digest(stderr),
            "fixtures": {mutation.name: digest(mutation)} if mutation else {},
        }
        records[name] = row
        put(folder / "outcome.json", row)
    controls_path = put(
        controls / "controls.json",
        {
            "schema_version": 2,
            "analysis_commit": request["analysis_commit"],
            "matrix_index_sha256": request["matrix_index_sha256"],
            "review_request_sha256": digest(request_path),
            "controls": records,
        },
    )
    decision = put(
        tmp_path / "decision.json",
        {
            "reviewer": "Synthetic independent reviewer",
            "review_request_sha256": digest(request_path),
            "controls_sha256": digest(controls_path),
            "criteria": dict.fromkeys(export_m5.CRITERIA, "passed"),
        },
    )
    args = (config_path, freeze_path, origin, review, decision, digest(decision), controls_path)
    return args, tmp_path / "public", report, matrix, measurement


def test_release_projects_only_named_fields(bundle):
    args, output, _, _, _ = bundle
    export_m5.release(*args, output)
    assert sorted(path.name for path in output.iterdir()) == ["provenance.json", "summary.json"]
    assert all("/private/" not in path.read_text() for path in output.iterdir())
    summary = read_json(output / "summary.json")
    counts = summary["results"]["pairs"][0]["modes"]["native"]["strata"]["all"]["counts"]
    assert counts["planned"] == sum(
        counts[name] for name in ("completed", "failed", "cancelled", "timeout", "unfinished")
    )
    assert read_json(output / "provenance.json")["review_sha256"] == args[-2]


def test_release_does_not_publish_a_partial_pair_of_files(bundle, monkeypatch):
    args, output, _, _, _ = bundle
    original = export_m5.write

    def fail_on_provenance(path, value):
        if path.name == "provenance.json":
            raise OSError("synthetic disk failure")
        original(path, value)

    monkeypatch.setattr(export_m5, "write", fail_on_provenance)
    with pytest.raises(OSError, match="synthetic disk failure"):
        export_m5.release(*args, output)
    assert not output.exists()
    assert not list(output.parent.glob(".m5-export-*"))


@pytest.mark.parametrize(
    ("changed", "reason"),
    (
        ("decision", "approved independent decision digest differs"),
        ("original_resources", "fresh reconstruction differs from independent review"),
        ("control_log", "control log changed"),
        ("control_fixture", "control mutation changed"),
    ),
)
def test_release_rejects_stale_or_modified_evidence(bundle, changed, reason):
    args, output, _, _, _ = bundle
    _, _, origin, _, decision, _, controls = args
    if changed == "decision":
        put(decision, {**read_json(decision), "reviewer": "Another name"})
    elif changed == "original_resources":
        path = origin / "review-request.json"
        source = read_json(path)
        source["releases"][0]["resource_files_sha256"]["watcher/watch.jsonl"] = "4" * 64
        put(path, source)
    elif changed == "control_log":
        (controls.parent / "missing_pair" / "stderr.log").write_text("unrelated crash\n")
    else:
        (controls.parent / "missing_pair" / "fixture.json").write_text("wrong mutation\n")
    with pytest.raises(ValueError, match=reason):
        export_m5.release(*args, output)
    assert not output.exists()


def test_whitelist_rejects_unregistered_freeze_and_numeric_labels(bundle):
    args, _, report, matrix, measurement = bundle
    config, freeze = args[:2]
    with pytest.raises(ValueError, match="unexpected public M5 freeze fields"):
        export_m5.public_freeze({**read_json(freeze), "node": "/private/path"}, config)
    report["pairs"][0]["contrasts"]["off-native"]["/private/numeric label"] = 42
    with pytest.raises(ValueError, match="public M5 metric keys differ"):
        export_m5.public_results(report, matrix, measurement)


@pytest.mark.parametrize(
    "name", ("clock_domain", "execution_interval", "missing_field", "failure_denominator")
)
def test_real_trace_subprocess_matches_control_reason(tmp_path, name):
    config, inputs, client, engine = setup_capture(tmp_path)
    positive = call_run(tmp_path, config, inputs)
    assert (
        export_m5.control_result("matrix_positive", positive.returncode, positive.stderr)
        == "accepted"
    )
    assert json.loads(positive.stdout)["status"] == "valid"
    if name == "clock_domain":
        engine["clock"]["boot_id"] = "different boot"
    elif name == "execution_interval":
        next(row["runner"] for row in engine["batches"] if row["runner"])["rows"][0]["end"] += 1
    elif name == "missing_field":
        del client["requests"][0]["first_token_ns"]
    else:
        client["requests"].pop()
    put(tmp_path / "engine.json", engine)
    put(tmp_path / "client.json", client)
    negative = call_run(tmp_path, config, inputs)
    assert export_m5.control_result(name, negative.returncode, negative.stderr) == "rejected"
    with pytest.raises(ValueError, match="unexpected M5 release control"):
        export_m5.control_result(name, 1, "unrelated crash\n")
    with pytest.raises(ValueError, match="unexpected M5 release control"):
        export_m5.control_result(name, 2, negative.stderr)


@pytest.mark.parametrize(
    "name", ("missing_candidate_decision", "duplicate_eviction", "missing_cold_hit_observation")
)
def test_real_v2_trace_subprocess_matches_new_control_reason(tmp_path, name):
    config, inputs, client, engine = setup_capture_v2(tmp_path)
    positive = call_run(tmp_path, config, inputs)
    assert positive.returncode == 0, positive.stderr
    if name == "missing_candidate_decision":
        engine["decisions"].pop()
    elif name == "duplicate_eviction":
        engine["evictions"].append(engine["evictions"][0])
    else:
        next(row for row in client["requests"] if row["status"] == "completed")["cached_tokens"] = (
            None
        )
    put(tmp_path / "engine.json", engine)
    put(tmp_path / "client.json", client)
    negative = call_run(tmp_path, config, inputs)
    assert export_m5.control_result(name, negative.returncode, negative.stderr) == "rejected"
