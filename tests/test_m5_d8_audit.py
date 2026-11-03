"""D8 report assembly preserves the original qualification evidence boundary."""

import importlib
import json
from pathlib import Path

import pytest

from vllm_prefix_scheduler.m5_campaign import digest, read_json

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = ROOT / "configs/m5-d8.json"
CONFIG = ROOT / "configs/m5-performance.json"
T = 1_000_000_000


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def d8_fixture(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    checker = importlib.import_module("check_m5_d2")
    verifier = importlib.import_module("m5_c3_bundle")
    plan = read_json(PROPOSAL)
    proposal_sha = digest(PROPOSAL)
    commit = "b" * 40
    attempt = tmp_path / "attempts/d8-01"
    remote = "/synthetic/attempts/d8-01"
    node = attempt / "node"
    put(attempt / "submission.json", {"stdout": "116"})
    put(attempt / "authorization.json", {"roles": ["D8"], "d8_plan_sha256": proposal_sha})
    put(
        attempt / "planned.json",
        {
            "config_sha256": digest(CONFIG),
            "auxiliary_sha256": proposal_sha,
            "execution_commit": commit,
            "remote_attempt": remote,
        },
    )
    put(
        attempt.parent / "d7-01/planned.json",
        {
            "execution_commit": plan["retained_D7"]["execution_commit"],
            "auxiliary_sha256": plan["retained_D7"]["proposal_sha256"],
        },
    )
    old_c3 = put(attempt.parent / "c3-01/node/calibration-result.json", {"status": "blocked"})
    history = [
        {
            "phase": "calibration",
            "role": f"old-{number}",
            "job_id": str(100 + number),
            "requested_minutes": 20,
            "max_attempts": number,
            "complete": True,
        }
        for number in range(1, 17)
    ]
    history[14]["role"] = "D7"
    history[15]["role"] = "D8"
    put(attempt.parent / "d7-01/submission.json", {"stdout": history[14]["job_id"]})
    predecessor = {**plan["retained_D7"], "resource_release": history[14]}
    monkeypatch.setattr(checker, "audit_history", lambda _: history)
    monkeypatch.setattr(checker, "d8_predecessor", lambda *_: predecessor)
    monkeypatch.setattr(checker, "verify_analysis", lambda *_: None)
    monkeypatch.setattr(checker, "git_identity", lambda *_: {})
    monkeypatch.setattr(checker, "sealed_leaf", lambda *_, **__: {"synthetic": True})
    monkeypatch.setattr(checker, "device", lambda _: ("GPU-fixture",))
    monkeypatch.setattr(
        checker, "audit_run", lambda _, __, raw, **___: read_json(raw.parent / "audit.json")
    )

    def fixture_digest(path):
        if path == old_c3:
            return checker.C3_BLOCKED_SHA
        if path.name == "input.json":
            return checker.D1_INPUT_SHA
        return digest(path)

    monkeypatch.setattr(checker, "digest", fixture_digest)

    target = {"temperature_c": 62, "sm_clock_mhz": 1800, "memory_clock_mhz": 9501}
    plateau = {**target, "gpu_uuid": "GPU-fixture"}
    monkeypatch.setattr(
        checker,
        "prelude_summary",
        lambda *_, **__: {
            "verified": True,
            "issues": [],
            "selected_plateau": plateau,
        },
    )
    samples, rows = [], []
    for index, identity in enumerate(checker.RUNS):
        pilot = identity == checker.RUNS[0]
        leaf = node / identity
        raw = leaf / "raw"
        base = (1 + 100 * index) * T
        t0 = base + 60 * T
        pgid = 200 + index

        def point(stamp, group=pgid):
            return {
                "monotonic_ns": stamp,
                "temperature_c": target["temperature_c"],
                "sm_clock_mhz": target["sm_clock_mhz"],
                "memory_clock_mhz": target["memory_clock_mhz"],
                "gpu_utilization_percent": 80,
                "gpu_apps": [{"pgid": group}],
            }

        samples.append(point(base + 20 * T))
        before = point(t0 - T // 2)
        if not pilot:
            samples.extend([before, *(point(t0 + i * T) for i in range(8))])
        thermal = {
            "schema_version": 2,
            "plan_sha256": proposal_sha,
            "role": "pilot" if pilot else "measure",
            "status": "target-frozen" if pilot else "admitted",
            "conditioning_start_ns": base + 10 * T,
            "conditioning_end_ns": base + 50 * T,
            "target": target,
            "target_gpu_uuid": "GPU-fixture",
            "pre_t0": None
            if pilot
            else {
                "decision_ns": t0 - T // 10,
                "measure_start_ns": t0,
                "sample": before,
            },
        }
        put(raw / "d2-thermal.json", thermal)
        put(leaf / "input.json", {})
        phases = (
            {}
            if pilot
            else {"warmup_end_ns": t0 - T // 5, "measure_start_ns": t0, "drain_end_ns": t0 + 7 * T}
        )
        client_clock = {
            "pid": 300 + index,
            "host": "fixture",
            "boot_id": "b",
            "time_namespace": "tn",
        }
        put(
            raw / "client.json",
            {
                "tier": "endpoint",
                "clock": client_clock,
                "phases": phases,
                "requests": []
                if pilot
                else [
                    {
                        "request_id": f"request-{i:06d}",
                        "first_token_ns": t0 + i + 100_000_000,
                        "submit_ns": t0 + i,
                        "cached_tokens": 0,
                        "token_ids": [7],
                    }
                    for i in range(60)
                ],
            },
        )
        put(
            raw / "engine.json",
            {
                "tier": "endpoint",
                "clock": {**client_clock, "pid": 400 + index},
                "profile": {"max_num_active_seqs": None, "num_gpu_blocks": 5553},
            },
        )
        counts = {key: 60 for key in ("planned", "submitted", "completed", "timely_first_token")}
        audit = (
            thermal
            if pilot
            else {
                "metrics": {
                    "strata": {"all": {"counts": counts, "ttft_ns": {"p95": 100_000_000}}},
                    "throughput": {
                        "last_terminal_requests_per_s": 8.0,
                        "last_terminal_tokens_per_s": 128.0,
                    },
                },
            }
        )
        put(leaf / "audit.json", audit)
        run_remote = f"{remote}/{identity}/raw"
        artifact_hash = digest(raw / "d2-thermal.json")
        entries = [
            {"path": f"configs/{name}", "source": f"/stage/{name}", "sha256": value}
            for name, value in {
                checker.D8_PROPOSAL_NAME: proposal_sha,
                checker.D2R110_PROPOSAL_NAME: plan["D2R110_plan_sha256"],
                checker.D8_PREVIOUS_ANALYSIS: plan["retained_D7"]["analysis_sha256"],
                "original-c3-result.json": checker.C3_BLOCKED_SHA,
            }.items()
        ]
        if not pilot:
            entries.append(
                {
                    "path": "configs/pilot.json",
                    "source": f"{remote}/thermal-pilot/raw/d2-thermal.json",
                    "sha256": digest(node / "thermal-pilot/raw/d2-thermal.json"),
                }
            )
        entries.append(
            {
                "path": "artifacts/d2-thermal.json",
                "source": f"{run_remote}/d2-thermal.json",
                "sha256": artifact_hash,
            }
        )
        put(leaf / "collector/manifest.json", {"files": entries})
        row = {
            "id": identity,
            "pair_id": "D8",
            "mode": "guarded",
            "tier": "endpoint",
            "seed": 51213,
            "scenario": "mixed",
            "count": 60,
            "rate_rps": 8,
            "returncode": 0,
            "run": run_remote,
            "audit_sha256": digest(leaf / "audit.json"),
            "d2_thermal_sha256": artifact_hash,
            "d2_thermal_status": thermal["status"],
            "started_monotonic_ns": base,
            "ended_monotonic_ns": base + 90 * T,
            "engine_pgid": pgid,
        }
        if not pilot:
            row["d8_endpoint_complete"] = True
        rows.append(row)
    put(node / "index.json", {"schema_version": 1, "stage": "D8", "runs": rows})
    journal = node / "d2-device.jsonl"
    journal.write_text("".join(json.dumps(sample) + "\n" for sample in samples))
    (node / "gpu-during.jsonl").write_text(
        "".join(json.dumps({"thermal": sample}) + "\n" for sample in samples)
    )
    monkeypatch.setattr(checker, "device_samples", lambda *_: samples)
    put(node / "allocation.json", {"d2_device_sha256": digest(journal), "gpu_before": {}})
    put(
        node / "d8-result.json",
        {
            "status": "diagnostic-complete",
            "role": "D8",
            "proposal_sha256": proposal_sha,
            "inherited_d2r110_proposal_sha256": plan["D2R110_plan_sha256"],
            "retained_d7_analysis_sha256": plan["retained_D7"]["analysis_sha256"],
            "input_sha256": checker.D1_INPUT_SHA,
            "audited_runs": 4,
            "retained_c3_sha256": checker.C3_BLOCKED_SHA,
        },
    )
    return checker, verifier, plan, attempt, predecessor, history, commit


def test_d8_assembled_report_replays_through_unchanged_same_condition_verifier(d8_fixture):
    checker, verifier, plan, attempt, _, history, commit = d8_fixture
    report = checker.review(attempt, PROPOSAL)
    verifier.verify_same_condition_report(
        report,
        plan,
        {"role": "D8", "execution_commit": commit, "proposal_sha256": digest(PROPOSAL)},
        history[-1],
    )
    assert report["retained_D7"] == plan["retained_D7"]
    assert report["decision"]["qualification_started"] is False
    assert not any("native_timing" in row for row in report["runs"].values())


def test_d8_too_slow_is_retained_and_fails_original_pair_gate(d8_fixture):
    checker, verifier, plan, attempt, _, history, commit = d8_fixture
    audit = attempt / "node/E2/audit.json"
    raw = read_json(audit)
    raw["metrics"]["strata"]["all"]["ttft_ns"]["p95"] = 106_000_000
    put(audit, raw)
    report = checker.review(attempt, PROPOSAL)
    assert report["status"] == "diagnostic-complete"
    assert report["decision"]["endpoint_stability_under_control"] == "unstable"
    with pytest.raises(ValueError, match="original thermal or paired performance gate"):
        verifier.verify_same_condition_report(
            report,
            plan,
            {"role": "D8", "execution_commit": commit, "proposal_sha256": digest(PROPOSAL)},
            history[-1],
        )


@pytest.mark.parametrize("leaf", ("E2", "E4"))
def test_d8_missing_endpoint_count_cannot_pass_or_advance(d8_fixture, leaf):
    checker, _, _, attempt, _, _, _ = d8_fixture
    node = attempt / "node"
    audit = node / leaf / "audit.json"
    raw = read_json(audit)
    raw["metrics"]["strata"]["all"]["counts"]["timely_first_token"] = 59
    put(audit, raw)
    with pytest.raises(ValueError, match="indexed endpoint completeness differs"):
        checker.review(attempt, PROPOSAL)
    index = read_json(node / "index.json")
    next(row for row in index["runs"] if row["id"] == leaf)["d8_endpoint_complete"] = False
    put(node / "index.json", index)
    if leaf == "E2":
        with pytest.raises(ValueError, match="continued after independently incomplete endpoint"):
            checker.review(attempt, PROPOSAL)
    else:
        with pytest.raises(ValueError, match="node result differs"):
            checker.review(attempt, PROPOSAL)
        result = read_json(node / "d8-result.json")
        result["status"] = "diagnostic-incomplete"
        put(node / "d8-result.json", result)
        report = checker.review(attempt, PROPOSAL)
        assert report["status"] == "diagnostic-incomplete"
        assert report["runs"]["E4"]["counts"]["timely_first_token"] == 59
        assert report["runs"]["E4"]["endpoint_evidence_complete"] is False
        assert report["decision"]["endpoint_stability_under_control"] == "unverified"


def test_d8_predecessor_and_collector_lineage_are_independent_gates(d8_fixture):
    checker, _, _, attempt, predecessor, _, _ = d8_fixture
    predecessor["resource_release"] = {"job_id": "different"}
    with pytest.raises(ValueError, match="retained D7 diagnostic or release differs"):
        checker.review(attempt, PROPOSAL)
    predecessor["resource_release"] = d8_fixture[-2][14]
    manifest = attempt / "node/E2/collector/manifest.json"
    raw = read_json(manifest)
    raw["files"] = [
        row for row in raw["files"] if not row["source"].endswith("/previous-d7-analysis.json")
    ]
    put(manifest, raw)
    with pytest.raises(ValueError, match="previous-d7-analysis.json is not in the collector"):
        checker.review(attempt, PROPOSAL)


def test_d8_does_not_admit_native_timing_or_capacity_override(d8_fixture):
    checker, _, _, attempt, _, _, _ = d8_fixture
    engine = attempt / "node/E1/raw/engine.json"
    raw = read_json(engine)
    raw["profile"]["max_num_active_seqs"] = 4
    put(engine, raw)
    with pytest.raises(ValueError, match="endpoint profile or collector tier differs"):
        checker.review(attempt, PROPOSAL)
