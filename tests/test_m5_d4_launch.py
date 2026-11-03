"""Offline subprocess controls for D4 scope, inputs and pressure interpretation."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_m5_workload import encode

from vllm_prefix_scheduler.m5_contract import expected_arrival_offsets
from vllm_prefix_scheduler.m5_workload import prepare_workload

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/m5-performance.json"
PLAN = ROOT / "configs/m5-d4.json"
COMMIT = "d" * 40


def read(path):
    return json.loads(path.read_text())


def put(path, value, *, compact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":") if compact else None) + "\n")
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def child(code, *args):
    bootstrap = f"import sys; sys.path[:0] = [{str(ROOT / 'scripts')!r}, {str(ROOT / 'src')!r}]\n"
    return subprocess.run(
        [sys.executable, "-B", "-c", bootstrap + code, *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )


def approval():
    return {
        "schema_version": 1,
        "task_id": "M5-paired-performance",
        "phase": "calibration",
        "execution_commit": COMMIT,
        "config_sha256": sha(CONFIG),
        "user_approval_ref": "local synthetic D4 authority fixture",
        "roles": ["D4"],
        "max_attempts": 11,
        "max_requested_minutes": 220,
        "minutes_per_attempt": 20,
        "gpus_per_attempt": 1,
        "freeze_sha256": None,
        "d4_plan_sha256": sha(PLAN),
    }


def authorize(path, config=CONFIG):
    return child(
        """import json
from pathlib import Path
from vllm_prefix_scheduler.m5_campaign import check_authorization, read_json
record, config, commit = sys.argv[1:]
try:
    print(json.dumps(check_authorization(read_json(record), Path(config), commit, 'calibration')))
except (KeyError, OSError, TypeError, ValueError) as exc:
    sys.exit(str(exc))
""",
        path,
        config,
        COMMIT,
    )


def test_d4_approval_child_excludes_extra_attempt_and_roles(tmp_path):
    path = tmp_path / "authorization.json"
    allowed = authorize(put(path, approval()))
    assert allowed.returncode == 0, allowed.stderr
    assert json.loads(allowed.stdout) == ["D4"]
    for changed in (
        {"max_attempts": 12, "max_requested_minutes": 240},
        {"roles": ["D4", "C1"]},
        {"d4_plan_sha256": "f" * 64},
    ):
        rejected = authorize(put(path, {**approval(), **changed}))
        assert rejected.returncode != 0 and "D4" in rejected.stderr


def test_d4_cannot_be_authorized_as_qualification_predecessor(tmp_path):
    config = read(CONFIG)
    config["calibration"]["qualification_predecessor_roles"] = ["D4"]
    config_path = put(tmp_path / CONFIG.name, config)
    auth = approval()
    del auth["d4_plan_sha256"]
    auth.update(
        config_sha256=sha(config_path),
        max_attempts=18,
        max_requested_minutes=360,
        roles=["C1", "C2", "C3-burst", "C3-mixed", "C4", "C5", "C6"],
        qualification_batch={
            "schema_version": 2,
            "prior_allocation_count": 11,
            "prior_history_sha256": "a" * 64,
            "new_max_attempts": 7,
            "predecessor": {
                "protocol": "same-condition-ee-v1",
                "role": "D4",
                "attempt": "d4-01",
                "execution_commit": COMMIT,
                "proposal_file": "m5-d4.json",
                "proposal_sha256": sha(PLAN),
                "review_directory": "d4-review",
                "analysis_sha256": "b" * 64,
            },
        },
    )
    rejected = authorize(put(tmp_path / "qualification.json", auth), config_path)
    assert rejected.returncode != 0
    assert "approved same-condition predecessor" in rejected.stderr


def admission_fixture():
    rows = {}
    for index, (queued, scheduled, native_last) in enumerate(
        ((0, 0, 10), (0, 0, 12), (1, 10, 15), (2, 13, 16))
    ):
        rows[str(index)] = {
            "planned_ns": index,
            "submit_ns": queued,
            "first_components_ns": {
                "submit_to_native_queued": 0,
                "native_queued_to_first_scheduled": scheduled - queued,
                "native_first_scheduled_to_first_token": 0,
            },
            "end_components_ns": {"native_first_to_last_token": native_last - scheduled},
        }
    return {"status": "complete", "clock": {"verified": True}, "planned_count": 4, "per_id": rows}


def test_d4_admission_child_integrates_half_open_capacity_proxy(tmp_path):
    path = put(tmp_path / "native-timing.json", admission_fixture())
    result = child(
        """import json
from prepare_m5_d4 import admission_summary
with open(sys.argv[1]) as source:
    print(json.dumps(admission_summary(json.load(source), 2, 2)))
""",
        path,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "tail_wait_median_ns": 10,
        "tail_native_output_envelope_median_ns": 4,
        "peak_waiting": 2,
        "peak_native_output_envelopes": 2,
        "wait_total_ns": 20,
        "wait_overlap_with_capacity_ns": 19,
        "wait_overlap_with_capacity_fraction": 19 / 20,
    }


@pytest.fixture
def prepared_sources(tmp_path):
    # Fixture hashes replace the three historical pins in the isolated child only.
    config = read(CONFIG)
    plan = read(PLAN)
    target = plan["target"]
    original = prepare_workload(config, encode, "mixed", target["seed"], rate=8, count=60)
    archive = tmp_path / "history/attempts/c3-01/node"
    original_path = put(archive / "mixed-2-endpoint/input.json", original, compact=True)
    c3_path = put(archive / "calibration-result.json", {"status": "blocked"})
    analysis_path = put(
        tmp_path / "d3-analysis.json",
        {
            "role": "D3",
            "status": "diagnostic-complete",
            "decision": {
                "thermal_control": "verified",
                "endpoint_stability_under_control": "unstable",
            },
            "proposal_sha256": plan["D3_plan_sha256"],
            "runs": {
                leaf: {"native_timing": admission_fixture()} for leaf in ("E1", "E2", "E3", "E4")
            },
        },
    )
    source = read(analysis_path)
    for run in source["runs"].values():
        timing = run["native_timing"]
        timing["per_id"] = {
            str(i): {**timing["per_id"][str(i % 4)], "planned_ns": i} for i in range(60)
        }
        timing["planned_count"] = 60
    put(analysis_path, source)
    plan["retained_C3_sha256"], plan["D3_analysis_sha256"] = sha(c3_path), sha(analysis_path)
    offsets = expected_arrival_offsets(config, "mixed", target["seed"], 60, 6, None)
    table6 = {
        **original,
        "rate_rps": 6.0,
        "requests": [
            {**row, "offset_ns": offset}
            for row, offset in zip(original["requests"], offsets, strict=True)
        ],
    }
    derived = put(tmp_path / "expected-6.json", table6, compact=True)
    target["arrival_tables"]["6"]["sha256"] = sha(derived)
    target["arrival_tables"]["8"]["sha256"] = sha(original_path)
    return plan, original_path, c3_path, analysis_path, derived


def prepare_child(tmp_path, sources, mutation=None):
    plan, original, c3, analysis, _ = sources
    if mutation:
        mutation(plan)
    plan_path = put(tmp_path / "plan.json", plan)
    output = tmp_path / "output"
    result = child(
        """import prepare_m5_d4 as prepare
fixture_args, cli_args = sys.argv[1:4], sys.argv[4:]
prepare.D1_INPUT_SHA, prepare.C3_BLOCKED_SHA, prepare.D3_ANALYSIS_SHA256 = fixture_args
sys.argv[1:] = cli_args
prepare.main()
""",
        sha(original),
        sha(c3),
        sha(analysis),
        "--plan",
        plan_path,
        "--config",
        CONFIG,
        "--history",
        tmp_path / "history",
        "--d3-analysis",
        analysis,
        "--output",
        output,
    )
    return result, output


def test_d4_prepare_child_writes_exact_two_rate_tables(tmp_path, prepared_sources):
    result, output = prepare_child(tmp_path, prepared_sources)
    assert result.returncode == 0, result.stderr
    assert (output / "mixed-8rps.json").read_bytes() == prepared_sources[1].read_bytes()
    assert (output / "mixed-6rps.json").read_bytes() == prepared_sources[-1].read_bytes()
    assert read(output / "manifest.json")["budget"]["work_estimate_s"] == 1040


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        (lambda p: p["target"].update(measure_order_rps=[6, 8, 8, 6]), "rate contrast"),
        (lambda p: p["target"].update(preheat_rate_rps=6), "rate contrast"),
        (lambda p: p["target"]["arrival_tables"]["6"].update(sha256="f" * 64), "derived input"),
        (lambda p: p["assessment"].update(can_qualify_C3=True), "qualification scope"),
        (lambda p: p["budget"].update(retries=1), "budget differs"),
    ),
)
def test_d4_prepare_child_refuses_semantic_tampering(tmp_path, prepared_sources, mutation, reason):
    result, output = prepare_child(tmp_path, prepared_sources, mutation)
    assert result.returncode == 1 and reason in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("mutation", "signal"),
    (
        ("none", "supported"),
        ("equal", "not_supported"),
        ("thermal", "unverified"),
        ("unreleased", "unverified"),
        ("missing", "unverified"),
    ),
)
def test_d4_pressure_child_requires_all_four_rounds_and_all_six_pairs(tmp_path, mutation, signal):
    runs = {}
    for index, rate in enumerate((8, 6, 6, 8), 1):
        timing = admission_fixture()
        timing.update(first_decomposed_count=4, terminal_decomposed_count=4)
        if rate == 6 and not (index == 3 and mutation == "equal"):
            for row in timing["per_id"].values():
                row["first_components_ns"]["native_queued_to_first_scheduled"] = 0
        runs[f"E{index}"] = {
            "status": "admitted",
            "rate_rps": rate,
            "prelude": {"verified": True},
            "pre_t0_verified": True,
            "endpoint_evidence_complete": not (index == 3 and mutation == "missing"),
            "native_timing": timing,
            "in_window": {"identity": f"E{index}"},
        }
    result = child(
        """import json
from m5_d4_audit import assess_d4
with open(sys.argv[1]) as source:
    runs = json.load(source)
mutation = sys.argv[2]
def thermal(before, after, limits):
    # Cross-rate-only failure, while both same-rate comparisons remain comparable.
    return {'comparable': mutation != 'thermal' or (before['identity'], after['identity']) != ('E1', 'E2')}
print(json.dumps(assess_d4(runs, {'measure_order_rps': [8, 6, 6, 8], 'count_per_engine': 4,
                                 'max_num_seqs': 2}, 2, {}, thermal, released=mutation != 'unreleased')))
""",
        put(tmp_path / "runs.json", runs),
        mutation,
    )
    assert result.returncode == 0, result.stderr
    assessment = json.loads(result.stdout)
    assert assessment["pressure_signal"] == signal
    assert len(assessment["all_window_thermal_pairs"]) == 6
    assert assessment["qualification_started"] is False
