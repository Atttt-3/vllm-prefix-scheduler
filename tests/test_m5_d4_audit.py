"""D4 assessment controls use complete synthetic native intervals and evidence gates."""

import importlib
import json
from pathlib import Path

import pytest

from vllm_prefix_scheduler.m5_campaign import digest

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT / "configs/m5-d4.json").read_text())
TARGET = PLAN["target"]


@pytest.fixture
def d4(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("m5_d4_audit")


def timing(wait, *, spacing=100):
    count = TARGET["count_per_engine"]
    return {
        "status": "complete",
        "clock": {"verified": True},
        "planned_count": count,
        "first_decomposed_count": count,
        "terminal_decomposed_count": count,
        "per_id": {
            f"request-{i:06}": {
                "planned_ns": i * spacing,
                "submit_ns": i * spacing,
                "first_components_ns": {
                    "submit_to_native_queued": 0,
                    "native_queued_to_first_scheduled": wait,
                    "native_first_scheduled_to_first_token": 100,
                },
                "end_components_ns": {"native_first_to_last_token": 300},
            }
            for i in range(count)
        },
    }


def runs():
    return {
        f"E{i}": {
            "rate_rps": rate,
            "status": "admitted",
            "pre_t0_verified": True,
            "prelude": {"verified": True},
            "in_window": 65,
            "endpoint_evidence_complete": True,
            "native_timing": timing(120 if rate == 8 else 20),
        }
        for i, rate in enumerate(TARGET["measure_order_rps"], 1)
    }


def compare(a, b, limits):
    return {"comparable": abs(a - b) <= 2}


def classify(d4, data, *, released=True):
    return d4.assess_d4(data, TARGET, 15, {}, compare, released=released)


def test_d4_pressure_requires_both_lower_rate_rounds_and_reports_all_six_pairs(d4):
    result = classify(d4, runs())
    assert result["pressure_signal"] == "supported"
    assert result["qualification_started"] is False
    assert result["request_evidence_complete"] is True
    assert result["mutually_thermally_comparable"] is True
    assert [(row["reference"], row["observed"]) for row in result["all_window_thermal_pairs"]] == [
        ("E1", "E2"),
        ("E1", "E3"),
        ("E1", "E4"),
        ("E2", "E3"),
        ("E2", "E4"),
        ("E3", "E4"),
    ]
    assert d4.SAME_RATE_PAIRS == tuple(tuple(pair) for pair in TARGET["same_rate_pairs"])
    high = result["runs"]["E1"]["admission"]
    assert high["tail_wait_median_ns"] == 120
    assert high["peak_waiting"] == 2
    assert high["peak_native_output_envelopes"] == 4
    assert high["wait_total_ns"] == 120 * 60
    assert 0 < high["wait_overlap_with_capacity_fraction"] < 1


@pytest.mark.parametrize(
    "case",
    ["equal_wait", "larger_peak", "cross_rate_heat", "bad_clock", "incomplete", "unreleased"],
)
def test_d4_fails_pressure_classification_at_the_declared_gate(d4, case):
    data = runs()
    wanted = "unverified"
    if case == "equal_wait":
        data["E3"]["native_timing"] = timing(120)
        wanted = "not_supported"
    elif case == "larger_peak":
        data["E3"]["native_timing"] = timing(20, spacing=5)
        wanted = "not_supported"
    elif case == "cross_rate_heat":
        data["E2"]["in_window"] = data["E3"]["in_window"] = 62
    elif case == "bad_clock":
        data["E3"]["native_timing"]["clock"]["verified"] = False
    elif case == "incomplete":
        del data["E4"]
    result = classify(d4, data, released=case != "unreleased")
    assert result["pressure_signal"] == wanted
    if case == "cross_rate_heat":
        assert sum(row["thermal"]["comparable"] for row in result["all_window_thermal_pairs"]) == 2
    if case == "incomplete":
        assert result["runs"]["E4"]["status"] == "unattempted"
        assert result["runs"]["E4"]["admission"] is None


def test_d4_rejects_wrong_rate_even_in_an_attempted_incomplete_leaf(d4):
    data = runs()
    data["E2"] = {"status": "incomplete", "rate_rps": 8}
    with pytest.raises(ValueError, match="ABBA order"):
        classify(d4, data)


def test_d4_thermal_collector_requires_frozen_eight_rps_prelude(d4, tmp_path):
    checker = importlib.import_module("check_m5_d2")
    raw = tmp_path / "raw"
    raw.mkdir()
    thermal = raw / "d2-thermal.json"
    thermal.write_text('{"role":"measure"}')
    collector = tmp_path / "collector"
    collector.mkdir()
    proposal = "b" * 64
    preheat = TARGET["arrival_tables"]["8"]["sha256"]
    run = "/owned/node/E2/raw"
    files = [
        {
            "path": "configs/000-proposed-d4.json",
            "source": "/staged/proposed-d4.json",
            "sha256": proposal,
        },
        {
            "path": "configs/001-mixed-8rps.json",
            "source": "/staged/mixed-8rps.json",
            "sha256": preheat,
        },
        {
            "path": "artifacts/000-d2-thermal.json",
            "source": f"{run}/d2-thermal.json",
            "sha256": digest(thermal),
        },
    ]
    manifest = collector / "manifest.json"
    manifest.write_text(json.dumps({"files": files}))
    row = {"run": run, "d2_thermal_sha256": digest(thermal)}
    bindings = {"mixed-8rps.json": preheat}
    assert checker.bind_thermal_artifact(
        tmp_path, row, proposal, role="D4", d4_configs=bindings
    ) == {"role": "measure"}
    files[1]["sha256"] = TARGET["arrival_tables"]["6"]["sha256"]
    manifest.write_text(json.dumps({"files": files}))
    with pytest.raises(ValueError, match="mixed-8rps.json is not in the collector"):
        checker.bind_thermal_artifact(tmp_path, row, proposal, role="D4", d4_configs=bindings)
