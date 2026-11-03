"""Check D5-R's earliest eligible pilot and unchanged measured conditioning."""

import asyncio
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SECOND = 1_000_000_000
STARTED = SECOND
BAND = [64, 66]


def samples(temperatures):
    return [
        {
            "monotonic_ns": STARTED + i * SECOND,
            "gpu_uuid": "gpu",
            "temperature_c": temperature,
            "sm_clock_mhz": 1875,
            "memory_clock_mhz": 9501,
            "gpu_utilization_percent": 100,
        }
        for i, temperature in enumerate(temperatures)
    ]


@pytest.fixture
def thermal(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("m5_d2_thermal")


@pytest.fixture
def conditioning():
    return json.loads((ROOT / "configs/m5-d3.json").read_text())["conditioning"]


@pytest.mark.parametrize(("early", "later"), ((62, 64), (68, 66)))
def test_d5r_pilot_first_eligible_at_inclusive_band_edges(thermal, conditioning, early, later):
    observed = samples([early] * 35 + [later] * 30)
    original = thermal.first_plateau(observed, STARTED, conditioning)
    restricted = thermal.first_plateau(
        observed, STARTED, conditioning, pilot_temperature_range=BAND
    )
    assert original["end_ns"] == STARTED + 30 * SECOND
    assert original["temperature_c"] == early
    assert restricted["end_ns"] == STARTED + 45 * SECOND
    assert restricted["temperature_c"] == later
    assert (
        thermal.first_plateau(observed[:45], STARTED, conditioning, pilot_temperature_range=BAND)
        is None
    )


@pytest.mark.parametrize("case", ("visible_second_wave", "after_cutoff", "measured_unchanged"))
def test_d5r_producer_requires_eligible_visible_pilot_wave(
    thermal, conditioning, monkeypatch, case
):
    replay = importlib.import_module("replay_m5")
    measured = case == "measured_unchanged"
    clock = SimpleNamespace(now=STARTED)
    monkeypatch.setattr(replay, "time", SimpleNamespace(monotonic_ns=lambda: clock.now))
    cutoff = STARTED + (110 if measured else 150) * SECOND
    wave_ends = iter(
        (STARTED + 31 * SECOND,)
        if measured
        else (cutoff - 1,)
        if case == "after_cutoff"
        else (STARTED + 31 * SECOND, STARTED + 51 * SECOND)
    )
    observed = samples([67] * 65 if measured else [62] * 35 + [64] * 30)
    visible = []

    async def wave(llm, plans, rows, params, cutoff_ns, deadline_ns):
        assert cutoff_ns == cutoff
        rows[0].update(status="completed", token_ids=[1])
        clock.now = next(wave_ends)

    async def drain(label):
        return {"at_ns": clock.now}

    async def reset_prefix_cache(**kwargs):
        return True

    def read_samples(path, through):
        visible.append(through)
        if case == "after_cutoff":
            clock.now = cutoff + 1
        return [row for row in observed if row["monotonic_ns"] <= through]

    monkeypatch.setattr(replay, "_d2_replay_wave", wave)
    monkeypatch.setattr(thermal, "read_samples", read_samples)
    target = dict(temperature_c=65, sm_clock_mhz=1875, memory_clock_mhz=9501) if measured else None
    journal = {
        "schema_version": 2,
        "status": "invalid",
        "waves": [],
        "target": target,
        "target_gpu_uuid": "gpu" if measured else None,
        "plateau": None,
    }
    asyncio.run(
        replay._d2_condition(
            SimpleNamespace(d2_role="measure" if measured else "pilot", d2_samples=None),
            {"requests": [{"request_id": "r0", "offset_ns": 0}], "sampling": {"max_tokens": 1}},
            SimpleNamespace(reset_prefix_cache=reset_prefix_cache),
            None,
            drain,
            journal,
            conditioning,
            30 * SECOND,
            pilot_temperature_range=BAND,
        )
    )
    assert journal["conditioning_cutoff_ns"] == cutoff
    if measured:
        assert journal["plateau"]["temperature_c"] == 67
        assert journal["target"] == target
    elif case == "visible_second_wave":
        assert visible == [STARTED + 31 * SECOND, STARTED + 51 * SECOND]
        assert journal["target"]["temperature_c"] == 64
        assert journal["plateau"]["end_ns"] == STARTED + 45 * SECOND
        assert journal["status"] == "target-frozen"
    else:
        assert journal["status"] == "invalid"
        assert journal["plateau"] is None
        assert journal["waves"][0]["plateau_decision_ns"] == cutoff + 1
