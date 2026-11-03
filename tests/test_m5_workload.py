import asyncio
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_prefix_scheduler.m5_contract import workload_gate
from vllm_prefix_scheduler.m5_workload import prepare_workload

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/m5-performance.json").read_text())


def encode(text):
    """Produce predictable variable first blocks without importing any model runtime."""
    tokens = []
    for word in text.split():
        if word.rstrip(".") and all(c in "0123456789abcdef" for c in word.rstrip(".")):
            tokens.extend(ord(c) for c in word.rstrip("."))
        else:
            tokens.append(sum(map(ord, word)))
    return tokens


def test_burst_fixed_gaps_and_repeatable_token_pair(config):
    first = prepare_workload(config, encode, "burst", 1234, rate=5)
    assert first == prepare_workload(config, encode, "burst", 1234, rate=5)
    assert first != prepare_workload(config, encode, "burst", 1235, rate=5)
    requests = first["requests"]
    assert len(requests) == config["request_count"]
    assert requests[0]["offset_ns"] == 0
    assert requests[1]["offset_ns"] == 2_000_000
    assert requests[20]["offset_ns"] == 4_000_000_000
    assert sum(r["class"] == "cold" for r in requests) == 250
    warmup = first["warmup"]
    rows = [r for r in requests + warmup if r["family"] == requests[10]["family"]]
    assert len(rows) > 3
    assert len({tuple(r["prompt_token_ids"][:512]) for r in rows}) == 1
    assert len({tuple(r["prompt_token_ids"]) for r in rows}) == len(rows)
    assert len({r["cache_salt"] for r in rows}) == 1


def test_mixed_seeded_jitter_keeps_open_loop_duration(config):
    workload = prepare_workload(config, encode, "mixed", 51001, rate=5, count=60)
    rng = random.Random(51001 ^ 0x4D350001)
    requests = workload["requests"]
    expected = [
        0,
        *[i * 200_000_000 + rng.randint(-25_000_000, 25_000_000) for i in range(1, 59)],
        59 * 200_000_000,
    ]
    assert [r["offset_ns"] for r in requests] == expected
    assert len({tuple(r["prompt_token_ids"][:16]) for r in requests if r["class"] == "cold"}) == 30


def test_negative_controls_and_exact_mechanism_inputs(config):
    serial = prepare_workload(config, encode, "serial", 19, interval_ms="120.25", count=20)
    assert [r["offset_ns"] for r in serial["requests"]] == [i * 120_250_000 for i in range(20)]
    cold = prepare_workload(config, encode, "all-cold", 19, rate="2.5", count=20)
    assert not cold["warmup"]
    assert {r["class"] for r in cold["requests"]} == {"cold"}
    pressure = prepare_workload(config, encode, "mechanism", 19)
    ample = prepare_workload(config, encode, "mechanism", 19, mechanism="ample")
    assert {k: v for k, v in pressure.items() if k != "mechanism"} == {
        k: v for k, v in ample.items() if k != "mechanism"
    }
    a, *fillers = pressure["warmup"]
    b, c = pressure["requests"]
    assert [a["request_id"], b["request_id"], c["request_id"]] == ["prime-A", "B", "C"]
    assert len(a["prompt_token_ids"]) == len(c["prompt_token_ids"]) == 1025
    assert a["prompt_token_ids"][:1024] == c["prompt_token_ids"][:1024]
    assert a["prompt_token_ids"][-1] != c["prompt_token_ids"][-1]
    assert len(b["prompt_token_ids"]) == 1537
    assert sum(len(row["prompt_token_ids"]) // 16 for row in fillers) == 48


@pytest.mark.parametrize(
    ("scenario", "arguments"),
    [
        ("mechanism", {"mechanism": "pressure"}),
        ("mechanism", {"mechanism": "ample"}),
        ("burst", {"rate": 3}),
        ("mixed", {"rate": 8}),
        ("all-cold", {"rate": 8}),
        ("ample", {"rate": 8}),
        ("serial", {"interval_ms": 900}),
    ],
)
def test_v2_audits_generated_workload_geometry(config, scenario, arguments):
    config["measurement_schema_version"] = 2
    workload = prepare_workload(config, encode, scenario, 19, **arguments)
    phase = "diagnostic" if scenario == "mechanism" else "formal"
    audited = workload_gate(workload, config, None, phase)
    assert len(audited) == len(workload["requests"])


def test_invalid_arrival_spec(config):
    with pytest.raises(ValueError, match="positive fixed interval"):
        prepare_workload(config, encode, "serial", 1, rate=5)
    with pytest.raises(ValueError, match="positive absolute rate"):
        prepare_workload(config, encode, "mixed", 1, rate=0)
    with pytest.raises(ValueError, match="must fit"):
        prepare_workload(config, encode, "burst", 1, rate=10_000)


def test_real_async_client_loop_keeps_failures_and_first_token(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    # The CLI lazily imports vLLM only when inside an approved GPU allocation.
    import replay_m5

    entered = []
    released = None

    class FakeEngine:
        async def generate(self, prompt, params, *, request_id):
            entered.append(request_id)
            if request_id == "early":
                await released.wait()
                raise ValueError("synthetic validation failure")
            released.set()
            yield SimpleNamespace(outputs=[], finished=False, num_cached_tokens=0)
            yield SimpleNamespace(
                outputs=[SimpleNamespace(token_ids=[10, 11])],
                finished=True,
                num_cached_tokens=0,
            )

    async def drive():
        nonlocal released
        released = asyncio.Event()
        plans = [
            {"request_id": rid, "prompt_token_ids": [1], "cache_salt": rid}
            for rid in ("early", "later")
        ]
        start = replay_m5.time.monotonic_ns()
        rows = [replay_m5._entry(plan, start + i * 100_000) for i, plan in enumerate(plans)]
        tasks = []

        def launch(plan, row):
            tasks.append(
                asyncio.create_task(
                    replay_m5._consume(
                        FakeEngine(), plan, row, None, row["planned_ns"] + 2_000_000_000
                    )
                )
            )

        await replay_m5._open_loop(plans, rows, launch)
        await asyncio.gather(*tasks)
        return rows

    failed, complete = asyncio.run(drive())
    assert entered == ["early", "later"]
    assert failed["status"] == "failed"
    assert failed["first_token_ns"] is None
    assert complete["status"] == "completed"
    assert complete["token_ids"] == [10, 11]
    assert complete["first_token_ns"] == complete["last_token_ns"]
    assert complete["first_token_ns"] == complete["chunk_tokens"][0]["at_ns"]
    assert complete["submit_ns"] <= failed["end_ns"]


def test_expired_before_submit_stays_in_ledger(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import replay_m5

    plan = {"request_id": "expired"}
    row = replay_m5._entry(plan, 0)
    asyncio.run(replay_m5._consume(None, plan, row, None, 0))
    assert row["status"] == "timeout"
    assert row["submit_ns"] is None
    assert row["end_ns"] is not None
    assert row["token_ids"] == []


def test_batch_materialization_loads_tokenizer_once(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import prepare_m5_inputs

    calls = []

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, *, local_files_only, trust_remote_code):
            calls.append((path, local_files_only, trust_remote_code))
            return SimpleNamespace(encode=lambda text, add_special_tokens: encode(text))

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=AutoTokenizer))
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            [
                {"id": "mixed-01", "scenario": "mixed", "seed": 42, "count": 10, "rate_rps": 6},
                {
                    "id": "serial-01",
                    "scenario": "serial",
                    "seed": 43,
                    "count": 10,
                    "interval_ms": 200,
                },
            ]
        )
    )
    output = tmp_path / "tables"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_m5_inputs.py",
            "--config",
            str(ROOT / "configs/m5-performance.json"),
            "--model",
            str(tmp_path / "model"),
            "--batch",
            str(spec),
            "--output",
            str(output),
        ],
    )
    prepare_m5_inputs.main()
    assert len(calls) == 1
    assert json.loads((output / "mixed-01.json").read_text())["seed"] == 42
    assert json.loads((output / "serial-01.json").read_text())["seed"] == 43
    with pytest.raises(SystemExit):
        prepare_m5_inputs.main()
    assert len(calls) == 1
