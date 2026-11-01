"""Local interval accounting plus opt-in, unmocked pinned-runtime regression.

Set M1_NATIVE_MODEL_CONFIG to an existing model metadata directory to execute
the native CLI. M1_NATIVE_PYTHON optionally selects its interpreter; otherwise
the pytest interpreter is used. An opted-in unavailable runtime fails the test.
"""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path

if __name__ != "__main__":
    import pytest

    from vllm_prefix_scheduler.native_replay import DispatchLedger

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/native-bc-replay.json"


def assert_trace(trace_path, config=None):
    """Independently derive the causal result from events, not summary.json.

    This helper is also callable against the immutable archived native trace.
    It performs no vLLM operations and cannot alter native cache state.
    """
    if not __debug__:
        raise RuntimeError(
            "M1 validation requires assertions enabled; remove -O/-OO and unset PYTHONOPTIMIZE"
        )
    if config is None:
        config = json.loads(CONFIG.read_text())
    cases = defaultdict(list)
    for line in Path(trace_path).read_text().splitlines():
        event = json.loads(line)
        cases[event["case"]].append(event)
    expected_cases = {
        f"{repeat}-{capacity}-{''.join(order)}"
        for repeat in range(config["repeats"])
        for capacity in config["capacities"]
        for order in config["orders"]
    }
    assert set(cases) == expected_cases
    results = {}
    warm_states = {}
    shared_hashes = None
    for case, events in cases.items():
        _, capacity, order = case.split("-")
        indexed = list(enumerate(events))
        warm_rows = [(i, e) for i, e in indexed if e["event"] == "A_cached_and_evictable"]
        assert len(warm_rows) == 1
        warm_index, warm_event = warm_rows[0]
        warm = warm_event["state"]
        cached = warm_event["cached_blocks"]
        assert len(cached) == 2
        a_hashes = {b["hash"] for b in cached}
        assert len(a_hashes) == 2
        if shared_hashes is None:
            shared_hashes = a_hashes
        assert a_hashes == shared_hashes
        if capacity in warm_states:
            assert warm == warm_states[capacity]
        else:
            warm_states[capacity] = warm
        for block in cached:
            assert block["ref_cnt"] == 0 and not block["null"]
            assert block["id"] in warm["free_queue"]
            assert block["id"] in warm["hash_mapping"][block["hash"]]
        assert warm["requests"]["A"]["status"] == "FINISHED_LENGTH_CAPPED"
        assert warm["requests"]["A"]["in_flight"] == 0
        assert warm["requests"]["A"]["output_placeholders"] == 0

        enqueue = [e for e in events if e["event"] == "enqueue" and e["ids"] == ["B", "C"]]
        assert len(enqueue) == 1
        assert enqueue[0]["arrivals"] == {"B": 1.0, "C": 2.0}
        assert enqueue[0]["state"]["waiting"] == ["B", "C"]
        reorder = [e for e in events if e["event"] == "test_only_reorder"]
        assert len(reorder) == (1 if order == "CB" else 0)
        if reorder:
            assert reorder[0]["before"] == ["B", "C"]
            assert reorder[0]["after"] == ["C", "B"]
            assert reorder[0]["state"]["waiting"] == ["C", "B"]
            assert reorder[0]["count"] == 1

        calls = {e["call"]: (i, e) for i, e in indexed if e["event"] == "call"}
        returns = {e["call"]: (i, e) for i, e in indexed if e["event"] == "return"}
        assert calls.keys() == returns.keys()
        for call, (i, event) in calls.items():
            j, returned = returns[call]
            assert i < j
            assert event["method"] == returned["method"]
            assert event["owner"] == returned["owner"]

        c_lookups = [
            (i, e)
            for i, e in indexed
            if e["event"] == "return" and e["method"] == "get_computed_blocks" and e["owner"] == "C"
        ]
        assert c_lookups
        c_lookup_index, c_lookup = c_lookups[0]
        expected_hit = 0 if (capacity, order) == ("pressure", "BC") else 32
        assert c_lookup["result"]["hit_tokens"] == expected_hit

        a_evictions = []
        for call, (i, event) in calls.items():
            if event["method"] != "_maybe_evict_cached_block":
                continue
            block_id = event["details"]["block_id"]
            before = next(b for b in event["state"]["blocks"] if b["id"] == block_id)
            if before["hash"] not in a_hashes:
                continue
            j, returned = returns[call]
            assert returned["result"] is True
            assert before["ref_cnt"] == 0
            assert block_id in event["state"]["hash_mapping"][before["hash"]]
            assert block_id not in returned["state"]["hash_mapping"].get(before["hash"], [])
            a_evictions.append((i, j, event, before["hash"]))

        if capacity == "ample":
            assert not a_evictions
        elif order == "BC":
            assert {item[3] for item in a_evictions if item[1] < c_lookup_index} == a_hashes
            assert all(item[2]["owner"] == "B" for item in a_evictions)
            # Establish allocation ownership/nesting, rather than inferring
            # eviction merely from an absent hash in a final snapshot.
            for i, j, _event, _ in a_evictions:
                assert warm_index < i < j < c_lookup_index
                assert any(
                    begin < i < j < returns[call][0]
                    and call_event["owner"] == "B"
                    and call_event["method"] == "allocate_slots"
                    and returns[call][1]["result"]["success"]
                    for call, (begin, call_event) in calls.items()
                )
        else:
            assert all(item[0] > c_lookup_index for item in a_evictions)
            touches = [
                e
                for e in events
                if e["event"] == "return" and e["method"] == "touch" and e["owner"] == "C"
            ]
            assert touches
            assert any(
                all(
                    next(b for b in e["state"]["blocks"] if b["hash"] == h)["ref_cnt"] > 0
                    for h in a_hashes
                )
                for e in touches
            )

        # Independent interval union accounting detects both omitted chunks
        # and double-counted work, without using the implementation's ledger.
        positions = defaultdict(set)
        counts = defaultdict(lambda: {"prefill": 0, "decode": 0, "repeated": 0})
        lengths = {r["id"]: len(r["tokens"]) for r in config["requests"]}
        for event in events:
            if event["event"] != "dispatch":
                continue
            for interval in event["intervals"]:
                rid = interval["request_id"]
                dispatched = set(range(interval["start"], interval["end"]))
                assert len(dispatched) == interval["scheduled_tokens"]
                prompt = {p for p in dispatched if p < lengths[rid]}
                assert len(prompt) == interval["prefill_tokens"]
                assert len(dispatched - prompt) == interval["decode_tokens"]
                assert len(prompt & positions[rid]) == interval["repeated_prefill_tokens"]
                assert (
                    len((dispatched - prompt) & positions[rid])
                    == interval["repeated_decode_tokens"]
                )
                counts[rid]["prefill"] += len(prompt)
                counts[rid]["decode"] += len(dispatched - prompt)
                counts[rid]["repeated"] += len(dispatched & positions[rid])
                positions[rid].update(dispatched)
        assert counts["A"]["prefill"] == 33
        assert counts["B"]["prefill"] == 49
        assert counts["C"]["prefill"] == 33 - expected_hit
        assert all(c["decode"] == 2 and c["repeated"] == 0 for c in counts.values())

        pending = deque()
        for event in events:
            if event["event"] == "schedule_after":
                pending.append(event["step"])
                assert len(pending) == event["pending_outputs"]
            elif event["event"] == "output_after":
                assert pending.popleft() == event["output_step"]
                assert len(pending) == event["pending_outputs"]
            elif event["event"] == "drained":
                assert not pending and event["pending_outputs"] == 0
        assert not pending
        drained = [e for e in events if e["event"] == "drained"]
        assert [e["phase"] for e in drained] == ["warmup", "BC"]
        final = drained[-1]["state"]
        assert not final["waiting"] and not final["running"]
        assert not final["kv_holding_waiting"] and not final["deferred_waiting"]
        assert len(final["free_queue"]) == config["capacities"][capacity] - 1
        assert len(set(final["free_queue"])) == len(final["free_queue"])
        assert all(b["ref_cnt"] == 0 for b in final["blocks"] if not b["null"])
        assert all(
            not block_ids for group in final["request_blocks"] for block_ids in group.values()
        )
        for request in final["requests"].values():
            assert request["status"] == "FINISHED_LENGTH_CAPPED"
            assert len(request["output_tokens"]) == config["max_tokens"]
            assert request["in_flight"] == request["output_placeholders"] == 0
            assert request["stale_output_tokens"] == request["preemptions"] == 0
        results[case] = dict(counts)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Independently audit archived native trace")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    arguments = parser.parse_args()
    audited = assert_trace(arguments.trace, json.loads(arguments.config.read_text()))
    print(json.dumps({"audited_native_cases": len(audited), "causal_trace_verified": True}))
    raise SystemExit(0)


def test_dispatch_intervals_separate_prompt_decode_and_repeated_positions():
    ledger = DispatchLedger()
    chunks = [ledger.record("C", 0, 16, 33), ledger.record("C", 16, 16, 33)]
    chunks.append(ledger.record("C", 32, 3, 33))
    assert sum(item["prefill_tokens"] for item in chunks) == 33
    assert sum(item["decode_tokens"] for item in chunks) == 2
    assert sum(item["repeated_prefill_tokens"] for item in chunks) == 0

    # A rollback can redispatch old prompt/output positions and new output
    # positions together. It is not equivalent to prompt length minus hits.
    repeated = ledger.record("C", 8, 28, 33)
    assert repeated == {
        "start": 8,
        "end": 36,
        "scheduled_tokens": 28,
        "prefill_tokens": 25,
        "repeated_prefill_tokens": 25,
        "decode_tokens": 3,
        "repeated_decode_tokens": 2,
    }
    other_request = ledger.record("B", 8, 28, 33)
    assert other_request["repeated_prefill_tokens"] == 0
    assert other_request["repeated_decode_tokens"] == 0


def test_cached_prefix_is_not_mistaken_for_dispatched_work():
    ledger = DispatchLedger()
    tail = ledger.record("C", 32, 2, 33)
    assert tail["prefill_tokens"] == 1
    assert tail["decode_tokens"] == 1
    # Prefix positions were reused, never dispatched. A later rollback that
    # dispatches them for the first time must not label them repeated work.
    replay = ledger.record("C", 0, 34, 33)
    assert replay["prefill_tokens"] == 33
    assert replay["repeated_prefill_tokens"] == 1
    assert replay["repeated_decode_tokens"] == 1


@pytest.mark.parametrize("start,count", [(-1, 1), (0, 0), (0, -1)])
def test_invalid_dispatch_interval_fails(start, count):
    with pytest.raises(ValueError, match="Invalid dispatched interval"):
        DispatchLedger().record("C", start, count, 33)


def run_cli_with_optimization(script, arguments, flag, optimization):
    """Exercise actual process flags, including environment-driven optimization."""
    environment = dict(os.environ)
    environment.pop("PYTHONOPTIMIZE", None)
    if optimization is not None:
        environment["PYTHONOPTIMIZE"] = optimization
    command = [sys.executable]
    if flag is not None:
        command.append(flag)
    return subprocess.run(
        [*command, str(script), *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


@pytest.mark.parametrize(
    "flag,optimization", [(None, None), ("-O", None), ("-OO", None), (None, "1"), (None, "2")]
)
def test_empty_trace_never_reports_verified(tmp_path, flag, optimization):
    trace = tmp_path / "empty.jsonl"
    trace.write_text("")
    completed = run_cli_with_optimization(
        Path(__file__), ["--trace", str(trace)], flag, optimization
    )
    assert completed.returncode != 0
    assert "causal_trace_verified" not in completed.stdout
    if flag is not None or optimization is not None:
        assert "M1 validation requires assertions enabled" in completed.stderr
    else:
        assert "AssertionError" in completed.stderr


@pytest.mark.parametrize(
    "flag,optimization", [(None, None), ("-O", None), ("-OO", None), (None, "1"), (None, "2")]
)
def test_native_entry_rejects_optimized_interpreter_before_imports(flag, optimization):
    completed = run_cli_with_optimization(
        ROOT / "scripts/replay_bc.py", ["--help"], flag, optimization
    )
    if flag is not None or optimization is not None:
        assert completed.returncode != 0
        assert "M1 validation requires assertions enabled" in completed.stderr
        assert "usage:" not in completed.stdout
    else:
        assert completed.returncode == 0
        assert "usage:" in completed.stdout


@pytest.mark.native
def test_native_cli_replays_causal_matrix(tmp_path):
    model_config = os.environ.get("M1_NATIVE_MODEL_CONFIG")
    if not model_config:
        pytest.skip("Set M1_NATIVE_MODEL_CONFIG to opt into the pinned Linux native runtime")
    interpreter = os.environ.get("M1_NATIVE_PYTHON", sys.executable)
    output = tmp_path / "native-replay"
    environment = {
        **os.environ,
        "PYTHONHASHSEED": "42",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "HF_HUB_OFFLINE": "1",
    }
    completed = subprocess.run(
        [
            interpreter,
            str(ROOT / "scripts/replay_bc.py"),
            "--config",
            str(CONFIG),
            "--model-config",
            model_config,
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    runtime = json.loads((output / "runtime.json").read_text())
    assert runtime["vllm_version"] == "0.32.0"
    assert runtime["modules"] and all(m["matches"] for m in runtime["modules"].values())
    config = json.loads(CONFIG.read_text())
    assert config["repeats"] >= 3
    assert_trace(output / "trace.jsonl", config)
