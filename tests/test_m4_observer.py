"""Local M4 observer contracts; these tests do not stand in for native/GPU runs."""

import gc
import importlib.util
import json
import os
import subprocess
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

from vllm_prefix_scheduler.m4_observer import WeakIdentityRegistry, storage_snapshot

ROOT = Path(__file__).parents[1]


def test_weak_registration_does_not_hold_historical_objects():
    class Request:
        pass

    registry = WeakIdentityRegistry()
    previous_generation = -1
    for _ in range(100):
        request = Request()
        ref = weakref.ref(request)
        binding = registry.binding(request)
        assert registry.binding(request) == binding
        assert binding["generation"] > previous_generation
        previous_generation = binding["generation"]
        del request
        gc.collect()
        assert ref() is None
        assert len(registry) == 0
    assert registry.next_generation == 100


def test_probe_storage_snapshot_is_stable_and_detects_equal_value_replacements():
    class Storage:
        __slots__ = ("data",)

        @property
        def dangerous(self):
            raise AssertionError("snapshot called a property")

    pool = Storage()
    pool.data = [1, 2]
    request = Storage()
    request.data = {"tokens": [7, 8]}
    manager = SimpleNamespace(block_pool=pool, stats={"hits": 3}, events=[])
    instance = SimpleNamespace(
        kv_cache_manager=manager,
        kv_metrics_collector={"queries": 2},
        waiting=[request],
        running=[],
        kv_holding_waiting=[],
        deferred_waiting=set(),
    )
    first = storage_snapshot(instance, request)
    assert first == storage_snapshot(instance, request)
    pool.data = [1, 2]
    assert first != storage_snapshot(instance, request)
    second = storage_snapshot(instance, request)
    manager.events.append("new_event")
    assert second != storage_snapshot(instance, request)


def test_journal_reader_preserves_incomplete_line_and_reads_only_new_records(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "m4_engine_fixture", ROOT / "scripts/m4_engine_fixture.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    journal = module.JournalReader(tmp_path)
    path = tmp_path / "1.jsonl"
    path.write_text('{"seq":1}\n{"seq":')
    assert journal.read() == [{"seq": 1}]
    assert journal.read() == []
    with path.open("a") as handle:
        handle.write("2}\n")
    assert journal.read() == [{"seq": 2}]
    assert journal.read() == []


def test_gpu_fixture_refuses_without_allocation_before_creating_output(tmp_path):
    env = dict(os.environ)
    env.pop("SLURM_JOB_ID", None)
    command = [
        sys.executable,
        str(ROOT / "scripts/m4_engine_fixture.py"),
        "--model",
        str(tmp_path),
        "--inputs",
        str(tmp_path / "missing.json"),
        "--sources",
        str(tmp_path / "missing.json"),
        "--output",
        str(tmp_path / "trace"),
        "--mode",
        "guarded",
    ]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "requires an existing Slurm allocation" in result.stderr
    assert not (tmp_path / "trace").exists()


def test_gpu_fixture_optimized_interpreter_refuses(tmp_path):
    result = subprocess.run(
        [sys.executable, "-O", str(ROOT / "scripts/m4_engine_fixture.py")],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "refuses optimized interpreter" in result.stderr


def test_identity_helper_does_not_edit_object():
    class Request:
        def __init__(self):
            self.request_id = "reused"

    a, b = Request(), Request()
    registry = WeakIdentityRegistry()
    before = json.dumps(vars(a), sort_keys=True)
    first, second = registry.binding(a), registry.binding(b)
    assert first != second
    assert first["generation"] < second["generation"]
    assert json.dumps(vars(a), sort_keys=True) == before
