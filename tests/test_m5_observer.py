"""Local measurement-hook tests without a native engine or GPU."""

import hashlib
import json
import sys
from array import array
from types import SimpleNamespace

import pytest

from vllm_prefix_scheduler import m5_observer


@pytest.fixture
def scheduler(monkeypatch):
    class BlockPool:
        def get_num_free_blocks(self):
            return 32

        def _remove_cached_block_hashes(self, block):
            return block.hashes

        def _maybe_evict_cached_block(self, block):
            return bool(self._remove_cached_block_hashes(block))

    class CacheManager:
        def __init__(self):
            self.block_pool = BlockPool()

        def allocate_slots(self, request, blocks, **kwargs):
            for block in blocks:
                self.block_pool._maybe_evict_cached_block(block)
            return blocks

    class AsyncScheduler:
        def __init__(self):
            self.kv_cache_manager = CacheManager()
            self.waiting = []
            self.requests = {}

        def add_request(self, request):
            self.requests[request.request_id] = request
            return request

        def schedule(self, *, throttle_prefills):
            return SimpleNamespace(num_scheduled_tokens={})

    class PrefixAsyncScheduler(AsyncScheduler):
        pass

    for name, module in {
        "vllm.v1.core.kv_cache_utils": SimpleNamespace(get_block_hash=lambda value: value),
        "vllm.v1.core.sched.async_scheduler": SimpleNamespace(AsyncScheduler=AsyncScheduler),
        "vllm_prefix_scheduler.scheduler": SimpleNamespace(
            PrefixAsyncScheduler=PrefixAsyncScheduler
        ),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("M5_WINDOW", "16")
    monkeypatch.setattr(m5_observer, "_scheduler", None)
    monkeypatch.setattr(
        m5_observer,
        "_state",
        {
            "schema_version": 2,
            "stage": "measure",
            "tier": "light",
            "limit": 100,
            "pending": {},
            "_capture": m5_observer._capture_buffer(12),
            **{
                name: []
                for name in (
                    "allocations",
                    "evictions",
                    "batches",
                    "overhead",
                    "errors",
                    "enqueues",
                )
            },
        },
    )
    native, _ = m5_observer._define_native_classes()
    return native()


def test_v2_observer_binds_successful_native_evictions_to_allocation(scheduler):
    class Request:
        request_id = "request"

    manager = scheduler.kv_cache_manager
    cached = SimpleNamespace(block_id=4, hashes=[b"a" * 32, b"b" * 32])
    vacant = SimpleNamespace(block_id=5, hashes=[])
    request = Request()
    blocks = [cached, vacant]
    assert manager.allocate_slots(request, blocks, num_new_computed_tokens=16) is blocks
    assert manager.allocate_slots(request, [vacant]) == [vacant]

    state = m5_observer._state
    first, second = state["allocations"]
    assert (first["allocation_id"], first["eviction_count"], first["hit_tokens"]) == (1, 1, 16)
    assert (second["allocation_id"], second["eviction_count"]) == (2, 0)
    (eviction,) = state["evictions"]
    assert eviction["allocation_id"] == 1
    assert eviction["block_id"] == 4
    assert eviction["removed_cache_keys"] == [(b"a" * 32).hex(), (b"b" * 32).hex()]
    assert eviction["at_ns"] <= first["at_ns"] <= second["at_ns"]
    assert not state["errors"]

    # An unsupported eviction path keeps the native return and records a fault.
    assert manager.block_pool._maybe_evict_cached_block(cached) is True
    assert len(state["evictions"]) == 1
    assert state["errors"] == [
        {"operation": "eviction", "error": "M5 native eviction outside an allocation"}
    ]


def test_v2_scheduler_envelope_is_only_collected_in_profile(scheduler):
    state = m5_observer._state
    scheduler.schedule()
    assert "schedule_envelope_ns" not in state["batches"][0]
    assert not state["overhead"]
    assert scheduler._m5_timings is None

    state["tier"] = "profile"
    scheduler.schedule()
    measured = state["batches"][1]
    assert measured["schedule_envelope_ns"] >= measured["schedule_ns"]
    assert [row["batch_id"] for row in state["overhead"]] == [2]
    assert not state["errors"]


def test_enqueue_hashes_detached_values_after_drain(scheduler):
    class Request:
        request_id = "r"
        prompt_token_ids = [1, 2, 3]
        block_hashes = [bytearray(b"original")]

    request = Request()
    assert scheduler.add_request(request) is request
    state = m5_observer._state
    assert "prompt_sha256" not in state["enqueues"][0]
    assert "block_hashes" not in state["enqueues"][0]
    request.prompt_token_ids[0] = 99
    request.block_hashes[0][:] = b"changed"
    state["stage"] = "done"
    worker = m5_observer.M5WorkerExtension()
    exported = worker.m5_export()
    assert exported["enqueues"][0]["prompt_sha256"] == hashlib.sha256(b"[1,2,3]").hexdigest()
    assert exported["enqueues"][0]["block_hashes"] == [b"original".hex()]
    assert "_capture" not in exported
    assert "_raw_enqueue" not in exported["enqueues"][0]
    assert worker.m5_export() == exported
    assert not exported["errors"]
    json.dumps(exported, allow_nan=False)


def capture_positions(positions, *, consumed=True, output_ids=None, rows=None):
    state = m5_observer._state
    batch = {
        "runner": None,
        "materialized": False,
        "consumed": consumed,
        "scheduled": [{"request_id": "r", "generation": 7}],
    }
    state["batches"].append(batch)
    pinned = array("q", positions)
    slot = SimpleNamespace(numpy=lambda: pinned)
    record = {
        "slot": slot,
        "shape": {
            "effective_tokens": len(pinned),
            "padded_tokens": len(pinned) + 1,
            "rows": [("r", 0, len(pinned), 9)] if rows is None else rows,
        },
        "batch": batch,
        "position_copy_submitted": True,
    }
    output = SimpleNamespace(req_ids=["r"] if output_ids is None else output_ids)
    free = []
    m5_observer._observe("materialized", m5_observer._capture_materialized, record, output, free)
    assert free == [slot]
    return batch, pinned


def test_export_validates_exact_detached_positions_after_slot_reuse(scheduler):
    first, pinned = capture_positions([15, 16, 17])
    assert first["materialized"] is True
    assert first["runner"] is None
    pinned[:] = array("q", [100, 100, 100])
    second, _ = capture_positions([24])
    assert second["runner"] is None
    m5_observer._state["stage"] = "done"
    exported = m5_observer.M5WorkerExtension().m5_export()
    assert not exported["errors"]
    assert first["runner"] == {
        "effective_tokens": 3,
        "padded_tokens": 4,
        "rows": [
            {
                "request_id": "r",
                "generation": 7,
                "start": 15,
                "end": 18,
                "prefill_length": 9,
                "position_source": "runner_positions_d2h",
            }
        ],
    }
    assert second["runner"]["rows"][0]["start"] == 24
    assert "_raw_runner" not in first and "_raw_runner" not in second
    json.dumps(exported, allow_nan=False)


@pytest.mark.parametrize(
    ("positions", "output_ids", "rows", "issue"),
    [
        ([15, 99, 17], None, None, "not a contiguous request interval"),
        ([15, 16, 17], ["other"], None, "output request order changed"),
        ([15, 16, 17], None, [("r", 1, 3, 9)], "do not partition effective inputs"),
    ],
)
def test_bad_native_positions_reject_after_drain(scheduler, positions, output_ids, rows, issue):
    batch, _ = capture_positions(positions, output_ids=output_ids, rows=rows)
    assert batch["materialized"] is True and batch["runner"] is None
    assert not m5_observer._state["errors"]
    m5_observer._state["stage"] = "done"
    exported = m5_observer.M5WorkerExtension().m5_export()
    assert issue in exported["errors"][0]["error"]
    assert exported["batches"][0]["runner"] is None
    assert "_raw_runner" not in batch


def test_capture_exhaustion_and_premature_export_fail_closed(scheduler):
    class Request:
        request_id = "large"
        prompt_token_ids = list(range(13))
        block_hashes = []

    request = Request()
    assert scheduler.add_request(request) is request
    first, _ = capture_positions([1], consumed=False)
    overflow, _ = capture_positions(list(range(12)))
    assert first["materialized"] is True
    assert overflow["materialized"] is False
    assert m5_observer._state["_capture"]["position_count"] == 1
    assert m5_observer._state["_capture"]["prompt_count"] == 0
    exported = m5_observer.M5WorkerExtension().m5_export()
    assert not exported["enqueues"]
    assert first["runner"] is None
    assert overflow["runner"] is None
    errors = [row["error"] for row in exported["errors"]]
    assert any("enqueue prompt buffer exhausted" in error for error in errors)
    assert any("runner position buffer exhausted" in error for error in errors)
    assert "M5 observer export before full drain" in errors
    assert "M5 native runner capture was not materialized and consumed" in errors


def test_export_rejects_pointer_into_uncaptured_host_buffer(scheduler):
    batch, _ = capture_positions([15])
    offset, shape, ids = batch["_raw_runner"]
    batch["_raw_runner"] = (offset + 1, shape, ids)
    m5_observer._state["stage"] = "done"
    exported = m5_observer.M5WorkerExtension().m5_export()
    assert batch["runner"] is None
    assert "outside committed host bytes" in exported["errors"][0]["error"]


def test_runner_defers_host_read_until_native_materialization(scheduler, monkeypatch):
    activity = []
    stream_copies = []

    class TensorSlice:
        def __init__(self, tensor, indices):
            self.tensor = tensor
            self.indices = indices

        def copy_(self, source, *, non_blocking):
            assert non_blocking is True
            activity.append("copy_submitted")
            stream_copies.append(
                lambda: self.tensor.values.__setitem__(
                    self.indices, source.tensor.values[source.indices]
                )
            )

    class Tensor:
        ndim = 1
        dtype = "int64"

        def __init__(self, values):
            self.values = array("q", values)

        def __getitem__(self, indices):
            return TensorSlice(self, indices)

        def numel(self):
            return len(self.values)

        def numpy(self):
            assert activity[-1] == "native_materialized"
            activity.append("host_read")
            return self.values

    class ModelState:
        rope_state = None

    class AsyncOutput:
        def get_output(self):
            for copy in stream_copies:
                copy()
            activity.append("native_materialized")
            return SimpleNamespace(req_ids=["r"])

    class Runner:
        model_state = ModelState()
        speculator = pcp_manager = None
        use_pp = False
        device = "cuda:0"
        main_stream = "runner-stream"

        def execute_model(self, output):
            self.execute_model_state = SimpleNamespace(
                input_batch=SimpleNamespace(
                    num_tokens=2,
                    num_tokens_after_padding=3,
                    positions=Tensor([5, 6, -100]),
                    query_start_loc_np=[0, 2],
                    req_ids=["r"],
                    prefill_len_np=[9],
                )
            )

        def sample_tokens(self):
            return AsyncOutput()

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            empty=lambda size, **kwargs: Tensor([0] * size),
            int64="int64",
            cuda=SimpleNamespace(current_stream=lambda device: Runner.main_stream),
        ),
    )
    types = {
        Runner: "vllm.v1.worker.gpu.model_runner.GPUModelRunner",
        ModelState: "vllm.v1.worker.gpu.model_states.default.DefaultModelState",
    }
    monkeypatch.setattr(m5_observer, "qualified_type", lambda value: types[type(value)])
    runner = Runner()
    m5_observer._install_runner(
        runner,
        SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=8), max_concurrent_batches=1
        ),
    )
    state = m5_observer._state
    output = SimpleNamespace(total_num_scheduled_tokens=2)
    batch = {
        "runner": None,
        "materialized": False,
        "consumed": False,
        "scheduled": [{"request_id": "r", "generation": 1}],
    }
    state["batches"].append(batch)
    state["pending"][id(output)] = {"output": output, "batch": batch, "slot": None}
    runner.execute_model(output)
    native_output = runner.sample_tokens()
    assert activity == ["copy_submitted"]
    assert not batch["materialized"]
    assert native_output.get_output().req_ids == ["r"]
    assert activity == ["copy_submitted", "native_materialized", "host_read"]
    assert batch["materialized"] and batch["runner"] is None
    state["pending"].pop(id(output))
    batch["consumed"] = True
    state["stage"] = "done"
    exported = m5_observer.M5WorkerExtension().m5_export()
    assert not exported["errors"]
    assert batch["runner"]["rows"][0]["start"] == 5
    assert batch["runner"]["rows"][0]["end"] == 7
