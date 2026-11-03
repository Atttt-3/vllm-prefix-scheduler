"""Local measurement-hook tests without a native engine or GPU."""

import sys
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
            **{name: [] for name in ("allocations", "evictions", "batches", "overhead", "errors")},
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

    state["tier"] = "profile"
    scheduler.schedule()
    measured = state["batches"][1]
    assert measured["schedule_envelope_ns"] >= measured["schedule_ns"]
    assert [row["batch_id"] for row in state["overhead"]] == [2]
    assert not state["errors"]
