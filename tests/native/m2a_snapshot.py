"""Read-only snapshots of the pinned native metadata graph (no draining APIs)."""

import dataclasses
import enum
import functools
import json
import random
import types
from collections import deque

REQUEST_FIELDS = set(
    """
request_id client_index priority sampling_params pooling_params lora_request
structured_output_request arrival_time status events stop_reason kv_transfer_params
ec_transfer_params max_tokens kv_cache_report_mode prompt_token_ids prompt_embeds
prompt_is_token_ids _prompt_embeds_per_block_hashes num_prompt_tokens _output_token_ids
_all_token_ids num_output_placeholders num_stale_output_tokens drop_stale_output
num_in_flight_tokens next_decode_eligible_step last_sched_seq spec_token_ids
num_computed_tokens cache_salt mm_features output_token_ids all_token_ids trace_headers
session_id kv_hints is_prefill_chunk shared_prefix_boundary replay_start num_nans_in_logits
num_preemptions prefill_stats spec_decode_metrics block_hashes _block_hasher
skip_reading_prefix_cache resumable streaming_queue abort_immediately
""".split()
)
POOL_FIELDS = set(
    """num_gpu_blocks medium enable_caching hash_block_size blocks
free_block_queue cached_block_hash_to_block cached_block_hashes_by_block null_block
enable_kv_cache_events kv_event_queue metrics_collector _reuse_watchers""".split()
)
BLOCK_FIELDS = set(
    """block_id ref_cnt pool _block_hash _block_hash_num_tokens
prev_free_block next_free_block is_null""".split()
)
SINGLE_FIELDS = set(
    """scheduler_block_size cache_hit_alignment_tokens dcp_world_size
pcp_world_size block_size kv_cache_spec block_pool enable_caching
max_admission_blocks_per_request _record_new_block_ids new_block_ids req_to_blocks
num_cached_block kv_cache_group_id _null_block use_eagle shared_prefix_checkpoint
_partial_hit_reqs _pending_cow_copies _pending_boundary_state_offloads""".split()
)
# Only native methods wrapped by the observer may appear in instance dictionaries.
OBSERVED = {
    "get_computed_blocks",
    "allocate_slots",
    "cache_blocks",
    "free",
    "get_new_blocks",
    "_maybe_evict_cached_block",
    "touch",
    "free_blocks",
}


def _fields(obj):
    """Read storage only; never inspect arbitrary properties/cached properties."""
    fields = dict(getattr(obj, "__dict__", {}))
    for cls in type(obj).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name not in {"__dict__", "__weakref__"}:
                try:
                    fields[name] = object.__getattribute__(obj, name)
                except AttributeError:
                    pass
    return fields


def _check_fields(obj, expected):
    actual = set(_fields(obj)) - OBSERVED
    if actual != expected:
        raise RuntimeError(
            f"Unreviewed {type(obj).__name__} fields: extra={actual - expected}, missing={expected - actual}"
        )


def value(obj, seen=None):
    """Serialize only storage-backed native value types; unknown types fail closed."""
    if obj is None or type(obj) in (str, int, bool, float):
        return obj
    if isinstance(obj, enum.Enum):
        return {"enum": f"{type(obj).__module__}.{type(obj).__name__}", "name": obj.name}
    if isinstance(obj, bytes):
        return {"bytes": obj.hex()}
    if isinstance(obj, (types.FunctionType, types.MethodType)):
        func = getattr(obj, "__func__", obj)
        return {"callable": f"{func.__module__}.{func.__qualname__}"}
    if isinstance(obj, functools.partial):
        return {"partial": value(obj.func), "args": value(obj.args), "kwargs": value(obj.keywords)}
    if type(obj).__module__ == "torch" and type(obj).__name__ in {"dtype", "device"}:
        return {"torch_type": str(obj)}
    seen = {} if seen is None else seen
    identity = id(obj)
    if identity in seen:
        return {"ref": seen[identity]}
    seen[identity] = len(seen)
    if isinstance(obj, dict):
        return {"dict": [[value(k, seen), value(v, seen)] for k, v in obj.items()]}
    if isinstance(obj, (list, tuple, deque)):
        return [value(v, seen) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return {
            "set": sorted((value(v, {}) for v in obj), key=lambda x: json.dumps(x, sort_keys=True))
        }
    cls = type(obj)
    if cls.__name__ == "KVCacheBlock":
        return {"block": obj.block_id}
    if cls.__name__ == "Request":
        return {"request": obj.request_id}
    if cls.__name__ == "ConstantList" and cls.__module__.startswith("vllm."):
        return {
            "constant_list": list(obj),
            "storage": {k: value(v, seen) for k, v in _fields(obj).items()},
        }
    if cls.__module__.startswith("vllm."):
        if hasattr(cls, "__struct_fields__"):
            names = cls.__struct_fields__
            fields = {name: object.__getattribute__(obj, name) for name in names}
        elif dataclasses.is_dataclass(obj):
            fields = {f.name: object.__getattribute__(obj, f.name) for f in dataclasses.fields(obj)}
        elif cls.__name__ in {
            "BlockMetrics",
            "BlockMetricsState",
            "KVCacheMetricsCollector",
            "PrefixCacheStats",
            "PrefillStats",
        }:
            fields = _fields(obj)
        else:
            raise TypeError(f"Unreviewed snapshot type {cls.__module__}.{cls.__name__}")
        return {
            "type": f"{cls.__module__}.{cls.__name__}",
            "fields": {k: value(v, seen) for k, v in fields.items()},
        }
    raise TypeError(f"Unreviewed snapshot type {cls.__module__}.{cls.__name__}")


def identity_graph(roots):
    """Storage-only object/alias sidecar; includes equal-content replacements.

    Native block/pool cycles terminate at the first identity. Functions and
    immutable scalar/type objects are leaves; no descriptors are evaluated.
    """
    identities, aliases, visited = {}, {}, {}

    def walk(obj, path):
        if obj is None or isinstance(obj, (str, int, float, bool, bytes, enum.Enum)):
            return
        if isinstance(obj, (types.FunctionType, types.MethodType)):
            identities[path] = id(obj)
            return
        if isinstance(obj, type):
            return
        if type(obj).__module__ == "torch":
            return
        oid = id(obj)
        identities[path] = oid
        if oid in visited:
            aliases[path] = visited[oid]
            return
        visited[oid] = path
        if isinstance(obj, dict):
            for i, (k, v) in enumerate(obj.items()):
                walk(k, f"{path}.key[{i}]")
                walk(v, f"{path}.value[{i}]")
        elif isinstance(obj, (list, tuple, deque)):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")
        elif isinstance(obj, (set, frozenset)):
            for i, v in enumerate(sorted(obj, key=lambda x: json.dumps(value(x), sort_keys=True))):
                walk(v, f"{path}{{{i}}}")
        elif isinstance(obj, functools.partial):
            walk(obj.args, path + ".args")
            walk(obj.keywords, path + ".keywords")
        elif type(obj).__module__.startswith("vllm."):
            fields = _fields(obj)
            if hasattr(type(obj), "__struct_fields__"):
                fields.update(
                    {
                        name: object.__getattribute__(obj, name)
                        for name in type(obj).__struct_fields__
                    }
                )
            for name, v in fields.items():
                if name not in OBSERVED:
                    walk(v, path + "." + name)
        else:
            raise TypeError(
                f"Unreviewed identity graph type {type(obj).__module__}.{type(obj).__name__}"
            )

    for name, obj in roots.items():
        walk(obj, name)
    return {"identities": identities, "aliases": aliases}


def compact_request(r):
    return {
        "status": r.status.name,
        "computed_optimistic": r.num_computed_tokens,
        "in_flight": r.num_in_flight_tokens,
        "output_placeholders": r.num_output_placeholders,
        "stale_output_tokens": r.num_stale_output_tokens,
        "output_tokens": list(r.output_token_ids),
        "preemptions": r.num_preemptions,
        "is_prefill_chunk": r.is_prefill_chunk,
    }


def snapshot(scheduler, requests, generations=None, full=False):
    pool = scheduler.kv_cache_manager.block_pool
    manager = scheduler.kv_cache_manager
    generations = generations or {}
    mapping = {}
    for key, blocks in pool.cached_block_hash_to_block._cache.items():
        mapping[key.hex()] = list(blocks) if isinstance(blocks, dict) else [blocks.block_id]
    state = {
        "waiting": [r.request_id for r in scheduler.waiting],
        "kv_holding_waiting": [r.request_id for r in scheduler.kv_holding_waiting],
        "deferred_waiting": sorted(r.request_id for r in scheduler.deferred_waiting),
        "running": [r.request_id for r in scheduler.running],
        "active_requests": list(scheduler.requests),
        "requests": {key: compact_request(r) for key, r in requests.items()},
        "blocks": [
            {
                "id": b.block_id,
                "ref_cnt": b.ref_cnt,
                "hash": b._block_hash.hex() if b._block_hash else None,
                "hash_tokens": b._block_hash_num_tokens,
                "null": b.is_null,
            }
            for b in pool.blocks
        ],
        "hash_mapping": mapping,
        "free_queue": [b.block_id for b in pool.free_block_queue.get_all_free_blocks()],
        "request_blocks": [
            {rid: [b.block_id for b in bs] for rid, bs in m.req_to_blocks.items()}
            for m in manager.coordinator.single_type_managers
        ],
    }
    if not full:
        return state
    _check_fields(pool, POOL_FIELDS)
    request_fields = {}
    identities = {}
    for key, r in requests.items():
        _check_fields(r, REQUEST_FIELDS)
        fields = _fields(r)
        request_fields[key] = {name: value(v) for name, v in fields.items()}
        identities[f"request:{key}"] = id(r)
        for name, v in fields.items():
            if isinstance(v, (list, dict, deque)) or type(v).__name__ == "ConstantList":
                identities[f"request:{key}.{name}"] = id(v)
    block_fields = []
    queue = pool.free_block_queue

    def link(b):
        if b is None:
            return None
        if b is queue.fake_free_list_head:
            return "head"
        if b is queue.fake_free_list_tail:
            return "tail"
        return b.block_id

    for b in pool.blocks:
        _check_fields(b, BLOCK_FIELDS)
        block_fields.append(
            {
                "block_id": b.block_id,
                "ref_cnt": b.ref_cnt,
                "pool_identity": id(b.pool),
                "hash": value(b._block_hash),
                "hash_num_tokens": b._block_hash_num_tokens,
                "prev": link(b.prev_free_block),
                "next": link(b.next_free_block),
                "is_null": b.is_null,
            }
        )
        identities[f"block:{b.block_id}"] = id(b)
    singles = []
    for m in manager.coordinator.single_type_managers:
        _check_fields(m, SINGLE_FIELDS)
        singles.append({k: value(v) for k, v in _fields(m).items() if k not in {"block_pool"}})
    state["full"] = {
        "graph": identity_graph(
            {
                "requests": requests,
                "manager": manager,
                "metrics": scheduler.kv_metrics_collector,
                "waiting": scheduler.waiting,
                "holding": scheduler.kv_holding_waiting,
                "deferred": scheduler.deferred_waiting,
                "running": scheduler.running,
            }
        ),
        "requests": request_fields,
        "identities": identities,
        "blocks": block_fields,
        "hash_mapping_ordered": value(pool.cached_block_hash_to_block._cache),
        "reverse_hash_mapping": value(pool.cached_block_hashes_by_block),
        "reuse_watchers": value(pool._reuse_watchers),
        "single_managers": singles,
        "queue_endpoints": [
            link(queue.fake_free_list_head.next_free_block),
            link(queue.fake_free_list_tail.prev_free_block),
        ],
        "queue_count": queue.num_free_blocks,
        "pool_flags": {
            k: value(getattr(pool, k))
            for k in (
                "num_gpu_blocks",
                "medium",
                "enable_caching",
                "hash_block_size",
                "enable_kv_cache_events",
            )
        },
        "stats": value(manager.prefix_cache_stats),
        "events": value(pool.kv_event_queue),
        "metrics": value(scheduler.kv_metrics_collector),
        "manager_fields": {
            k: value(v)
            for k, v in _fields(manager).items()
            if k not in {"coordinator", "block_pool", "metrics_collector"} | OBSERVED
        },
        "coordinator_fields": {
            k: value(v)
            for k, v in _fields(manager.coordinator).items()
            if k not in {"block_pool", "single_type_managers"}
        },
        "scheduler_lifecycle": {
            k: value(scheduler.__dict__[k])
            for k in (
                "_inflight_prefills",
                "finished_req_ids",
                "reset_preempted_req_ids",
                "deferred_frees",
                "sched_step_seq",
                "processed_step_seq",
                "current_step",
            )
            if k in scheduler.__dict__
        },
        "random_state": value(random.getstate()),
    }
    return state
