"""Private M3 observation transport; no vLLM imports or scheduling operations.

Observation failures are sticky evidence failures, never replacements for the
observed method's return value or exception. JSONL writes add CPU/I/O overhead;
these runs establish functionality only, not timing or performance.
"""

import hashlib
import inspect
import json
import os
import threading
import time
import weakref
from pathlib import Path

_lock = threading.RLock()
_pid = None
_seq = 0
_clock_domain = None
_errors = []
_scheduler = None
_batches = {}
_batch_sequence = 0


def qualified_type(value):
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def module_source(module):
    path = Path(inspect.getfile(module)).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def observation_error(operation, error):
    with _lock:
        _errors.append(
            {"operation": operation, "type": qualified_type(error), "message": str(error)}
        )


def observation_errors():
    return list(_errors)


def emit(event, **fields):
    """Write one private event without changing the observed control flow."""
    global _pid, _seq, _clock_domain
    try:
        with _lock:
            pid = os.getpid()
            if _pid != pid:
                _pid, _seq = pid, 0
                boot = Path("/proc/sys/kernel/random/boot_id")
                host_id = boot.read_text().strip() if boot.exists() else "unverified"
                _clock_domain = f"monotonic_ns:boot:{host_id}"
            _seq += 1
            data = {
                "schema": 1,
                "event": event,
                "pid": pid,
                "seq": _seq,
                "monotonic_ns": time.monotonic_ns(),
                "clock_domain": _clock_domain,
                "engine_id": os.environ.get("M3_ENGINE_ID"),
                "mode": os.environ.get("M3_MODE"),
                "phase": os.environ.get("M3_PHASE"),
                **fields,
            }
            campaign_id = os.environ.get("M3_CAMPAIGN_ID")
            campaign_sha256 = os.environ.get("M3_CAMPAIGN_SHA256")
            if campaign_id is not None or campaign_sha256 is not None:
                if (
                    not campaign_id
                    or not campaign_sha256
                    or len(campaign_sha256) != 64
                    or any(char not in "0123456789abcdef" for char in campaign_sha256)
                    or not os.environ.get("SLURM_JOB_ID", "").isdigit()
                    or any(
                        key in fields
                        for key in ("campaign_id", "campaign_sha256", "campaign_job_id")
                    )
                ):
                    raise ValueError("invalid or conflicting M3 campaign event binding")
                data.update(
                    campaign_id=campaign_id,
                    campaign_sha256=campaign_sha256,
                    campaign_job_id=os.environ["SLURM_JOB_ID"],
                )
            root = Path(os.environ["M3_TRACE_DIR"])
            root.mkdir(parents=True, exist_ok=True)
            payload = (json.dumps(data, sort_keys=True, allow_nan=False) + "\n").encode()
            fd = os.open(root / f"{pid}.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                if os.write(fd, payload) != len(payload):
                    raise OSError("short observation write")
            finally:
                os.close(fd)
    except Exception as error:
        observation_error(f"emit:{event}", error)


def observe(operation, callback):
    """Best effort observer body, with a sticky error returned in checkpoints."""
    try:
        return callback()
    except Exception as error:
        observation_error(operation, error)
        return None


def register_scheduler(scheduler):
    global _scheduler
    _scheduler = weakref.ref(scheduler)


def get_scheduler():
    return None if _scheduler is None else _scheduler()


def output_descriptor(output):
    data = {
        "num_scheduled_tokens": dict(output.num_scheduled_tokens),
        "total_num_scheduled_tokens": output.total_num_scheduled_tokens,
        "new_req_ids": [r.req_id for r in output.scheduled_new_reqs],
        "cached_req_ids": list(output.scheduled_cached_reqs.req_ids),
        "finished_req_ids": sorted(output.finished_req_ids),
    }
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    data["output_sha256"] = hashlib.sha256(payload).hexdigest()
    return data


def register_batch(output):
    """Sidecar only: pin output until native consumption, never mutate it."""
    global _batch_sequence
    _batch_sequence += 1
    batch_id = f"{os.getpid()}:{_batch_sequence}"
    if id(output) in _batches:
        raise RuntimeError("native scheduler reused an in-flight output object")
    _batches[id(output)] = (output, batch_id)
    return batch_id


def batch_id_for(output):
    entry = _batches.get(id(output))
    if entry is None or entry[0] is not output:
        raise RuntimeError("missing same-process scheduler/runner batch binding")
    return entry[1]


def consume_batch(output):
    batch_id = batch_id_for(output)
    del _batches[id(output)]
    return batch_id


def pending_batches():
    return [entry[1] for entry in _batches.values()]


def model_output_fields(output):
    return {
        "output_type": qualified_type(output),
        "req_ids": list(output.req_ids),
        "sampled_token_ids": [list(tokens) for tokens in output.sampled_token_ids],
    }


def observe_materialization(output, batch_id):
    """Wrap native get_output in place; never invoke it ahead of its consumer.

    The original object and native method are preserved. Every invocation calls
    the original exactly once; native copy-event synchronization stays there.
    """
    # A stored bound method would create output -> wrapper -> bound method ->
    # output, retaining native GPU tensors until cyclic GC. Keep no such cycle.
    original = output.get_output.__func__
    output_ref = weakref.ref(output)

    def get_output(*args, **kwargs):
        active_output = output_ref()
        if active_output is None:
            raise ReferenceError("native async output no longer exists")
        try:
            result = original(active_output, *args, **kwargs)
        except BaseException as error:
            emit("model_output_error", batch_id=batch_id, error_type=qualified_type(error))
            raise
        observe(
            "model_output_materialized",
            lambda: emit(
                "model_output_materialized", batch_id=batch_id, **model_output_fields(result)
            ),
        )
        return result

    output.get_output = get_output
    return output
