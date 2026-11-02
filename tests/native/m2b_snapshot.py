"""M2B storage-only observations; native M2A snapshots remain unchanged."""

from m2a_snapshot import snapshot as native_snapshot


class ObjectRegistry:
    """Retain actual objects so observer tokens never alias recycled id() values."""

    def __init__(self):
        self.objects = []

    def token(self, request):
        for index, known in enumerate(self.objects):
            if known is request:
                return f"object:{index}"
        self.objects.append(request)
        return f"object:{len(self.objects) - 1}"


def guard_snapshot(scheduler, registry):
    guard = getattr(scheduler, "_wait_guard", None)
    if guard is None:
        return {"present": False, "records": []}
    return {
        "present": True,
        "records": [
            {
                "object": registry.token(request),
                "request_id": request.request_id,
                "enqueued_ns": record.enqueued_ns,
                "sequence": record.sequence,
            }
            for request, record in guard.records.items()
        ],
        "next_sequence": guard.next_sequence,
        "last_ns": guard.last_ns,
        "threshold_ns": guard.threshold_ns,
        "fault_code": guard.fault_code,
        "closed": guard.closed,
    }


def snapshot(scheduler, requests, generations, registry, full=False):
    state = native_snapshot(scheduler, requests, generations, full)
    state["guard"] = guard_snapshot(scheduler, registry)
    state["object_bindings"] = {
        rid: registry.token(request) for rid, request in scheduler.requests.items()
    }
    state["queue_objects"] = {
        "waiting": [registry.token(request) for request in scheduler.waiting],
        "running": [registry.token(request) for request in scheduler.running],
        "holding": [registry.token(request) for request in scheduler.kv_holding_waiting],
        "deferred": sorted(registry.token(request) for request in scheduler.deferred_waiting),
    }
    return state
