"""Independent M2B audit from native objects, clocks and lifecycle events.

This module never imports the production scheduler or waiting-protection module.
The accepted M2A independent read-only checks remain reusable historical oracles;
their frozen matrix and guarded rejection entry point are not changed.
"""

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import m2a_trace_check as legacy  # noqa: E402

from vllm_prefix_scheduler.m2b_contract import (  # noqa: E402
    canonical,
    expected_cases,
    require,
    require_assertions,
)

rows = legacy.rows
one = legacy.one


def threshold_oracle(milliseconds):
    """Fraction implements the oracle independently of production integer conversion."""
    require(type(milliseconds) in (int, float), "wait must be an exact int/float")
    require(milliseconds >= 0, "negative wait")
    if type(milliseconds) is float:
        require(math.isfinite(milliseconds), "nonfinite wait")
    duration = Fraction(milliseconds) * 1_000_000
    return math.ceil(duration)


def audit_calls(events):
    """Every observed native call has exactly one properly nested return OR raise."""
    stack, pairs = [], {}
    for index, event in enumerate(events):
        kind = event["event"]
        if kind == "call":
            call = event.get("call")
            require(type(call) is int and call not in pairs, "duplicate/invalid native call")
            pairs[call] = [index, event]
            stack.append(call)
        elif kind in {"return", "raise"}:
            call = event.get("call")
            require(stack and stack.pop() == call, "unbalanced native call/return|raise")
            begin = pairs[call][1]
            require(
                event.get("method") == begin.get("method")
                and event.get("owner") == begin.get("owner"),
                "native completion identity mismatch",
            )
            if kind == "raise":
                require(type(event.get("error_type")) is str, "native raise lacks error type")
            pairs[call].extend([index, event])
    require(not stack and all(len(p) == 4 for p in pairs.values()), "unfinished native call")
    return pairs


def _raw_requests(state):
    require(
        type(state) is dict and type(state.get("full")) is dict, "raw full native state missing"
    )
    return state["full"]["requests"]


def _fields_for(state, request_id, generation):
    fields = _raw_requests(state)
    key = request_id if generation == 0 else f"{request_id}@{generation}"
    require(key in fields, "actual Request fields missing from raw snapshot")
    return fields[key]


def fresh_oracle(fields, request_id, state, *, actual_class="vllm.v1.request.Request"):
    """Recompute M2A's audited ordinary-fresh predicate from stored native fields."""
    required = {
        "status",
        "num_computed_tokens",
        "num_preemptions",
        "num_output_placeholders",
        "num_in_flight_tokens",
        "num_stale_output_tokens",
        "next_decode_eligible_step",
        "last_sched_seq",
        "shared_prefix_boundary",
        "replay_start",
        "_output_token_ids",
        "is_prefill_chunk",
        "drop_stale_output",
        "spec_token_ids",
        "resumable",
        "streaming_queue",
        "session_id",
        "prompt_token_ids",
        "sampling_params",
        "pooling_params",
        "prompt_embeds",
        "prompt_is_token_ids",
        "lora_request",
        "structured_output_request",
        "kv_transfer_params",
        "ec_transfer_params",
        "kv_hints",
        "mm_features",
        "abort_immediately",
        "stop_reason",
    }
    require(type(fields) is dict and required <= set(fields), "raw freshness fields missing")
    require(type(actual_class) is str and actual_class, "actual native Request class missing")
    if actual_class != "vllm.v1.request.Request":
        return False
    status = fields.get("status")
    if not isinstance(status, dict) or status.get("name") != "WAITING":
        return False
    if request_id in state["deferred_waiting"]:
        return False
    inflight = state["full"]["scheduler_lifecycle"].get("_inflight_prefills", {"set": []})
    if {"request": request_id} in inflight.get("set", []):
        return False
    zero_fields = (
        "num_computed_tokens",
        "num_preemptions",
        "num_output_placeholders",
        "num_in_flight_tokens",
        "num_stale_output_tokens",
        "next_decode_eligible_step",
        "last_sched_seq",
        "shared_prefix_boundary",
        "replay_start",
    )
    if any(fields.get(name) for name in zero_fields):
        return False
    if fields.get("_output_token_ids"):
        return False
    if any(
        fields.get(name)
        for name in (
            "is_prefill_chunk",
            "drop_stale_output",
            "spec_token_ids",
            "resumable",
        )
    ):
        return False
    if fields.get("streaming_queue") is not None or fields.get("session_id") is not None:
        return False
    if not fields.get("prompt_token_ids") or fields.get("sampling_params") is None:
        return False
    if any(
        fields.get(name) is not None
        for name in (
            "pooling_params",
            "prompt_embeds",
            "prompt_is_token_ids",
            "lora_request",
            "structured_output_request",
            "kv_transfer_params",
            "ec_transfer_params",
            "kv_hints",
        )
    ):
        return False
    return (
        not fields.get("mm_features")
        and not fields.get("abort_immediately")
        and fields.get("stop_reason") is None
    )


def _record_map(snapshot):
    records = snapshot.get("records")
    require(type(records) is list, "guard records must be a list")
    output = {}
    for record in records:
        require(type(record) is dict, "guard record must be an object")
        token = record.get("object")
        require(
            type(token) is str and token and token not in output, "duplicate/invalid object token"
        )
        require(
            type(record.get("enqueued_ns")) is int and type(record.get("sequence")) is int,
            "guard time/sequence must be strict integers",
        )
        require(record["sequence"] >= 0, "negative enqueue sequence")
        output[token] = {k: record[k] for k in ("request_id", "enqueued_ns", "sequence")}
    return output


class GuardOracle:
    """Replay required state from successful native admission and raw clock reads."""

    def __init__(self, metadata, config, events):
        self.metadata, self.config = metadata, config
        self.enabled = metadata["mode"] == "guarded"
        self.records, self.objects, self.generations = {}, {}, {}
        self.pending_adds, self.native_adds = {}, []
        self.sequence, self.read_index, self.last_ns = 0, 0, None
        self.last_clock, self.clock_pending = None, None
        self.fault, self.closed = None, False
        self.guard_stack, self.native_stack = [], []
        self.successful_dispatch, self.bindings = {}, {}
        self.decisions, self.active_decision = [], None
        self.native_failures, self.clock_failures = [], []
        self.shutdown_open = False
        self.shutdown_completed = 0
        self.cleanups = defaultdict(int)
        self.maximum_records = 0
        identity = rows(events, "native_objects")
        wait = config["max_wait_ms"]
        if metadata["scenario"] == "clock_thresholds":
            wait = config["clock_plans"][metadata["variant"]]["max_wait_ms"]
        if identity and metadata["mode"] != "native":
            require(
                identity[0]["effective_config"]["max_wait_ms"] == wait, "effective wait changed"
            )
        self.threshold = threshold_oracle(wait)

    def check_snapshot(self, event):
        snapshot = event.get("guard")
        require(type(snapshot) is dict, "event missing independent guard storage snapshot")
        require(snapshot.get("present") is self.enabled, "guard state present outside guarded mode")
        require(
            _record_map(snapshot) == self.records, "live records differ from native/clock oracle"
        )
        if self.enabled:
            require(snapshot.get("next_sequence") == self.sequence, "enqueue sequence drift/reset")
            require(snapshot.get("last_ns") == self.last_ns, "last clock differs from raw reads")
            require(
                snapshot.get("threshold_ns") == self.threshold,
                "wait conversion differs from Fraction oracle",
            )
            require(
                snapshot.get("fault_code") == self.fault, "fault state differs from causal events"
            )
            require(snapshot.get("closed") is self.closed, "shutdown closed state differs")
        self.maximum_records = max(self.maximum_records, len(self.records))

    def _object(self, token):
        require(token in self.objects, "object has no successful native binding")
        return self.objects[token]

    def _native_request(self, event, token):
        rid, generation = self._object(token)
        state = event.get("state", {})
        key = rid if generation == 0 else f"{rid}@{generation}"
        return state.get("requests", {}).get(key)

    def _clock(self, event):
        require(
            self.enabled and not self.closed and self.fault is None,
            "protection clock read when inactive",
        )
        require(event.get("purpose") in {"enqueue", "decision"}, "unknown clock purpose")
        self.read_index += 1
        require(
            type(event.get("read_index")) is int and event["read_index"] == self.read_index,
            "clock read dropped/duplicated",
        )
        require(self.clock_pending is None, "overlapping clock reads")
        self.clock_pending = event
        value = event.get("value")
        error = "clock_provider" if event.get("error") is not None else None
        if error is None and type(value) is not int:
            error = "clock_type"
        if error is None and self.last_ns is not None and value < self.last_ns:
            error = "clock_backwards"
        self.clock_pending = {**event, "expected_fault": error}
        if error:
            self.clock_failures.append(error)
        if event["purpose"] == "decision":
            require(self.active_decision is not None, "decision clock outside actual sort")
            self.active_decision["clocks"].append(self.clock_pending)

    def _guard_after(self, event, begin):
        operation = event["operation"]
        if operation == "read_clock":
            require(self.clock_pending is not None, "read_clock has no raw provider observation")
            require(self.clock_pending["expected_fault"] is None, "invalid clock was accepted")
            self.last_ns = self.clock_pending["value"]
            self.last_clock = self.clock_pending
            self.clock_pending = None
        elif operation == "register":
            token = event["object"]
            require(
                token in self.pending_adds,
                "registration did not follow new successful native admission",
            )
            require(token not in self.records, "duplicate admission refreshed old age")
            require(
                self.last_clock is not None and self.last_clock["purpose"] == "enqueue",
                "registration lacks enqueue clock",
            )
            require(
                self.last_clock["read_index"] > self.pending_adds[token]["read_index"],
                "registration reused an old clock reading",
            )
            rid, _ = self._object(token)
            self.records[token] = {
                "request_id": rid,
                "enqueued_ns": self.last_clock["value"],
                "sequence": self.sequence,
            }
            self.sequence += 1
            del self.pending_adds[token]
        elif operation == "discard":
            token, reason = event["object"], event.get("reason")
            if reason == "first_dispatch":
                require(
                    token in self.successful_dispatch.get(event["step"], set()),
                    "record cleared before actual successful positive dispatch",
                )
            elif reason in {"terminal", "preempted"}:
                method = "_free_request" if reason == "terminal" else "_preempt_request"
                require(
                    any(
                        e["method"] == method and e.get("details", {}).get("object") == token
                        for e in self.native_stack
                    ),
                    "cleanup has no actual object lifecycle call",
                )
                if reason == "terminal":
                    call = next(
                        e
                        for e in reversed(self.native_stack)
                        if e["method"] == method and e.get("details", {}).get("object") == token
                    )
                    actual = self._native_request(call, token)
                    require(
                        actual is not None and actual["status"].startswith("FINISHED"),
                        "nonterminal object lost its record",
                    )
            elif reason == "identity_probe":
                require(
                    self.metadata["scenario"] == "id_reuse" and token not in self.records,
                    "identity probe removed a live binding",
                )
            else:
                raise ValueError(f"M2B: unrecognized cleanup reason {reason}")
            removed = token in self.records
            require(
                event.get("result") is removed, "cleanup return disagrees with actual live state"
            )
            self.records.pop(token, None)
            self.cleanups[reason] += int(removed)
        elif operation == "fail":
            code = event.get("code")
            require(
                type(code) is str
                and code
                in {
                    "clock_provider",
                    "clock_type",
                    "clock_backwards",
                    "negative_elapsed",
                    "missing_record",
                    "native_add",
                    "native_schedule",
                    "native_preempt",
                    "native_free",
                    "admission_identity",
                },
                "unknown fault code",
            )
            if code.startswith("clock_"):
                require(
                    self.clock_pending is not None and self.clock_pending["expected_fault"] == code,
                    "clock fault has no raw failing input",
                )
            if self.fault is None:
                self.fault = code
        elif operation == "close":
            require(
                self.shutdown_completed > 0 and not self.shutdown_open,
                "guard close did not follow actual native shutdown completion/failure",
            )
            self.records.clear()
            self.pending_adds.clear()
            self.closed = True
        else:
            raise ValueError(f"M2B: unknown guard operation {operation}")

    def _decision_after(self, event):
        decision = self.active_decision
        require(decision is not None, "sort return has no matching begin")
        objects = decision["objects"]
        require(
            len(objects) >= 2 and len(set(objects)) == len(objects),
            "sort outside real multi-candidate window",
        )
        require(set(event["objects"]) == set(objects), "decision changed candidate membership")
        protected = set()
        if self.enabled:
            require(len(decision["clocks"]) == 1, "guarded sort did not use exactly one shared now")
            clock = decision["clocks"][0]
            require(clock["expected_fault"] is None, "sort accepted invalid time")
            now = clock["value"]
            for token in objects:
                require(
                    token in decision["records"], "sort candidate lacks original admission record"
                )
                elapsed = now - decision["records"][token]["enqueued_ns"]
                require(elapsed >= 0, "negative waiting elapsed")
                if elapsed >= self.threshold:
                    protected.add(token)
        else:
            require(not decision["clocks"], "prefix sort read protection clock")
        scores = decision["scores"]
        for token in objects:
            if token not in protected:
                require(token in scores, "unprotected candidate has no current prefix observation")

        def key(token):
            if token in protected:
                return (0, decision["records"][token]["sequence"])
            return (1, -scores[token])

        ordered = sorted(objects, key=key)
        require(
            event["objects"] == ordered,
            "actual decision differs from independent protection/FIFO/prefix order",
        )
        decision.update(
            after=ordered,
            protected=sorted(protected),
            now=decision["clocks"][0]["value"] if self.enabled else None,
        )
        self.decisions.append(decision)
        self.active_decision = None

    def feed(self, event):
        kind = event["event"]
        if kind == "binding":
            token, rid, generation = (
                event.get("object"),
                event.get("request_id"),
                event.get("generation"),
            )
            require(
                type(token) is str and token and token not in self.objects,
                "object identity rebound",
            )
            require(
                type(generation) is int and generation == self.generations.get(rid, -1) + 1,
                "native generation not independently increasing",
            )
            self.objects[token] = (rid, generation)
            self.generations[rid] = generation
            self.bindings[rid] = token
        elif kind == "native_add_call":
            require(
                not self.closed and (not self.enabled or self.fault is None),
                "native add entered after guard fault/close",
            )
            self.native_adds.append(event)
        elif kind in {"native_add_return", "native_add_raise"}:
            require(self.native_adds, "native add completion without call")
            before = self.native_adds.pop()
            require(event["request_id"] == before["request_id"], "native add identity changed")
            if kind == "native_add_return":
                token = event.get("actual_object")
                if token is not None:
                    rid, generation = self._object(token)
                    fields = _fields_for(event["state"], rid, generation)
                    eligible = fresh_oracle(
                        fields, rid, event["state"], actual_class=event.get("actual_class")
                    )
                    if self.enabled and before["previous_object"] is None and eligible:
                        require(
                            token == before["object"], "native accepted an unexpected actual object"
                        )
                        self.pending_adds[token] = {"read_index": self.read_index}
            else:
                self.native_failures.append("native_add")
        elif kind == "enqueue":
            require(
                event.get("actual_object") not in self.pending_adds,
                "successful ordinary admission has no protection record",
            )
        elif kind == "enqueue_raise":
            token = event.get("object")
            if token in self.pending_adds:
                require(self.fault is not None, "post-admission exception did not fail closed")
                del self.pending_adds[token]
        elif kind == "clock":
            self._clock(event)
        elif kind == "guard_before":
            require(self.enabled, "non-guarded mode accessed guard")
            self.guard_stack.append(event)
        elif kind in {"guard_after", "guard_raise"}:
            require(self.guard_stack, "guard completion without call")
            begin = self.guard_stack.pop()
            require(
                all(
                    begin.get(k) == event.get(k) for k in ("operation", "object", "reason", "code")
                ),
                "guard call/completion mismatch",
            )
            if kind == "guard_after":
                self._guard_after(event, begin)
            elif event["operation"] == "read_clock":
                require(
                    self.clock_pending is not None
                    and self.clock_pending["expected_fault"] is not None,
                    "clock raised without invalid raw input",
                )
                self.clock_pending = None
        elif kind == "call":
            self.native_stack.append(event)
        elif kind in {"return", "raise"}:
            require(
                self.native_stack and self.native_stack[-1]["call"] == event["call"],
                "native lifecycle nesting changed",
            )
            begin = self.native_stack.pop()
            token = begin.get("details", {}).get("object")
            if begin["method"] in {"_free_request", "_preempt_request"} and self.enabled:
                actual = self._native_request(event, token)
                if actual and (
                    actual["status"].startswith("FINISHED") or actual["status"] == "PREEMPTED"
                ):
                    require(
                        token not in self.records, "terminal/preempted object retained protection"
                    )
                if begin["method"] == "_preempt_request" and actual:
                    prior = self._native_request(begin, token)
                    transitioned = (
                        prior["status"] != "PREEMPTED" and actual["status"] == "PREEMPTED"
                    ) or actual["preemptions"] > prior["preemptions"]
                    old_record = _record_map(begin["guard"]).get(token)
                    if not transitioned and old_record is not None:
                        require(
                            self.records.get(token) == old_record,
                            "failed preemption removed age before actual native transition",
                        )
            if kind == "raise":
                self.native_failures.append(begin["method"])
        elif kind == "native_schedule_enter":
            require(
                not self.closed and (not self.enabled or self.fault is None),
                "native schedule entered after fault/close",
            )
        elif kind == "native_schedule_return":
            require(
                event["step"] not in self.successful_dispatch, "duplicate native schedule return"
            )
            scheduled, objects = event["scheduled"], event["objects"]
            require(set(objects) == set(scheduled), "dispatch object bindings incomplete")
            self.successful_dispatch[event["step"]] = {
                objects[rid] for rid, count in scheduled.items() if count > 0
            }
            for rid, token in objects.items():
                require(self._object(token)[0] == rid, "dispatch routed to wrong object")
                require(
                    event["state"]["object_bindings"].get(rid) == token,
                    "dispatch object is not current native binding",
                )
        elif kind == "native_schedule_raise":
            self.native_failures.append("native_schedule")
        elif kind == "native_shutdown_call":
            require(not self.shutdown_open, "nested native shutdown")
            self.shutdown_open = True
        elif kind in {"native_shutdown_return", "native_shutdown_raise"}:
            require(self.shutdown_open, "native shutdown completion without call")
            self.shutdown_open = False
            self.shutdown_completed += 1
        elif kind == "schedule_after":
            require(
                event["step"] in self.successful_dispatch,
                "outer schedule lacks successful native return",
            )
            require(
                not (self.successful_dispatch[event["step"]] & set(self.records)),
                "positive dispatch retained protection",
            )
        elif kind == "decision_before":
            require(self.active_decision is None, "nested/repeated unfinished decision")
            self.active_decision = {
                "step": event["step"],
                "objects": event["objects"],
                "records": dict(self.records),
                "scores": {},
                "clocks": [],
            }
        elif kind == "score" and self.active_decision is not None:
            rid = event["request_id"]
            token = self.bindings[rid]
            require(token not in self.active_decision["scores"], "candidate scored twice")
            require(type(event.get("score")) is int and event["score"] >= 0, "invalid prefix score")
            self.active_decision["scores"][token] = event["score"]
        elif kind == "decision_after":
            self._decision_after(event)
        elif kind in {"decision_raise", "schedule_raise", "expected_error"}:
            # An explicitly raised sort is not a completed decision. Its queue
            # restoration and fault semantics are audited by its named fixture.
            self.active_decision = None
        elif kind == "check" and event.get("error"):
            require(
                self.metadata["scenario"] == "window",
                "unexpected recoverable transaction exception",
            )
            self.active_decision = None
        elif kind == "fixture_remove_record":
            require(
                self.metadata["scenario"] == "window"
                and self.metadata["variant"] in {"missing_singleton", "missing_window_one"},
                "unexpected metadata corruption injection",
            )
            require(event["object"] in self.records, "corruption did not remove a real record")
            del self.records[event["object"]]
        self.check_snapshot(event)

    def finish(self):
        require(
            not self.guard_stack and not self.native_stack and not self.native_adds,
            "unfinished observed lifecycle call",
        )
        require(
            self.active_decision is None and self.clock_pending is None, "unfinished clock/decision"
        )
        require(not self.shutdown_open, "native shutdown did not complete or raise")
        require(
            not self.records and not self.pending_adds, "historical protection records retained"
        )
        if self.enabled:
            require(self.closed, "guarded scheduler was not closed")
        else:
            require(
                self.read_index == self.sequence == 0,
                "off/prefix accessed protection clock/records",
            )
        return {
            "clock_reads": self.read_index,
            "registered_objects": self.sequence,
            "maximum_live_records": self.maximum_records,
            "final_records": 0,
            "cleanups": dict(self.cleanups),
            "fault_code": self.fault,
            "decisions": [
                {k: d[k] for k in ("step", "objects", "after", "protected", "now")}
                for d in self.decisions
            ],
        }


def audit_guard(events, metadata, config):
    oracle = GuardOracle(metadata, config, events)
    for event in events:
        if event["event"] not in {"case_begin", "case_end"}:
            oracle.feed(event)
    return oracle.finish(), oracle


def audit_identity(events, metadata, config):
    native = one(events, "native_objects")
    expected = legacy.NATIVE_CLASS if metadata["mode"] == "native" else legacy.CUSTOM_CLASS
    require(
        native["class"] == expected and native["mro"][0] == expected, "wrong real scheduler class"
    )
    require(legacy.NATIVE_CLASS in native["mro"], "AsyncScheduler inheritance missing")
    if expected != legacy.NATIVE_CLASS:
        require(
            native["mro"][1] == legacy.NATIVE_CLASS,
            "extension no longer directly inherits AsyncScheduler",
        )
    for name in ("async_scheduling", "use_v2_model_runner", "log_stats", "kv_events", "kv_metrics"):
        require(native[name] is True, f"native required flag disabled: {name}")
    require(native["apc"] is (metadata["scenario"] != "apc_off"), "effective APC drift")
    require(
        native["block_size"] == 16 and native["publisher"] == "NullEventPublisher",
        "native cache/publisher drift",
    )
    require(native["max_concurrent_batches"] == 2, "native async concurrency changed")
    scenario, variant = metadata["scenario"], metadata["variant"]
    capacity = (
        5
        if scenario
        in {"bc_pressure", "preemption", "score_eviction", "allocation_retry", "hot_flow"}
        else 10
    )
    budget = 16 if scenario == "chunk" else 64
    active = (
        1
        if scenario in {"chunk", "hot_flow", "clock_thresholds", "protected_order", "window_age"}
        else 2
    )
    require(
        native["capacity"] == capacity
        and native["effective_config"]["budget"] == budget
        and native["effective_config"]["active"] == active,
        "effective native fixture geometry differs",
    )
    expected_parallel = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
    }
    actual_parallel = native["effective_config"].get("parallel")
    require(
        type(actual_parallel) is dict
        and set(actual_parallel) == set(expected_parallel)
        and all(type(value) is int and value == 1 for value in actual_parallel.values()),
        "effective native parallel semantics changed",
    )
    if metadata["mode"] != "native":
        window = (
            1
            if scenario == "window" and variant in {"window_one", "missing_window_one"}
            else 2
            if scenario == "window_age" or (scenario == "window" and variant == "window_limit")
            else 16
        )
        require(
            native["effective_config"]["window"] == window, "effective candidate window changed"
        )
    for cls, relative in (
        (legacy.NATIVE_CLASS, "vllm/v1/core/sched/async_scheduler.py"),
        ("vllm.v1.core.sched.scheduler.Scheduler", "vllm/v1/core/sched/scheduler.py"),
    ):
        require(
            native["sources"][cls]["sha256"] == config["source_sha256"][relative],
            "unverified native class source",
        )
    fields = set(native["instance_fields"])
    require(
        ("_wait_guard" in fields) is (metadata["mode"] == "guarded"),
        "persistent protection state in wrong mode",
    )
    return native


def audit_dispatch(events):
    counts = legacy.audit_dispatch(events)
    returned = {e["step"]: e for e in rows(events, "native_schedule_return")}
    dispatched = {e["step"]: e for e in rows(events, "dispatch")}
    after = {e["step"]: e for e in rows(events, "schedule_after")}
    require(
        set(returned) == set(dispatched) == set(after),
        "actual native return/dispatch/outer output steps differ",
    )
    for step, event in returned.items():
        require(
            event["scheduled"] == after[step]["scheduled"],
            "native dispatch token map changed after return",
        )
        require(
            events.index(event) < events.index(dispatched[step]) < events.index(after[step]),
            "dispatch was recorded before native successful return",
        )
        require(
            {i["request_id"]: i["object"] for i in dispatched[step]["intervals"]}
            == event["objects"],
            "dispatch object generation mapping changed",
        )
    return counts


def audit_bounds(events, metadata, oracle):
    native = one(events, "native_objects")
    window = native["effective_config"]["window"]
    if metadata["mode"] == "native":
        require(not rows(events, "window_before"), "native performed extension reordering")
        return
    expected_windows, actual_decisions = {}, {}
    for decision in oracle.decisions:
        actual_decisions[decision["step"]] = decision
    for event in rows(events, "window_before"):
        state = event["state"]
        require(type(window) is int and window > 0, "invalid actual window")
        require(event["step"] not in expected_windows, "multiple window passes in one step")
        require(
            event["waiting"] == state["waiting"], "raw window queue disagrees with native state"
        )
        tokens = state["queue_objects"]["waiting"]
        require(len(tokens) == len(event["waiting"]), "window object list length differs")
        candidates = []
        for rid, token in list(zip(event["waiting"], tokens, strict=True))[:window]:
            require(
                state["object_bindings"].get(rid) == token, "candidate is not actual native object"
            )
            generation = oracle._object(token)[1]
            fields = _fields_for(state, rid, generation)
            if not fresh_oracle(fields, rid, state):
                break
            candidates.append(token)
        expected_windows[event["step"]] = (event, candidates)
    for event in rows(events, "candidates"):
        require(event["step"] in expected_windows, "candidate observation outside window")
        _, candidates = expected_windows[event["step"]]
        require(
            event["ids"] == [oracle._object(t)[0] for t in candidates],
            "candidate window crossed barrier or omitted eligible head",
        )
    for event in rows(events, "window_after"):
        require(event["step"] in expected_windows, "window return has no before")
        before, candidates = expected_windows[event["step"]]
        require(event["before"] == before["waiting"], "window before identity changed")
        if len(candidates) >= 2:
            require(
                event["step"] in actual_decisions, "multi-candidate window has no completed sort"
            )
            decision = actual_decisions[event["step"]]
            require(
                decision["objects"] == candidates, "sort received wrong contiguous candidate window"
            )
            head = [oracle._object(t)[0] for t in decision["after"]]
            expected = head + before["waiting"][len(candidates) :]
        else:
            require(event["step"] not in actual_decisions, "unnecessary single-candidate sort")
            expected = before["waiting"]
        require(
            event["after"] == expected and event["state"]["waiting"] == expected,
            "window suffix/order changed outside policy",
        )
        for key in ("running", "kv_holding_waiting", "deferred_waiting"):
            require(
                before["state"][key] == event["state"][key], "window altered native priority queues"
            )
        counts = event["counts"]
        require(
            counts["checks"] <= min(window, len(before["waiting"])), "window scan exceeded bound"
        )
    for event in rows(events, "schedule_after"):
        counts = event["counts"]
        require(
            counts["super"] == 1 and counts["sorts"] <= 1 and counts["window_calls"] <= 1,
            "scheduler repeated parent/window/sort",
        )
        if metadata["mode"] == "off" or (metadata["mode"] == "prefix" and window == 1):
            require(
                all(value == 0 for name, value in counts.items() if name != "super"),
                "off/prefix-window1 left native fast path",
            )


def audit_probe_leaf(events, variant):
    before, after = one(events, "probe_before"), one(events, "probe_after")
    expected = {
        "cold": 0,
        "partial": 16,
        "full": 32,
        "exact16": 0,
        "exact32": 16,
        "one": 0,
        "skip_read": 0,
    }
    require(before["probe_id"] == after["probe_id"] == variant, "probe leaf identity mismatch")
    require(
        before["state"] == after["state"] and after["score"] == expected[variant],
        "read-only probe mutated state or returned wrong score",
    )
    full = before["state"]["full"]
    for key in (
        "graph",
        "identities",
        "blocks",
        "hash_mapping_ordered",
        "reverse_hash_mapping",
        "reuse_watchers",
        "single_managers",
        "queue_endpoints",
        "queue_count",
        "pool_flags",
        "stats",
        "events",
        "metrics",
        "manager_fields",
        "coordinator_fields",
        "scheduler_lifecycle",
        "random_state",
    ):
        require(key in full, "probe native snapshot is incomplete")
    require(
        full["metrics"] is not None and full["stats"] is not None and full["identities"],
        "metrics/stats/identity observation missing",
    )
    require(
        full["pool_flags"]["enable_kv_cache_events"] is True
        and full["requests"][variant]["kv_cache_report_mode"] == "full",
        "full event reporting not enabled",
    )
    checks = {e["name"]: e for e in rows(events, "check")}
    require(
        set(checks) == {"sensitivity_event", "sensitivity_touch", "sensitivity_stats"},
        "probe sensitivity controls incomplete",
    )
    for name, field in (("sensitivity_event", "events"), ("sensitivity_stats", "stats")):
        require(
            checks[name]["before"]["full"][field] != checks[name]["after"]["full"][field],
            "probe observation cannot detect native effects",
        )
    require(
        checks["sensitivity_touch"]["before"]["free_queue"]
        != checks["sensitivity_touch"]["after"]["free_queue"],
        "probe cannot detect block touch",
    )
    start, finish = one(events, "post_probe_begin"), one(events, "post_probe_end")
    first, last = events.index(start), events.index(finish)
    require(
        first < last and all(events.index(c) > last for c in checks.values()),
        "native reporting mixed with sensitivity controls",
    )
    phase = events[first + 1 : last]
    require(
        start["state"]["full"]["events"] == []
        and rows(phase, "published")
        and rows(phase, "output_stats"),
        "normal reporting absent or contaminated",
    )
    stored = [
        obj
        for e in rows(phase, "published")
        for obj in legacy.typed_values(e["batch"], "vllm.distributed.kv_events.BlockStored")
    ]
    require(
        any(
            obj["fields"]["block_size"] == 16
            and obj["fields"]["token_ids"] == list(range(100, 132))
            for obj in stored
        ),
        "native reused-prefix events missing",
    )
    stats = [
        obj["fields"]
        for e in rows(phase, "output_stats")
        for obj in legacy.typed_values(e["outputs"], "vllm.v1.metrics.stats.PrefixCacheStats")
    ]
    require(
        [sum(s[k] for s in stats) for k in ("requests", "queries", "hits")] == [1, 33, 32],
        "native prefix stats changed",
    )
    prompt = sum(
        i["prefill_tokens"]
        for e in rows(phase, "dispatch")
        for i in e["intervals"]
        if i["request_id"] == "full"
    )
    require(
        prompt == 1 and rows(phase, "drained"),
        "normal probe reuse did not actually dispatch and drain",
    )


def audit_resources(state):
    legacy.audit_released(state, historical_cancellation=True)
    lifecycle = state["full"]["scheduler_lifecycle"]
    require(
        lifecycle.get("_inflight_prefills") == {"set": []},
        "native in-flight Request references remain",
    )
    require(lifecycle.get("deferred_frees") == [], "native deferred KV resources remain")
    require(
        all(not group for group in state["request_blocks"]),
        "native request block mappings retained historical IDs",
    )
    return {
        "active_requests": 0,
        "live_request_block_mappings": 0,
        "deferred_frees": 0,
        "remaining_terminal_notifications": lifecycle.get("finished_req_ids"),
        "remaining_preempt_notifications": lifecycle.get("reset_preempted_req_ids"),
        "note": "Notification IDs are reported separately from live objects, outputs, references and KV resources.",
    }


def audit_rejected_retries(events, *, schedule_phase, request_id, state):
    """Require actual attempts and their exceptions, not merely absence of work."""
    require(state in {"fault", "closed"}, "unknown rejected retry state")
    schedules = [
        (index, event)
        for index, event in enumerate(events)
        if event["event"] == "schedule_before" and event.get("phase") == schedule_phase
    ]
    adds = [
        (index, event)
        for index, event in enumerate(events)
        if event["event"] == "enqueue_call" and event.get("request_id") == request_id
    ]
    require(schedules, "guard retry schedule attempt missing")
    require(adds, "guard retry add attempt missing")
    require(len(schedules) == 1, "guard retry schedule attempt duplicated")
    require(len(adds) == 1, "guard retry add attempt duplicated")
    for index, call in schedules:
        guard = call.get("guard", {})
        require(
            guard.get("closed") is True
            if state == "closed"
            else guard.get("fault_code") is not None,
            "retry schedule did not occur after fault/close",
        )
        completed = [
            (j, event)
            for j, event in enumerate(events)
            if event["event"] == "schedule_raise"
            and event.get("step") == call["step"]
            and event.get("phase") == schedule_phase
        ]
        require(
            len(completed) == 1
            and completed[0][0] > index
            and completed[0][1].get("error_type") == "WaitGuardError",
            "guard retry schedule did not raise WaitGuardError",
        )
        require(
            not any(
                event["event"] in {"clock", "native_schedule_enter", "schedule_after"}
                for event in events[index : completed[0][0] + 1]
            ),
            "rejected schedule entered protected/native work",
        )
    for index, call in adds:
        guard = call.get("guard", {})
        require(
            guard.get("closed") is True
            if state == "closed"
            else guard.get("fault_code") is not None,
            "retry add did not occur after fault/close",
        )
        completed = [
            (j, event)
            for j, event in enumerate(events)
            if event["event"] == "enqueue_raise"
            and event.get("request_id") == request_id
            and event.get("object") == call["object"]
            and event.get("step") == call["step"]
        ]
        require(
            len(completed) == 1
            and completed[0][0] > index
            and completed[0][1].get("error_type") == "WaitGuardError",
            "guard retry add did not raise WaitGuardError",
        )
        require(
            not any(
                event["event"] in {"clock", "native_add_call", "enqueue"}
                for event in events[index : completed[0][0] + 1]
            ),
            "rejected add entered protected/native work",
        )
    return {"schedule_attempts": len(schedules), "add_attempts": len(adds)}


def audit_reuse_boundary(event):
    """A new ID generation requires drained ownership AND native notifications."""
    require(event.get("pending_outputs") == 0, "ID reuse retains pending native outputs")
    require(not event["guard"].get("records"), "ID reuse retains protection records")
    audit_resources(event["state"])
    lifecycle = event["state"]["full"]["scheduler_lifecycle"]
    require(
        lifecycle.get("finished_req_ids") == {"set": []}
        and lifecycle.get("reset_preempted_req_ids") == {"set": []},
        "ID reuse retains native terminal/preemption notifications",
    )


def audit_fault_cause(events, metadata, oracle):
    variant = metadata["variant"]
    faults = {
        "fault_bool": "clock_type",
        "fault_float": "clock_type",
        "fault_string": "clock_type",
        "fault_backward": "clock_backwards",
        "fault_provider_add": "clock_provider",
        "fault_provider_decision": "clock_provider",
        "missing_singleton": "missing_record",
        "missing_window_one": "missing_record",
        "add_failure_prebind": "native_add",
        "add_failure_postbind": "native_add",
        "free_error": "native_free",
        "batch_free_error": "native_free",
        "schedule_before_update_error": "native_schedule",
        "schedule_after_update_error": "native_schedule",
        "preempt_free_error": "native_preempt",
        "preempt_after_transition_error": "native_preempt",
        "fault_weakref_release": "clock_provider",
    }
    expected = faults.get(variant)
    require(
        oracle.fault == expected, "observed fault does not match independently required scenario"
    )
    if expected is None:
        return
    if expected.startswith("clock_"):
        require(expected in oracle.clock_failures, "clock fault lacks a raw invalid clock input")
    elif expected == "native_add":
        native_failure = rows(events, "native_add_raise")
        postbind_failure = [
            e
            for e in rows(events, "fault_injection")
            if e.get("site") == "native_add_returned_with_actual_binding"
        ]
        require(
            native_failure or postbind_failure, "add fault lacks actual native admission exception"
        )
        require(rows(events, "enqueue_raise"), "failed native add was reported as successful")
    elif expected == "native_schedule":
        require(
            rows(events, "native_schedule_raise") and rows(events, "schedule_raise"),
            "schedule fault lacks actual failed parent return",
        )
    elif expected in {"native_preempt", "native_free"}:
        method = "_preempt_request" if expected == "native_preempt" else "_free_request"
        require(
            any(e["method"] == method for e in rows(events, "raise")),
            "lifecycle fault has no corresponding native exception",
        )
    elif expected == "missing_record":
        require(
            rows(events, "fixture_remove_record"),
            "missing-record fixture never removed its original object record",
        )
    first_fault = next(
        i for i, e in enumerate(events) if e.get("guard", {}).get("fault_code") is not None
    )
    later = events[first_fault:]
    require(
        not rows(later, "clock")
        and not rows(later, "native_add_call")
        and not rows(later, "native_schedule_enter"),
        "latched retries entered clock or native work",
    )
    if metadata["scenario"] == "clock_thresholds" and variant.startswith("fault_"):
        return audit_rejected_retries(
            events,
            schedule_phase="fault_retry_schedule",
            request_id="blocked_after_fault",
            state="fault",
        )
    return audit_rejected_retries(
        events, schedule_phase=f"{variant}_retry", request_id="rejected_after_fault", state="fault"
    )


def audit_exception_fixture(events, metadata, oracle):
    variant = metadata["variant"]
    if variant not in {
        "add_failure_prebind",
        "add_failure_postbind",
        "generator_raise",
        "batch_free_error",
        "schedule_before_update_error",
        "schedule_after_update_error",
        "preempt_free_error",
        "preempt_after_transition_error",
        "fault_weakref_release",
    }:
        return
    check = one(events, "fault_checkpoint")
    require(
        check["variant"] == variant and check["pending_outputs"] == 0,
        "fault checkpoint has pending output or wrong identity",
    )
    if variant.startswith("schedule_") or variant.startswith("preempt_"):
        require(
            not rows(events, "native_schedule_return") and not rows(events, "dispatch"),
            "unreturned partial native step counted as dispatched",
        )
        req = next(iter(check["state"]["requests"].values()))
        live = _record_map(check["guard"])
        if variant == "schedule_before_update_error":
            require(
                req["in_flight"] == 0 and len(live) == 1 and not rows(events, "advance"),
                "before-update fault occurred after native update or lost age",
            )
        elif variant == "schedule_after_update_error":
            require(
                req["in_flight"] > 0 and len(live) == 1 and rows(events, "advance"),
                "after-update exception did not retain actual pending protection",
            )
        elif variant == "preempt_free_error":
            require(
                req["status"] == "RUNNING" and req["preemptions"] == 0 and len(live) == 1,
                "failed preempt prematurely removed age",
            )
        else:
            require(
                req["status"] == "PREEMPTED" and req["preemptions"] == 1 and not live,
                "actual preempt transition did not revoke protection",
            )
    elif variant == "generator_raise":
        begin = one(events, "finish_call")
        require(
            [e["request_id"] for e in rows(events, "generator_yield")] == ["G0", "G1"],
            "raising generator was consumed twice or not consumed",
        )
        require(
            begin["state"]["waiting"] == check["state"]["waiting"]
            and begin["state"]["active_requests"] == check["state"]["active_requests"]
            and _record_map(begin["guard"]) == _record_map(check["guard"]),
            "generator materialization failure changed state",
        )
        require(
            all(r["status"] == "WAITING" for r in check["state"]["requests"].values()),
            "generator failure partially terminated native requests",
        )
    elif variant == "batch_free_error":
        states = check["state"]["requests"]
        terminal = {rid for rid, state in states.items() if state["status"].startswith("FINISHED")}
        live = {record["request_id"] for record in check["guard"]["records"]}
        require(
            len(terminal) == 2 and len(live) == 1 and not (terminal & live),
            "batch failure did not clean exactly transitioned objects",
        )
        require(
            not check["state"]["waiting"], "native queue removal did not precede individual frees"
        )
    elif variant.startswith("add_failure_"):
        require(
            check["guard"]["next_sequence"] == 1 and len(check["guard"]["records"]) == 1,
            "failed admission allocated or refreshed protection generation",
        )
        require(
            check["guard"]["records"][0]["request_id"] == "old_waiter",
            "failed add erased pre-existing age",
        )
        require(
            ("failed_add" in check["state"]["active_requests"])
            is (variant == "add_failure_postbind"),
            "add failure did not occur at claimed binding boundary",
        )
        if variant == "add_failure_prebind":
            recoveries = [
                e
                for e in rows(events, "fault_recovery")
                if e.get("action") == "fixture_detach_unbound_failed_admission"
            ]
            require(
                len(recoveries) == 1
                and recoveries[0]["native_terminal"] is False
                and recoveries[0]["request_state"]["status"] == "WAITING",
                "unbound failed admission was falsely reported as natively terminated",
            )
    else:
        probes = rows(events, "weakref_probe")
        require(
            [p["label"] for p in probes] == ["before_release", "after_release"],
            "actual Request weakref boundary missing",
        )
        require(
            probes[0]["object"] == probes[1]["object"] and probes[0]["alive"] is True,
            "weakref probe changed identity or lacked real object",
        )
        require(
            all(
                probes[1][key] is False
                for key in ("alive", "owned_by_guard", "native_bound", "observer_retained")
            ),
            "fault/traceback retained actual Request after cleanup",
        )


def audit_rejection(events, metadata):
    name = metadata["variant"]
    config = one(events, "negative_config")
    check = one(events, "check")
    expected, fragment = legacy.REJECTION_EXPECTATIONS[name]
    if name == "window1_apc_off":
        expected = {"mode": metadata["mode"], "window": 1, "apc": False}
    require(config["name"] == check["name"] == name, "rejection leaf mislabeled")
    require(
        canonical(config["effective"]) == canonical(expected), "negative input did not take effect"
    )
    require(config["error_fragment"] == fragment, "wrong rejection target")
    require(events.index(config) < events.index(check), "rejection precedes effective observation")
    require(
        check.get("rejected") is True and fragment in check.get("error", ""),
        "configuration did not reject for expected reason",
    )
    require(
        "not implemented" not in check["error"], "historical guarded contract incorrectly retained"
    )
    return {"rejected": name}


def audit_bc_cause(events, *, pressure, prefix_policy, config, counts, pairs):
    warm = one(events, "A_cached_and_evictable")
    warm_index = events.index(warm)
    state = warm["state"]
    cached = [b for b in state["blocks"] if b["hash"] is not None]
    require(len(cached) == 2, "A did not leave exactly two cache blocks")
    hashes = {b["hash"] for b in cached}
    for block in cached:
        require(
            block["ref_cnt"] == 0
            and block["id"] in state["free_queue"]
            and block["id"] in state["hash_mapping"][block["hash"]],
            "A's cached prefix is not mapped/free/evictable",
        )
    enqueue = rows(events, "bc_enqueued")
    require(
        len(enqueue) == 1 and enqueue[0]["state"]["waiting"] == ["B", "C"],
        "BC must enter in native B,C order",
    )
    actual_enqueues = [e for e in rows(events, "enqueue") if e.get("ids") in (["B"], ["C"])]
    require(
        [e["ids"][0] for e in actual_enqueues] == ["B", "C"],
        "actual enqueue order differs from B,C",
    )
    inputs = {r["id"]: r for r in config["requests"]}
    require(
        all(
            e["tokens"] == inputs[e["ids"][0]]["tokens"]
            and e["arrivals"] == {e["ids"][0]: inputs[e["ids"][0]]["arrival_time"]}
            for e in actual_enqueues
        ),
        "BC arrivals/tokens changed",
    )
    require(not rows(events, "test_only_reorder"), "fixture manual reorder is forbidden")
    lookups = [
        (index, e)
        for index, e in enumerate(events)
        if e["event"] == "return" and e["method"] == "get_computed_blocks" and e.get("owner") == "C"
    ]
    require(bool(lookups), "C lacks a native prefix relookup")
    lookup_index, lookup = lookups[0]
    prefix = prefix_policy
    hit = 0 if pressure and not prefix else 32
    require(lookup["result"]["hit_tokens"] == hit, "C native hit does not match causal condition")
    require(counts.get("A@0", {}).get("prefill") == 33, "A warmup dispatch differs")
    require(
        counts.get("B@0", {}).get("prefill") == 49
        and counts.get("C@0", {}).get("prefill") == 33 - hit,
        "B/C prompt dispatch does not establish 82/50 result",
    )
    require(
        all(
            v["decode"] == 2 and v["repeated_prefill"] == v["repeated_decode"] == 0
            for v in counts.values()
        ),
        "unexpected BC decode or repeated work",
    )
    evictions = []
    for begin, call, end, result in pairs.values():
        if call["method"] != "_maybe_evict_cached_block" or not result["result"]:
            continue
        block = next(b for b in call["state"]["blocks"] if b["id"] == call["details"]["block_id"])
        if block["hash"] not in hashes:
            continue
        require(block["ref_cnt"] == 0, "evicted referenced cache block")
        require(
            block["id"] not in result["state"]["hash_mapping"].get(block["hash"], []),
            "eviction did not remove hash mapping",
        )
        require(
            any(
                a < begin < end < z
                and c["method"] == "allocate_slots"
                and c.get("owner") == call.get("owner")
                and r["result"]["success"]
                for a, c, z, r in pairs.values()
            ),
            "eviction is not nested in successful native allocation",
        )
        evictions.append((begin, end, call.get("owner"), block["hash"]))
    if not pressure:
        require(not evictions, "ample control evicted A's prefix")
    elif not prefix:
        require(
            {
                h
                for begin, end, owner, h in evictions
                if warm_index < begin < end < lookup_index and owner == "B"
            }
            == hashes,
            "pressure native/off lacks B-allocation -> eviction -> C-miss causality",
        )
    else:
        require(
            all(begin > lookup_index for begin, _, _, _ in evictions),
            "prefix lost A before C's native lookup",
        )
        touches = [
            e for e in rows(events, "return") if e["method"] == "touch" and e.get("owner") == "C"
        ]
        require(
            any(
                all(
                    any(b["hash"] == h and b["ref_cnt"] > 0 for b in e["state"]["blocks"])
                    for h in hashes
                )
                for e in touches
            ),
            "C did not natively pin its reused prefix",
        )
    return {"B_prefill": 49, "C_prefill": 33 - hit, "BC_prefill": 82 - hit, "C_native_hit": hit}


def audit_leaf(events, metadata, config):
    if metadata["scenario"] == "global_rejection":
        return audit_rejection(events, metadata)
    if metadata["scenario"] == "postcheck_failure":
        native = one(events, "native_objects")
        require(
            native["class"] == legacy.CUSTOM_CLASS and native["mro"][1] == legacy.NATIVE_CLASS,
            "postcheck bypassed actual native construction",
        )
        observed, rejected = one(events, "postcheck_input"), one(events, "postcheck_rejected")
        targets = {
            "queue": ("waiting", "deque", "FCFSRequestQueue"),
            "coordinator": ("coordinator", "object", "UnitaryKVCacheCoordinator"),
            "full_attention": ("single_type_managers", "object", "FullAttentionManager"),
            "block_size": ("block_size", 32, "native cache block sizes"),
            "v2": ("use_v2_model_runner", False, "native Model Runner V2"),
        }
        field, effective, fragment = targets[metadata["variant"]]
        require(
            observed["variant"] == rejected["variant"] == metadata["variant"],
            "postcheck leaf mislabeled",
        )
        require(
            observed["field"] == field and canonical(observed["effective"]) == canonical(effective),
            "postcheck incompatible native object was not observed",
        )
        require(fragment in rejected["error"], "postcheck failed for wrong reason")
        call, done = one(events, "native_shutdown_call"), one(events, "native_shutdown_return")
        require(
            events.index(native)
            < events.index(observed)
            < events.index(call)
            < events.index(done)
            < events.index(rejected),
            "postcheck did not perform actual cleanup after construction",
        )
        return {"postcheck": metadata["variant"]}
    native = audit_identity(events, metadata, config)
    pairs = audit_calls(events)
    counts = audit_dispatch(events)
    maximum = legacy.audit_fifo(events)
    guard, oracle = audit_guard(events, metadata, config)
    audit_bounds(events, metadata, oracle)
    drained = rows(events, "fixture_drained")
    require(drained, "leaf has no complete native resource drain")
    resources = audit_resources(drained[-1]["state"])
    retry_checks = audit_fault_cause(events, metadata, oracle)
    audit_exception_fixture(events, metadata, oracle)
    scenario, mode = metadata["scenario"], metadata["mode"]
    if scenario in {"bc_pressure", "bc_ample"}:
        prefix_policy = mode == "prefix" or (
            mode == "guarded" and metadata["variant"] == "unprotected"
        )
        audit_bc_cause(
            events,
            pressure=scenario == "bc_pressure",
            prefix_policy=prefix_policy,
            config=config,
            counts=counts,
            pairs={k: p for k, p in pairs.items() if p[3]["event"] == "return"},
        )
    elif scenario == "apc_off":
        require(
            counts.get("B@0", {}).get("prefill") == 49
            and counts.get("C@0", {}).get("prefill") == 33,
            "APC-off dispatched work changed",
        )
    elif scenario == "probe":
        audit_probe_leaf(events, metadata["variant"])
    elif scenario == "chunk":
        intervals = [
            i
            for e in rows(events, "dispatch")
            for i in e["intervals"]
            if i["request_id"] == "D" and i["prefill_tokens"]
        ]
        require(
            [[i["start"], i["end"]] for i in intervals]
            == [[0, 16], [16, 32], [32, 48], [48, 64], [64, 65]],
            "actual multistep prompt chunk changed",
        )
        require(
            native["capacity"] == 10
            and native["effective_config"]["budget"] == 16
            and native["effective_config"]["active"] == 1,
            "chunk fixture geometry changed",
        )
        require(counts["D@0"]["prefill"] == 65, "chunk prompt dispatch missing")
        require(
            all(
                "D" not in e["ids"]
                for e in rows(events, "candidates")
                if e["step"] > intervals[0]["step"]
            ),
            "dispatched chunk regained freshness",
        )
        d_steps = [
            e["step"]
            for e in rows(events, "dispatch")
            if any(i["request_id"] == "D" for i in e["intervals"])
        ]
        c_steps = [
            e["step"]
            for e in rows(events, "dispatch")
            if any(i["request_id"] == "C" for i in e["intervals"])
        ]
        require(
            c_steps and min(c_steps) > max(d_steps), "hot request crossed running chunk priority"
        )
    elif scenario == "preemption" and metadata["variant"] == "default":
        preempts = [
            p
            for p in pairs.values()
            if p[1]["method"] == "_preempt_request" and p[1]["owner"] == "R1"
        ]
        require(
            preempts and preempts[0][3]["event"] == "return", "natural native preemption missing"
        )
        victim = preempts[0][3]["state"]["requests"]["R1"]
        require(
            victim["computed_optimistic"] == 0
            and victim["preemptions"] >= 1
            and victim["stale_output_tokens"] > 0,
            "preempted request did not retain pending stale-output state",
        )
        barrier = [e for e in rows(events, "schedule_before") if e["phase"] == "preempted_barrier"]
        require(
            len(barrier) == 1 and barrier[0]["state"]["waiting"][:2] == ["R1", "H"],
            "natural resumed barrier not exercised",
        )
        scheduled = next(
            e for e in rows(events, "schedule_after") if e["step"] == barrier[0]["step"]
        )
        require("H" not in scheduled["scheduled"], "hot request crossed native preemption barrier")
        require(
            any(
                i["request_id"] == "R1" and i["preemptions"] > 0
                for e in rows(events, "dispatch")
                for i in e["intervals"]
            ),
            "request never natively resumed",
        )
        require(maximum == 2, "natural preemption did not exercise async pending depth two")
    elif scenario == "cancellation":
        cancelled = rows(events, "cancel")
        require(
            [e["request_id"] for e in cancelled] == ["waiting_cancel", "running_cancel"],
            "waiting and pending cancellation coverage missing",
        )
        require(
            "waiting_cancel@0" not in counts
            and "running_cancel@0" in counts
            and counts["survivor@0"]["prefill"] == 33,
            "native cancellation dispatch behavior changed",
        )
        require(
            events.index(cancelled[-1])
            < next(i for i, e in enumerate(events) if e["event"] == "output_after"),
            "cancellation was not exercised with pending output",
        )
    elif scenario == "id_reuse":
        boundaries = rows(events, "reuse_boundary")
        require(
            len(boundaries) == 2
            and all(
                e["pending_outputs"] == 0 and not e["state"]["active_requests"] for e in boundaries
            ),
            "ID reused before prior native outputs drained",
        )
        for event in boundaries:
            audit_reuse_boundary(event)
        for rid in ("reuse", "cancel_reuse"):
            enqueues = [e for e in rows(events, "enqueue") if e["ids"] == [rid]]
            require(
                [e["generation"] for e in enqueues] == [0, 1]
                and enqueues[0]["actual_object"] != enqueues[1]["actual_object"],
                "ID reuse did not bind a fresh actual object",
            )
            require(
                all(
                    counts[f"{rid}@{g}"]["prefill"] == 33
                    and counts[f"{rid}@{g}"]["repeated_prefill"] == 0
                    for g in (0, 1)
                ),
                "native dispatch accounting crossed generations",
            )
        if metadata["variant"] == "stale_object_cleanup":
            probe = one(events, "identity_probe")
            require(
                probe["old_object"] != probe["new_object"]
                and probe["removed"] is False
                and probe["before"] == probe["after"],
                "old object cleanup removed new binding",
            )
            require(
                any(r["object"] == probe["new_object"] for r in probe["after"]["records"]),
                "identity probe lacked a live new-generation record",
            )
    elif scenario == "score_eviction":
        found = False
        for index, score in enumerate(events):
            if (
                score["event"] != "score"
                or score.get("request_id") != "C"
                or score.get("score") != 32
            ):
                continue
            lookup = next(
                (
                    (j, e)
                    for j, e in enumerate(events)
                    if j > index
                    and e["event"] == "return"
                    and e["method"] == "get_computed_blocks"
                    and e.get("owner") == "C"
                    and e["step"] == score["step"]
                ),
                None,
            )
            if lookup and lookup[1]["result"]["hit_tokens"] < 32:
                require(
                    any(
                        index < a < z < lookup[0]
                        and c["method"] == "_maybe_evict_cached_block"
                        and c["owner"] == "R"
                        and r["result"] is True
                        for a, c, z, r in pairs.values()
                        if r["event"] == "return"
                    ),
                    "score changed without actual running-stage eviction",
                )
                found = True
        require(found, "native allocation/relookup never superseded prior read-only score")
    elif scenario == "clock_thresholds":
        variant = metadata["variant"]
        plan = config["clock_plans"][variant]
        if variant.startswith("fault_"):
            require(
                oracle.fault is not None and oracle.clock_failures,
                "invalid clock did not latch a real fault",
            )
            require(rows(events, "expected_error"), "clock failure not exercised")
        else:
            decisions = [d for d in oracle.decisions if d["now"] == plan["decision_ns"]]
            require(decisions, "frozen threshold decision was not observed")
            require(
                any(
                    any(r["enqueued_ns"] == plan["enqueue_ns"] for r in d["records"].values())
                    for d in decisions
                ),
                "threshold uses wrong enqueue origin",
            )
    elif scenario == "protected_order":
        require(
            any(len(d["protected"]) >= 2 for d in oracle.decisions),
            "multiple protected requests were not exercised",
        )
        if metadata["variant"] == "prior_reorder":
            require(
                any(d["objects"] != d["after"] and not d["protected"] for d in oracle.decisions),
                "queue was never prefix reordered before protection",
            )
            require(
                any(
                    d["objects"] != d["after"] and len(d["protected"]) >= 2
                    for d in oracle.decisions
                ),
                "protected FIFO did not recover original admission order",
            )
        else:
            require(
                any(
                    len(d["protected"]) >= 2
                    and len({d["records"][token]["enqueued_ns"] for token in d["protected"]}) == 1
                    and len({d["records"][token]["sequence"] for token in d["protected"]})
                    == len(d["protected"])
                    for d in oracle.decisions
                ),
                "same-tick ordering did not use distinct enqueue sequences",
            )
    elif scenario == "window_age":
        old = [token for token, (rid, _) in oracle.objects.items() if rid == "B"]
        require(len(old) == 1, "window-age fixture replaced B object")
        token = old[0]
        outside = [
            d for d in oracle.decisions if token not in d["objects"] and token in d["records"]
        ]
        inside = [d for d in oracle.decisions if token in d["protected"]]
        require(
            outside and inside and outside[0]["step"] < inside[0]["step"],
            "request did not wait outside window before protection",
        )
        require(
            outside[0]["records"][token] == inside[0]["records"][token]
            and inside[0]["records"][token]["enqueued_ns"] == 0,
            "window entry reset original waiting age/sequence",
        )
    elif scenario == "allocation_retry":
        failures = [
            p
            for p in pairs.values()
            if p[1]["method"] == "allocate_slots"
            and p[1]["owner"] == "B"
            and p[3]["event"] == "return"
            and p[3]["result"]["success"] is False
        ]
        require(
            len(failures) >= 2 and len({p[1]["step"] for p in failures}) >= 2,
            "repeated real allocation failure missing",
        )
        attempts = rows(events, "allocation_retry")
        require(len(attempts) == 2, "resource retry observations missing")
        original = next(r for r in attempts[0]["first_guard"]["records"] if r["request_id"] == "B")
        require(
            all(
                next(r for r in e["guard"]["records"] if r["request_id"] == "B") == original
                for e in attempts
            ),
            "allocation failure refreshed B age or sequence",
        )
        require(
            counts.get("B@0", {}).get("prefill") == 49,
            "resource-starved request never actually dispatched",
        )
    elif scenario == "hot_flow":
        rounds = rows(events, "hot_round")
        require(
            [e["round"] for e in rounds] == [1, 2, 3, 4]
            and [e["tick"] for e in rounds] == [100000000, 200000000, 300000000, 500000000],
            "frozen hot-arrival stream changed",
        )
        scheduled = [list(e["scheduled"]) for e in rounds]
        expected = (
            [["B"], ["H1"], ["H2"], ["H3"]]
            if mode in {"native", "off"}
            else [["H1"], ["H2"], ["H3"], ["B" if mode == "guarded" else "H4"]]
        )
        require(
            scheduled == expected,
            "actual hot/cold admission does not establish repeated bypass and bounded-window protection",
        )
        require(
            native["capacity"] == 5 and native["effective_config"]["active"] == 1,
            "hot-flow resource geometry changed",
        )
        for event in rounds:
            after = next(e for e in rows(events, "schedule_after") if e["step"] == event["step"])
            require(
                after["scheduled"] == event["scheduled"],
                "hot-flow label differs from real native output",
            )
        require(
            counts["B@0"]["prefill"] == 49 and all(f"H{i}@0" in counts for i in range(1, 5)),
            "hot-flow fixture did not drain every real request",
        )
    elif scenario == "clock_absence":
        observed = one(events, "clock_absence")
        require(
            observed["reads"] == 0
            and observed["guard_present"] is False
            and not rows(events, "clock"),
            "off/prefix consulted protection state or clock",
        )
        require(
            rows(events, "native_add_return")
            and rows(events, "native_schedule_return")
            and {e["request_id"] for e in rows(events, "cancel")}
            == {"waiting_cancel", "running_cancel"}
            and {"A@0", "B@0", "C@0", "running_cancel@0", "survivor@0"} <= set(counts),
            "clock absence lacks actual admission/scheduling/cancellation controls",
        )
    elif scenario == "enqueue_identity" and not metadata["variant"].startswith("add_failure_"):
        variant = metadata["variant"]
        if variant in {"same_object_duplicate", "different_object_duplicate"}:
            calls = rows(events, "native_add_call")
            require(
                len(calls) == 2 and calls[1]["previous_object"] == calls[0]["object"],
                "duplicate ID did not address old native binding",
            )
            require(
                (calls[1]["object"] == calls[0]["object"]) is (variant == "same_object_duplicate"),
                "duplicate-object identity variant mislabeled",
            )
            require(
                len(rows(events, "native_add_raise")) == 1
                and oracle.sequence == 1
                and oracle.read_index == 1
                and oracle.fault is None,
                "duplicate add created/refreshed generation or poisoned existing state",
            )
        elif variant == "session_update":
            returned = rows(events, "native_add_return")
            require(
                len(returned) == 2
                and returned[0]["actual_object"] == returned[1]["actual_object"]
                and returned[1]["object"] != returned[1]["actual_object"],
                "session continuation did not update the actual old native object",
            )
            require(
                oracle.sequence == oracle.read_index == 0 and rows(events, "session_updated"),
                "session update gained ordinary fresh protection",
            )
        else:
            require(
                variant in {"initial_session_id", "initial_resumable", "initial_abort_immediately"},
                "unknown initial nonordinary variant",
            )
            require(
                oracle.sequence == oracle.read_index == 0,
                "initial nonordinary request read clock or registered",
            )
            returned = one(events, "native_add_return")
            rid, generation = oracle.objects[returned["actual_object"]]
            fields = _fields_for(returned["state"], rid, generation)
            field = {
                "initial_session_id": "session_id",
                "initial_resumable": "resumable",
                "initial_abort_immediately": "abort_immediately",
            }[variant]
            require(bool(fields[field]), "nonordinary native input flag was not actually set")
    elif scenario == "metadata_terminal" and metadata["variant"] in {
        "str",
        "list",
        "generator",
        "none",
        "unknown_duplicate",
    }:
        variant = metadata["variant"]
        call, returned = one(events, "finish_call"), one(events, "finish_return")
        supplied = call.get("supplied_input")
        target = {
            "str": "T0",
            "list": ["T0", "T1", "T2"],
            "none": None,
            "unknown_duplicate": ["unknown", "T0", "T0", "T1"],
        }
        if variant == "generator":
            yielded = [e["request_id"] for e in rows(events, "generator_yield")]
            require(
                yielded == ["T0", "T1", "T1", "unknown"],
                "one-shot generator not consumed exactly once",
            )
            expected_ids = {"T0", "T1"}
        else:
            require(
                canonical(supplied) == canonical(target[variant]),
                "actual finish input shape/IDs differ from required variant",
            )
            expected_ids = (
                {"T0"}
                if variant == "str"
                else {"T0", "T1"}
                if variant == "unknown_duplicate"
                else {"T0", "T1", "T2"}
            )
        returned_ids = [oracle._object(t)[0] for t in returned["objects"]]
        require(
            set(returned_ids) == expected_ids and len(returned_ids) == len(set(returned_ids)),
            "native finish consumed wrong/duplicate request identities",
        )
        require(
            all(r["request_id"] not in expected_ids for r in returned["guard"]["records"]),
            "finished IDs retained metadata",
        )
    elif scenario == "shutdown":
        observation = one(events, "shutdown_observation")
        require(
            observation["after"]["active_requests"] == observation["before"]["active_requests"]
            and observation["after"]["active_requests"],
            "native shutdown was incorrectly treated as request termination",
        )
        require(
            observation["after"]["guard"]["records"] == []
            and observation["after"]["guard"]["closed"] is True,
            "shutdown did not clear and close protection",
        )
        if metadata["variant"] == "running":
            require(
                observation["pending_outputs"] > 0
                and any(b["ref_cnt"] > 0 for b in observation["after"]["blocks"] if not b["null"]),
                "shutdown-running fixture did not retain real native pending/resources",
            )
        if metadata["variant"] == "raises":
            require(
                rows(events, "native_shutdown_raise"),
                "exceptional native shutdown was not observed",
            )
        retry_checks = audit_rejected_retries(
            events, schedule_phase="after_shutdown", request_id="after_closed", state="closed"
        )
    elif scenario == "repeat_cycles":
        cycles = rows(events, "cycle_drained")
        require(
            [e["cycle"] for e in cycles] == list(range(8)),
            "repeated mixed lifecycle cycles incomplete",
        )
        require(
            all(not e["guard"]["records"] and e["pending_outputs"] == 0 for e in cycles),
            "repeated cycles accumulated protected/pending history",
        )
        for event in cycles:
            audit_reuse_boundary(event)
        require(
            oracle.generations.get("cycled") == 7 and oracle.sequence == 8,
            "repeated ID reuse did not form fresh monotonically sequenced objects",
        )
    elif scenario == "window":
        if metadata["variant"].startswith("missing_"):
            require(
                oracle.fault == "missing_record", "single-candidate missing metadata bypassed guard"
            )
        else:
            check = one(events, "check")
            require(check["name"] == metadata["variant"], "window leaf mislabeled")
            if metadata["variant"] in {"score_exception", "pop_exception", "prepend_exception"}:
                require(
                    check["before"] == check["after"] and check["error"],
                    "queue transaction failed to restore",
                )
                require(
                    oracle.fault is None, "recoverable M2A queue exception poisoned guarded state"
                )
            elif metadata["variant"] == "window_one":
                enters = [
                    e for e in rows(events, "native_schedule_enter") if e["step"] == check["step"]
                ]
                require(
                    len(enters) == 1 and enters[0]["state"]["waiting"] == check["before"],
                    "window-one changed native admission order",
                )
            elif metadata["variant"] == "window_limit":
                require(
                    check["after"] == ["hot", "cold", "tail"],
                    "bounded window did not preserve suffix",
                )
            elif metadata["variant"] == "other_queues":
                require(
                    check["after"] == ["hot", "tail", "cold"]
                    and all(
                        check["state"][k]
                        for k in ("running", "kv_holding_waiting", "deferred_waiting")
                    ),
                    "other-queue fixture lacked real barriers or changed ordering",
                )
            elif metadata["variant"] == "repeated_step":
                selected = [
                    e
                    for e in rows(events, "window_after")
                    if e["before"] and e["before"][0] in {"cold", "hot"}
                ]
                require(
                    len(selected) == 2
                    and selected[0]["step"] != selected[1]["step"]
                    and all(len(e["scores"]) == 3 for e in selected),
                    "repeated steps did not freshly score each candidate",
                )
            else:
                require(
                    check["before"] == check["after"],
                    "stable FIFO/barrier/prepend behavior changed",
                )
    return {
        "dispatch": counts,
        "maximum_pending": maximum,
        "guard": guard,
        "native_resources": resources,
        "rejected_retry_attempts": retry_checks,
    }


def without_guard(value):
    if isinstance(value, list):
        return [without_guard(v) for v in value]
    if not isinstance(value, dict):
        return value
    return {k: without_guard(v) for k, v in value.items() if k != "guard"}


def comparison_view(events):
    """Hash the full normalized native stream while keeping only small comparisons."""
    kinds = {
        "enqueue",
        "bc_enqueued",
        "A_cached_and_evictable",
        "call",
        "return",
        "dispatch",
        "advance",
        "schedule_before",
        "schedule_after",
        "native_schedule_enter",
        "native_schedule_return",
        "output_before",
        "output_after",
        "output_stats",
        "published",
        "drained",
        "cancel",
        "reuse_boundary",
    }
    digest = hashlib.sha256()
    for event in events:
        if event["event"] in kinds:
            raw = {
                k: v
                for k, v in event.items()
                if k not in {"case", "scenario", "mode", "variant", "repeat", "counts", "guard"}
            }
            digest.update(canonical(legacy.normalized(without_guard(raw))).encode())
            digest.update(b"\n")
    identity = rows(events, "native_objects")
    warm = rows(events, "A_cached_and_evictable")
    return {
        "native_digest": digest.hexdigest(),
        "instance_fields": identity[0]["instance_fields"] if identity else [],
        "decisions": [[e["step"], e["request_ids"]] for e in rows(events, "decision_after")],
        "outputs": [[e["step"], e["scheduled"]] for e in rows(events, "schedule_after")],
        "warm_digest": hashlib.sha256(
            canonical(legacy.normalized(without_guard(warm[0]["state"]))).encode()
        ).hexdigest()
        if warm
        else None,
    }


def audit_controls(views, results, expected):
    off_pairs = prefix_pairs = 0
    warm_states = {}
    for case, meta in expected.items():
        if meta["mode"] == "off" and meta["scenario"] != "clock_absence":
            native_case = f"{meta['repeat']}:{meta['scenario']}:native:{meta['variant']}"
            require(native_case in views, "off case has no real native control")
            require(
                views[case]["native_digest"] == views[native_case]["native_digest"],
                f"off/native full normalized behavior differs: {case}",
            )
            require(
                set(views[case]["instance_fields"]) - set(views[native_case]["instance_fields"])
                == {"prefix_config"},
                "off introduced additional persistent metadata",
            )
            off_pairs += 1
        if meta["mode"] == "guarded" and meta["scenario"] in {
            "bc_pressure",
            "bc_ample",
            "probe",
            "window",
            "score_eviction",
            "chunk",
            "preemption",
            "cancellation",
            "id_reuse",
        }:
            variant = "default" if meta["variant"] == "unprotected" else meta["variant"]
            prefix_case = f"{meta['repeat']}:{meta['scenario']}:prefix:{variant}"
            if prefix_case in views and meta["variant"] != "protected":
                require(
                    views[case]["decisions"] == views[prefix_case]["decisions"]
                    and views[case]["outputs"] == views[prefix_case]["outputs"],
                    f"unexpired guarded decision/native dispatch differs from prefix: {case}",
                )
                prefix_pairs += 1
        if meta["scenario"] in {"bc_pressure", "bc_ample"}:
            warm = views[case]["warm_digest"]
            require(warm is not None, "BC warm state absent")
            prior = warm_states.setdefault(meta["scenario"], warm)
            require(warm == prior, "BC native initial cache state differs across modes/repeats")
    costs = {}
    for case, meta in expected.items():
        if meta["scenario"] != "hot_flow":
            continue
        costs[case] = {
            key: sum(value[key] for value in results[case]["dispatch"].values())
            for key in ("prefill", "decode", "repeated_prefill", "repeated_decode")
        }
    return {
        "native_off_pairs": off_pairs,
        "unexpired_guarded_prefix_pairs": prefix_pairs,
        "hot_flow_actual_dispatch_work": costs,
        "metric": "host-side dispatched tokens; no GPU performance conclusion",
    }


def iter_case_groups(trace_path, expected):
    """Bound memory to one contiguous leaf; reject interleaving and reopened cases."""
    seen, current, events = set(), None, []
    with Path(trace_path).open() as stream:
        for line in stream:
            require(bool(line.strip()), "empty trace record")
            event = json.loads(line)
            require(type(event) is dict and event.get("case") in expected, "unexpected trace case")
            case = event["case"]
            require(
                event.get("event")
                not in {"failed", "skipped", "case_failure", "test_only_reorder"},
                "failed/skipped/manual reorder trace cannot pass",
            )
            require(type(event.get("event")) is str, "trace event name missing")
            for key, value in expected[case].items():
                require(
                    event.get(key) == value and (key != "repeat" or type(event[key]) is int),
                    "trace case metadata mismatch",
                )
            if case != current:
                require(case not in seen, "trace case duplicated or interleaved")
                if current is not None:
                    yield current, events
                seen.add(case)
                current, events = case, []
            events.append(event)
    require(current is not None, "trace must be nonempty")
    yield current, events
    require(seen == set(expected), "trace must contain entire unique frozen leaf matrix")


def assert_trace(trace_path, config=None, suite="full"):
    require_assertions()
    if config is None:
        config = json.loads((ROOT / "configs/native-m2b.json").read_text())
    expected = expected_cases(config, suite=suite)
    results, views = {}, {}
    for case, events in iter_case_groups(trace_path, expected):
        metadata = expected[case]
        begin, end = one(events, "case_begin"), one(events, "case_end")
        require(events[0] is begin and events[-1] is end, "leaf boundaries incomplete/repeated")
        require(
            type(begin.get("seed")) is int and begin["seed"] == config["seed"],
            "leaf seed differs from frozen input",
        )
        require(end.get("result", {}).get("status") == "passed", "leaf failed/skipped/incomplete")
        results[case] = audit_leaf(events, metadata, config)
        views[case] = comparison_view(events)
    comparisons = audit_controls(views, results, expected)
    return {
        "matrix_verified": True,
        "suite": suite,
        "case_count": len(results),
        "cases": results,
        "comparisons": comparisons,
    }


def main():
    require_assertions()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/native-m2b.json")
    parser.add_argument("--suite", choices=("full", "compat"), default="full")
    args = parser.parse_args()
    print(
        json.dumps(
            assert_trace(args.trace, json.loads(args.config.read_text()), suite=args.suite),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
