"""D6 call and identity binding with a CPU-only UniProc-shaped harness."""

import sys
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from vllm_prefix_scheduler import m5_d6_observer as observer

START = 10_000_000_000
END = START + observer.D6_WINDOW_NS


class Clock:
    stamp = 0

    def monotonic_ns(self):
        self.stamp += 1
        return self.stamp


class Request:
    def __init__(self, identity):
        self.request_id = identity
        self.num_computed_tokens = 0
        self.num_prompt_tokens = 641
        self.num_preemptions = 0


class Schedule:
    pending_structured_output_tokens = False

    def __init__(self, tokens):
        self.num_scheduled_tokens = tokens
        self.total_num_scheduled_tokens = sum(tokens.values())


class AsyncOutput:
    def __init__(self, events, value):
        self.events, self.value = events, value

    def get_output(self):
        self.events.append("materialize")
        return self.value


class AsyncFuture(Future):
    single_value = True

    def __init__(self, output, events):
        super().__init__()
        self.async_output, self.events = output, events

    def result(self, timeout=None):
        self.events.append("fifo-result")
        if not self.done():
            self.set_result(self.async_output.get_output())
        return super().result(timeout=timeout)


class Scheduler:
    max_num_active_reqs = 4

    def __init__(self, clock, events, plans):
        self.clock, self.events = clock, events
        self.plans = deque(plans)
        self.requests = {}
        self.running, self.waiting, self.kv_holding_waiting = [], [], []

    def add_request(self, request):
        self.requests[request.request_id] = request
        self.waiting.append(request)

    def has_requests(self):
        return bool(self.plans)

    def schedule(self, throttle):
        self.events.append("schedule")
        output = self.plans.popleft()
        for identity, count in output.num_scheduled_tokens.items():
            request = self.requests[identity]
            request.num_computed_tokens += count
            if request in self.waiting:
                self.waiting.remove(request)
            if request not in self.running:
                self.running.append(request)
        return output

    def get_grammar_bitmask(self, output):
        return None

    def update_from_output(self, schedule, output):
        self.events.append("consume")
        for identity in output.req_ids:
            request = self.requests[identity]
            if request in self.running:
                self.running.remove(request)
        return {
            0: SimpleNamespace(
                timestamp=self.clock.monotonic_ns() / 1_000_000_000,
                outputs=[
                    SimpleNamespace(request_id=rid, new_token_ids=tokens)
                    for rid, tokens in zip(output.req_ids, output.sampled_token_ids, strict=True)
                ],
            )
        }


class Executor:
    def __init__(self, runner, events):
        self.runner, self.events = runner, events

    def execute_model(self, schedule, *, non_block):
        assert non_block
        self.events.append("execute")
        tokens = schedule.num_scheduled_tokens
        self.runner.execute_model_state = None
        future = Future()
        if tokens:
            effective = sum(tokens.values())
            offsets = [0]
            for count in tokens.values():
                offsets.append(offsets[-1] + count)
            self.runner.execute_model_state = SimpleNamespace(
                input_batch=SimpleNamespace(
                    req_ids=list(tokens),
                    query_start_loc_np=offsets,
                    prefill_len_np=[641] * len(tokens),
                    num_tokens=effective,
                    num_tokens_after_padding=effective + 2,
                )
            )
            future.set_result(None)
        else:
            future.set_result(SimpleNamespace(req_ids=[], sampled_token_ids=[]))
        # Native UniProc surfaces exceptions synchronously, before the hook
        # wraps a later FIFO result.
        future.result()
        return future

    def sample_tokens(self, grammar, *, non_block):
        assert non_block
        self.events.append("sample")
        ids = self.runner.execute_model_state.input_batch.req_ids
        self.runner.execute_model_state = None
        output = SimpleNamespace(req_ids=ids, sampled_token_ids=[[3] for _ in ids])
        return AsyncFuture(AsyncOutput(self.events, output), self.events)


class Core:
    is_mm_encoder_only = is_pooling_model = check_for_draft_tokens = False
    batch_queue_size = 2

    def __init__(self, scheduler, executor):
        self.scheduler, self.model_executor = scheduler, executor
        self.batch_queue = deque(maxlen=2)
        self.step_fn = self.step

    def step(self):
        executed = False
        if self.scheduler.has_requests():
            output = self.scheduler.schedule(False)
            execution = self.model_executor.execute_model(output, non_block=True)
            executed = output.total_num_scheduled_tokens > 0
            future = (
                self.model_executor.sample_tokens(None, non_block=True) if executed else execution
            )
            self.batch_queue.appendleft((future, output, execution))
            if len(self.batch_queue) < 2 and (executed or self.scheduler.has_requests()):
                return None, executed
        elif not self.batch_queue:
            return None, False
        future, output, _ = self.batch_queue.pop()
        return self.scheduler.update_from_output(output, future.result()), executed

    def _should_throttle_prefills(self):
        return False

    def _process_input_queue(self):
        pass

    def _process_engine_step(self):
        self.last_step = self.step_fn()
        return self.last_step[1]

    def log_error_detail(self, output):
        return nullcontext()

    def capture_iteration_details(self, output):
        return nullcontext()

    def _process_aborts_queue(self):
        pass

    def _attach_iteration_details(self, packets, details):
        pass

    def _attach_weight_version(self, packets):
        pass


def harness(monkeypatch, plans, step_body=None, future_type=AsyncFuture):
    clock = Clock()
    monkeypatch.setattr(observer, "time", clock)
    events = []
    runner = SimpleNamespace(execute_model_state=None)
    scheduler = Scheduler(clock, events, plans)
    core = Core(scheduler, Executor(runner, events))
    if step_body is not None:
        core.step_fn = step_body.__get__(core)
    trace = observer._Trace(core, runner, future_type, AsyncOutput, START, END)
    return core, trace, clock, events


def test_d6_observes_actual_fifo_once_and_retains_crossing_batch(monkeypatch):
    core, trace, clock, events = harness(
        monkeypatch, [Schedule({"r": 3}), Schedule({"r": 2}), Schedule({"r": 1})]
    )
    clock.stamp = START
    core._process_input_queue()
    core.scheduler.add_request(Request("r"))
    assert core._process_engine_step() is True
    assert core.last_step == (None, True)
    assert core._process_engine_step() is True
    clock.stamp = END
    assert core._process_engine_step() is True
    assert core._process_engine_step() is False
    report = trace.export()
    assert report["status"] == "complete"
    assert report["errors"] == []
    assert [step["scheduled_batch_id"] for step in report["steps"]] == [1, 2, None]
    assert [step["consumed_batch_id"] for step in report["steps"]] == [None, 1, 2]
    assert [step["queue_after"] for step in report["steps"]] == [1, 1, 1]
    assert [row["step_id"] for row in report["loops"]] == [1, 2, 3]
    assert len(report["input_polls"]) == 1
    assert (
        events.count("materialize") == events.count("fifo-result") == events.count("consume") == 3
    )
    first, second = report["batches"]
    assert first["scheduled"][0] == {
        "request_id": "r",
        "generation": 0,
        "tokens": 3,
        "start": 0,
        "end": 3,
        "prompt_length": 641,
        "preemptions": 0,
    }
    assert (second["scheduled"][0]["start"], second["scheduled"][0]["end"]) == (3, 5)
    assert first["schedule_before"]["waiting"] == [["r", 0]]
    assert first["schedule_after"]["running"] == [["r", 0]]
    assert first["consume_before"]["pending_batch_ids"] == [2]
    assert first["consume_after"]["running"] == []
    assert second["consume_before"]["pending_batch_ids"] == [None]
    assert second["consume_end_ns"] > report["end_ns"]
    assert first["runner"] == {
        "effective_tokens": 3,
        "padded_tokens": 5,
        "rows": [
            {
                "request_id": "r",
                "generation": 0,
                "query_start": 0,
                "query_end": 3,
                "prefill_length": 641,
            }
        ],
    }
    assert first["materialized_counts"] == [
        {"request_id": "r", "generation": 0, "generated_tokens": 1}
    ]
    assert (
        first["consume_start_ns"] < first["output_counts"][0]["packet_ns"] < first["consume_end_ns"]
    )
    assert first["output_counts"][0]["token_ids"] == [3]
    assert (
        first["future_result_start_ns"]
        < first["materialize_start_ns"]
        < first["materialize_end_ns"]
        < first["future_result_end_ns"]
    )
    assert core.step_fn == core.step
    assert "execute_model" not in vars(core.model_executor)


@pytest.mark.parametrize("overflow", [False, True])
def test_d6_preserves_native_empty_batch_and_fails_closed_at_limit(monkeypatch, overflow):
    if overflow:
        monkeypatch.setattr(observer, "D6_BATCH_LIMIT", 1)
    core, trace, clock, events = harness(monkeypatch, [Schedule({}), Schedule({"r": 1})])
    clock.stamp = START
    core.scheduler.add_request(Request("r"))
    assert core._process_engine_step() is False
    assert core.last_step == (None, False)
    assert core._process_engine_step() is True
    assert core._process_engine_step() is False
    clock.stamp = END
    report = trace.export()
    assert report["overflow"] is overflow
    assert report["status"] == ("incomplete" if overflow else "complete")
    first = report["batches"][0]
    assert first["runner"] is first["sample_start_ns"] is first["materialize_start_ns"] is None
    assert first["future_result_end_ns"] < first["consume_start_ns"]
    assert events.count("materialize") == 1
    assert events.count("consume") == 2


def test_d6_never_claims_unfinished_crossing_is_complete(monkeypatch):
    core, trace, clock, _ = harness(monkeypatch, [Schedule({"r": 1})])
    clock.stamp = START
    core.scheduler.add_request(Request("r"))
    core._process_engine_step()
    clock.stamp = END
    report = trace.export()
    assert report["status"] == "incomplete"
    assert any("drained" in row["error"] for row in report["errors"])


def test_d6_arm_binds_exact_owner_from_native_rpc_stack(monkeypatch):
    class Runner:
        execute_model_state = None

    class UniProc(Executor):
        def collective_rpc(self, method, timeout, args, kwargs):
            return [getattr(self.driver_worker.worker, method)(*args, **kwargs)]

    class EngineCore(Core):
        async_scheduling = True
        step_with_batch_queue = Core.step

        def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
            return self.model_executor.collective_rpc(method, timeout, args, kwargs)

    clock = Clock()
    monkeypatch.setattr(observer, "time", clock)
    worker = observer.D6WorkerExtension()
    worker.model_runner = Runner()
    executor = UniProc(worker.model_runner, [])
    executor.driver_worker = SimpleNamespace(worker=worker)
    core = EngineCore(Scheduler(clock, [], []), executor)
    modules = {
        "vllm.v1.engine.core": SimpleNamespace(EngineCore=EngineCore),
        "vllm.v1.executor.uniproc_executor": SimpleNamespace(
            AsyncOutputFuture=AsyncFuture, UniProcExecutor=UniProc
        ),
        "vllm.v1.worker.gpu.async_utils": SimpleNamespace(AsyncOutput=AsyncOutput),
        "vllm.v1.worker.gpu.model_runner": SimpleNamespace(GPUModelRunner=Runner),
        "vllm_prefix_scheduler.scheduler": SimpleNamespace(PrefixAsyncScheduler=Scheduler),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(RuntimeError, match="idle pinned UniProc"):
        worker.m5_d6_arm(START, END)
    [proof] = core.collective_rpc("m5_d6_arm", args=(START, END), kwargs={})
    assert proof == {"armed_ns": 1, "start_ns": START, "end_ns": END}
    assert worker.m5_d6_export()["status"] == "incomplete"
    assert core.step_fn == core.step
