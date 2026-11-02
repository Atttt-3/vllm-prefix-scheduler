"""Observer transport preserves native return/exception and lazy consumption."""

import gc
import json
import weakref
from types import SimpleNamespace

import pytest

from vllm_prefix_scheduler import m3_events as events
from vllm_prefix_scheduler.m3_worker import install_runner_observer


def test_async_materialization_remains_lazy_and_identity_preserved(monkeypatch, tmp_path):
    monkeypatch.setenv("M3_TRACE_DIR", str(tmp_path))
    calls = []
    result = SimpleNamespace(req_ids=["case::B"], sampled_token_ids=[[42]])

    class Output:
        def get_output(self):
            calls.append("native")
            return result

    output = Output()
    assert events.observe_materialization(output, "7:1") is output
    assert calls == []
    assert output.get_output() is result
    assert calls == ["native"]
    trace = [
        json.loads(line)
        for path in tmp_path.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert trace[-1]["event"] == "model_output_materialized"
    assert trace[-1]["sampled_token_ids"] == [[42]]


def test_async_original_exception_is_not_replaced_by_trace_failure(monkeypatch, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    monkeypatch.setenv("M3_TRACE_DIR", str(blocker))
    failure = RuntimeError("native failure")
    calls = []

    class Output:
        def get_output(self):
            calls.append("native")
            raise failure

    output = Output()
    events.observe_materialization(output, "7:2")
    with pytest.raises(RuntimeError) as raised:
        output.get_output()
    assert raised.value is failure
    assert calls == ["native"]
    assert events.observation_errors()[-1]["operation"] == "emit:model_output_error"


def test_batch_identity_sidecar_does_not_mutate_output():
    output = SimpleNamespace(value=3)
    before = dict(vars(output))
    batch_id = events.register_batch(output)
    assert events.batch_id_for(output) == batch_id
    assert vars(output) == before
    assert events.consume_batch(output) == batch_id
    with pytest.raises(RuntimeError, match="missing same-process"):
        events.batch_id_for(output)
    assert vars(output) == before


def test_trace_error_does_not_change_observed_success(monkeypatch, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    monkeypatch.setenv("M3_TRACE_DIR", str(blocker))
    result = SimpleNamespace(req_ids=[], sampled_token_ids=[])

    class Output:
        def get_output(self):
            return result

    output = Output()
    events.observe_materialization(output, "7:3")
    assert output.get_output() is result
    assert events.observation_errors()[-1]["operation"] == "emit:model_output_materialized"


def test_runner_hooks_preserve_once_only_execute_sample_and_native_materialization(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("M3_TRACE_DIR", str(tmp_path))
    calls = []
    materialized = SimpleNamespace(req_ids=["case::C"], sampled_token_ids=[[7]])

    class AsyncOutput:
        def get_output(self):
            calls.append("get_output")
            return materialized

    output = AsyncOutput()

    def execute(self, scheduled):
        calls.append("execute")
        self.execute_model_state = SimpleNamespace(
            input_batch=SimpleNamespace(
                num_tokens=4,
                num_tokens_after_padding=8,
                req_ids=["case::C"],
            )
        )
        return None

    def sample(self, grammar):
        calls.append("sample")
        self.execute_model_state = None
        return output

    runner_type = type(
        "GPUModelRunner",
        (),
        {
            "__module__": "vllm.v1.worker.gpu.model_runner",
            "execute_model": execute,
            "sample_tokens": sample,
        },
    )
    runner = runner_type()
    scheduled = SimpleNamespace(
        num_scheduled_tokens={"case::C": 4},
        total_num_scheduled_tokens=4,
        scheduled_new_reqs=[SimpleNamespace(req_id="case::C")],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        finished_req_ids=set(),
    )
    batch = events.register_batch(scheduled)
    install_runner_observer(runner)
    install_runner_observer(runner)
    assert runner.execute_model(scheduled) is None
    assert runner.sample_tokens(None) is output
    assert calls == ["execute", "sample"]
    assert output.get_output() is materialized
    assert calls == ["execute", "sample", "get_output"]
    trace = [
        json.loads(line)
        for path in tmp_path.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    returned = next(event for event in trace if event["event"] == "runner_execute_success")
    assert returned["batch_id"] == batch
    assert returned["result_kind"] == "none"
    assert returned["forward_tokens"] == 4
    assert returned["forward_tokens_padded"] == 8
    assert events.consume_batch(scheduled) == batch


def test_runner_execute_exception_remains_identical(monkeypatch, tmp_path):
    monkeypatch.setenv("M3_TRACE_DIR", str(tmp_path))
    calls = []
    failure = ValueError("runner failure")

    def execute(self, output):
        calls.append("execute")
        raise failure

    runner_type = type(
        "GPUModelRunner",
        (),
        {
            "__module__": "vllm.v1.worker.gpu.model_runner",
            "execute_model": execute,
            "sample_tokens": lambda self, grammar: None,
        },
    )
    runner = runner_type()
    scheduled = SimpleNamespace(
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        finished_req_ids={"case::C"},
    )
    events.register_batch(scheduled)
    install_runner_observer(runner)
    with pytest.raises(ValueError) as raised:
        runner.execute_model(scheduled)
    assert raised.value is failure
    assert calls == ["execute"]
    events.consume_batch(scheduled)


def test_async_output_wrapper_does_not_extend_native_tensor_lifetime(monkeypatch, tmp_path):
    monkeypatch.setenv("M3_TRACE_DIR", str(tmp_path))

    class Output:
        def get_output(self):
            return SimpleNamespace(req_ids=[], sampled_token_ids=[])

    enabled = gc.isenabled()
    gc.disable()
    try:
        output = Output()
        reference = weakref.ref(output)
        events.observe_materialization(output, "test:cycle")
        output.get_output()
        del output
        assert reference() is None
    finally:
        if enabled:
            gc.enable()


def test_runner_wrappers_do_not_extend_native_model_lifetime(monkeypatch, tmp_path):
    monkeypatch.setenv("M3_TRACE_DIR", str(tmp_path))
    runner_type = type(
        "GPUModelRunner",
        (),
        {
            "__module__": "vllm.v1.worker.gpu.model_runner",
            "execute_model": lambda self, output: None,
            "sample_tokens": lambda self, grammar: None,
        },
    )
    enabled = gc.isenabled()
    gc.disable()
    try:
        runner = runner_type()
        reference = weakref.ref(runner)
        install_runner_observer(runner)
        del runner
        assert reference() is None
    finally:
        if enabled:
            gc.enable()
