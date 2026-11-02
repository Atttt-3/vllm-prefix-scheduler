"""Transport binding is optional for legacy runs and fail-closed for campaigns."""

import json

from vllm_prefix_scheduler import m3_events


def test_legacy_event_has_no_campaign_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("M3_TRACE_DIR", str(tmp_path))
    monkeypatch.delenv("M3_CAMPAIGN_ID", raising=False)
    monkeypatch.delenv("M3_CAMPAIGN_SHA256", raising=False)
    m3_events.emit("transport_test")
    event = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    assert "campaign_id" not in event
    assert "campaign_sha256" not in event


def test_campaign_event_is_bound_without_strategy_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("M3_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("M3_CAMPAIGN_ID", "test-campaign")
    monkeypatch.setenv("M3_CAMPAIGN_SHA256", "a" * 64)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    m3_events.emit("transport_test", marker=3)
    event = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    assert event["campaign_id"] == "test-campaign"
    assert event["campaign_sha256"] == "a" * 64
    assert event["campaign_job_id"] == "123"
    assert event["marker"] == 3


def test_incomplete_binding_is_sticky_evidence_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(m3_events, "_errors", [])
    monkeypatch.setenv("M3_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("M3_CAMPAIGN_ID", "test-campaign")
    monkeypatch.delenv("M3_CAMPAIGN_SHA256", raising=False)
    assert m3_events.emit("transport_test") is None
    assert not list(tmp_path.glob("*.jsonl"))
    assert m3_events.observation_errors()[0]["operation"] == "emit:transport_test"


def test_caller_cannot_spoof_campaign_binding(tmp_path, monkeypatch):
    monkeypatch.setattr(m3_events, "_errors", [])
    monkeypatch.setenv("M3_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("M3_CAMPAIGN_ID", "test-campaign")
    monkeypatch.setenv("M3_CAMPAIGN_SHA256", "a" * 64)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    m3_events.emit("transport_test", campaign_id="another-campaign")
    assert not list(tmp_path.glob("*.jsonl"))
    assert m3_events.observation_errors()
