"""Keep coordination status evidence-based rather than a decorative progress bar."""

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from vllm_prefix_scheduler.progress import (
    milestone_status,
    recent_events,
    render_state,
    validate_progress,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def progress():
    return json.loads((ROOT / "progress/project-status.json").read_text())


def test_progress_evidence_dependencies_and_generated_files_are_current(progress):
    counts = validate_progress(progress, ROOT)
    assert counts["done"] > 0
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_progress.py"), "--check"],
        check=True,
        capture_output=True,
        text=True,
    )


def test_completion_requires_evidence(progress):
    modified = deepcopy(progress)
    task = next(task for task in modified["tasks"] if task["status"] == "done")
    task["evidence"] = []
    with pytest.raises(ValueError, match="Completion requires evidence"):
        validate_progress(modified)


def test_dependent_work_cannot_be_claimed_complete_early(progress):
    modified = deepcopy(progress)
    task = next(task for task in modified["tasks"] if task["id"] == "M2-off-config")
    task["status"] = "done"
    task["evidence"] = [{"label": "unverified", "url": "https://example.invalid"}]
    with pytest.raises(ValueError, match="Unsatisfied dependency"):
        validate_progress(modified)


def test_cyclic_dependencies_are_rejected(progress):
    modified = deepcopy(progress)
    one = next(task for task in modified["tasks"] if task["id"] == "M5-workloads")
    two = next(task for task in modified["tasks"] if task["id"] == "M5-calibration")
    one["depends_on"] = [two["id"]]
    two["depends_on"] = [one["id"]]
    with pytest.raises(ValueError, match="cycle"):
        validate_progress(modified)


def test_restored_checkout_evidence_preserves_provenance_and_release():
    directory = ROOT / "evidence/2026-11-01-checkout-replay"
    receipt = json.loads((directory / "workspace-verification.json").read_text())
    assert receipt["tracked_worktree_clean"]
    assert receipt["all_resource_links_ignored_by_git"]
    assert receipt["development_gpu_environment_isolated"]
    assert receipt["allocation_released"]
    assert receipt["post_process_gpu_memory_mib"] == 1
    assert receipt["native_replay"]["status"] == "passed"
    provenance = json.loads((directory / "provenance.json").read_text())
    for name, expected in provenance["exported_file_sha256"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected


def test_fully_accepted_project_has_a_terminal_state(progress):
    modified = deepcopy(progress)
    for task in modified["tasks"]:
        task["status"] = "done"
        task["evidence"] = task["evidence"] or [
            {"label": "synthetic terminal-state fixture", "url": "https://example.invalid"}
        ]
    modified["focus_task"] = None
    assert validate_progress(modified)["done"] == len(modified["tasks"])
    assert "所有已定义验收任务均已完成" in render_state(modified)


def test_partial_phase_is_not_reported_as_not_started(progress):
    modified = deepcopy(progress)
    task = next(task for task in modified["tasks"] if task["id"] == "M1-call-chain")
    task["status"] = "done"
    assert milestone_status(modified, "M1") == "partial"


def test_appended_new_event_is_shown_first_in_generated_state(progress):
    modified = deepcopy(progress)
    modified["events"].append(
        {
            "date": "2026-11-02T01:00:00+08:00",
            "title": "new event fixture",
            "description": "synthetic ordering test",
        }
    )
    assert recent_events(modified)[0]["title"] == "new event fixture"
    state = render_state(modified)
    assert state.index("new event fixture") < state.index(progress["events"][0]["title"])
