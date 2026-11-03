"""Keep D1 limited to its approved remaining calibration slot."""

from pathlib import Path

import pytest

from vllm_prefix_scheduler.m5_campaign import (
    CALIBRATION,
    D1_ROLE,
    TASK,
    check_authorization,
    digest,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs/m5-performance.json"
COMMIT = "a" * 40


def test_d1_budget_is_exclusive_and_not_a_formal_prerequisite():
    approval = {
        "schema_version": 1,
        "task_id": TASK,
        "phase": "calibration",
        "execution_commit": COMMIT,
        "config_sha256": digest(CONFIG),
        "user_approval_ref": "explicit one-slot D1 approval",
        "roles": [D1_ROLE],
        "max_attempts": 6,
        "max_requested_minutes": 120,
        "minutes_per_attempt": 20,
        "gpus_per_attempt": 1,
        "freeze_sha256": None,
    }
    assert check_authorization(approval, CONFIG, COMMIT, "calibration") == [D1_ROLE]
    assert CALIBRATION == ("C1", "C2", "C3", "C4", "C5", "C6")
    for unauthorized in (
        {"roles": ["C3-burst", D1_ROLE]},
        {"max_attempts": 7, "max_requested_minutes": 140},
    ):
        with pytest.raises(ValueError, match="last original calibration allocation"):
            check_authorization({**approval, **unauthorized}, CONFIG, COMMIT, "calibration")
