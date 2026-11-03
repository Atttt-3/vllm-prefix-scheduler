"""Limit D2 to its one new serial allocation and keep formal stages separate."""

from pathlib import Path

import pytest

from vllm_prefix_scheduler.m5_campaign import (
    CALIBRATION,
    D2_ROLE,
    TASK,
    check_authorization,
    digest,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs/m5-performance.json"
COMMIT = "b" * 40


def test_d2_has_one_new_exclusive_slot():
    approval = {
        "schema_version": 1,
        "task_id": TASK,
        "phase": "calibration",
        "execution_commit": COMMIT,
        "config_sha256": digest(CONFIG),
        "user_approval_ref": "explicit one-slot D2 approval",
        "roles": [D2_ROLE],
        "max_attempts": 7,
        "max_requested_minutes": 140,
        "minutes_per_attempt": 20,
        "gpus_per_attempt": 1,
        "freeze_sha256": None,
    }
    assert check_authorization(approval, CONFIG, COMMIT, "calibration") == [D2_ROLE]
    assert D2_ROLE not in CALIBRATION
    for invalid in (
        {"roles": ["C4", D2_ROLE]},
        {"max_attempts": 8, "max_requested_minutes": 160},
        {"max_attempts": 6, "max_requested_minutes": 120},
    ):
        with pytest.raises(ValueError, match="one new diagnostic allocation"):
            check_authorization({**approval, **invalid}, CONFIG, COMMIT, "calibration")
