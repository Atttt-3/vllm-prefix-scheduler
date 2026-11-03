"""D6 token-gap bins and archived signature fail-closed checks."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

T0 = 10_000_000_000
MS = 1_000_000
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def d6(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("m5_d6_signature")


def client(*streams):
    return {
        "phases": {"measure_start_ns": T0},
        "requests": [
            {
                "request_id": str(index),
                "chunk_tokens": [
                    {"at_ns": T0 + offset * MS, "token_ids": [count]}
                    for count, offset in enumerate(offsets)
                ],
            }
            for index, offsets in enumerate(streams)
        ],
    }


def test_threshold_inclusive_and_trigger_requires_two_complete_adjacent_bins(d6):
    stream = [0, 30, 60, 90, 120, 150, 500, 530, 560, 590, 620, 650]
    raw = client(stream, stream)
    early = d6.client_gap_signature(raw, cutoff_ns=T0 + 2 * d6.BIN_NS - 1)
    assert early["candidate"] == "unverified"
    assert early["bins"][0]["above_threshold"]
    assert early["bins"][1]["median_ns"] == d6.MEDIAN_GATE_NS
    assert not early["bins"][1]["complete"] and not early["bins"][1]["eligible"]
    result = d6.client_gap_signature(raw, cutoff_ns=T0 + 2 * d6.BIN_NS)
    assert result["candidate"] == "detected"
    assert result["first_trigger_bin_indices"] == [0, 1]
    assert result["first_trigger_offset_ns"] == 2 * d6.BIN_NS
    assert [row["interval_count"] for row in result["bins"][:2]] == [10, 12]


def test_coalesced_tokens_and_exact_bin_boundary_use_client_timestamp(d6):
    raw = client([0, 490, 500, 3000], [490, 500])
    raw["requests"][0]["chunk_tokens"][0]["token_ids"].append(42)
    result = d6.client_gap_signature(raw)
    assert result["coalesced_zero_interval_count"] == 1
    assert result["bins"][0]["median_ns"] == 245 * MS
    assert result["bins"][0]["distinct_request_count"] == 1
    assert result["bins"][1]["median_ns"] == 10 * MS
    assert result["bins"][1]["distinct_request_count"] == 2
    assert sum(row["interval_count"] for row in result["bins"]) == 4
    assert result["candidate"] == "unverified"


def test_no_support_from_cross_id_or_nonadjacent_intervals(d6):
    one = client(range(0, 421, 30))
    result = d6.client_gap_signature(one)
    assert result["bins"][0]["interval_count"] == 14
    assert not result["bins"][0]["eligible"]

    stream = [
        0,
        35,
        70,
        105,
        140,
        175,
        500,
        510,
        520,
        530,
        540,
        550,
        1000,
        1035,
        1070,
        1105,
        1140,
        1175,
    ]
    raw = client(stream, stream)
    result = d6.client_gap_signature(raw, cutoff_ns=T0 + d6.HORIZON_NS)
    assert result["candidate"] == "not_detected"
    assert [row["above_threshold"] for row in result["bins"][:3]] == [True, False, True]


@pytest.mark.parametrize("defect", ("duplicate", "reverse", "empty", "negative", "cutoff"))
def test_bad_client_token_evidence_rejected(d6, defect):
    raw = client([0, 1, 2], [3, 4])
    cutoff = None
    if defect == "duplicate":
        raw["requests"][0]["request_id"] = raw["requests"][1]["request_id"]
    elif defect == "reverse":
        raw["requests"][0]["chunk_tokens"].reverse()
    elif defect == "empty":
        raw["requests"][0]["chunk_tokens"][0]["token_ids"] = []
    elif defect == "negative":
        raw["requests"][0]["chunk_tokens"][0]["token_ids"] = [-1]
    else:
        cutoff = T0 - 1
    with pytest.raises(ValueError, match="D6"):
        d6.client_gap_signature(raw, cutoff_ns=cutoff)


def test_cli_refuses_changed_archived_review_before_writing(tmp_path):
    review = tmp_path / "fake-analysis.json"
    review.write_text('{"role":"D5-R"}\n')
    output = tmp_path / "report.json"
    child = subprocess.run(
        [
            sys.executable,
            "-B",
            str(ROOT / "scripts/m5_d6_signature.py"),
            "--attempt",
            str(tmp_path / "attempt"),
            "--analysis",
            str(review),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert child.returncode == 1 and child.stderr == "D6 retained D5-R analysis differs\n"
    assert not output.exists()
