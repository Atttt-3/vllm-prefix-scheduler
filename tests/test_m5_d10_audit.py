"""Independent D10 placement and intervention fail-closed controls."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from m5_d10_audit import observed_placement, placement_assessment  # noqa: E402

ORDER = {"E1": "split", "E2": "inherited", "E3": "inherited", "E4": "split"}
T0 = 10_000_000_000
DRAIN = T0 + 8_500_000_000


def placement_client(mode):
    roles = {
        "allocated": list(range(8)),
        "client": [0, 1],
        "engine": list(range(2, 8)),
        "physical_cores": [
            {"package": 0, "core": index, "numa_node": 0, "cpus": [index]} for index in range(8)
        ],
    }

    def snapshot(start, tick):
        result = {"started_ns": start, "ended_ns": start + 1}
        for role, pid, parent in (("client", 10, 7), ("engine", 11, 10)):
            mask = roles["allocated"] if mode == "inherited" else roles[role]
            result[role] = {
                "pid": pid,
                "parent_pid": parent,
                "pgid": 9,
                "start_ticks": 100 + pid,
                "threads": [
                    {
                        "tid": pid,
                        "start_ticks": 100 + pid,
                        "affinity": list(mask),
                        "processor": mask[0],
                        "schedstat": {
                            "runtime_ns": 40 + tick,
                            "runnable_wait_ns": 10 + tick,
                            "timeslices": tick,
                        },
                    }
                ],
            }
        return result

    return {
        "clock": {"pid": 10},
        "phases": {"measure_start_ns": T0, "drain_end_ns": DRAIN},
        "d10_placement": {
            "schema_version": 1,
            "mode": mode,
            "roles": roles,
            "setup": snapshot(T0 - 1000, 1),
            "before_measure": snapshot(T0 - 100, 2),
            "after_measure": snapshot(DRAIN + 1, 4),
        },
    }


def audit_child(path, client, identity):
    path.write_text(json.dumps(client))
    return subprocess.run(
        [
            sys.executable,
            "-c",
            """import json, sys
from pathlib import Path
from m5_d10_audit import observed_placement
client = json.loads(Path(sys.argv[1]).read_text())
order = {"E1":"split","E2":"inherited","E3":"inherited","E4":"split"}
try:
    print(json.dumps(observed_placement(client, {"clock":{"pid":11}}, sys.argv[2], order,
        thermal={"conditioning_start_ns":9_999_999_500}, leaf_end_ns=18_500_000_100, pgid=9)))
except (ValueError, KeyError, TypeError) as error:
    sys.exit(str(error))
""",
            str(path),
            identity,
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "scripts")},
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda p: p.update(mode="inherited"), "frozen ABBA"),
        (lambda p: p["roles"]["physical_cores"][0]["cpus"].append(2), "SMT sibling"),
        (lambda p: p["setup"]["engine"].update(parent_pid=100), "process identity"),
        (
            lambda p: p["before_measure"]["client"]["threads"][0].update(affinity=[2, 3]),
            "thread affinity",
        ),
        (lambda p: p["after_measure"].update(started_ns=T0), "snapshot chronology"),
        (
            lambda p: p["after_measure"]["engine"]["threads"][0]["schedstat"].update(runtime_ns=1),
            "schedstat counters",
        ),
    ),
)
def test_d10_real_child_accepts_placement_and_rejects_tamper(tmp_path, mutate, message):
    client = placement_client("split")
    path = tmp_path / "client.json"
    positive = audit_child(path, client, "E1")
    assert positive.returncode == 0, positive.stderr
    assert json.loads(positive.stdout)["schedstat_delta"]["engine"]["runnable_wait_ns"] == 2
    mutate(client["d10_placement"])
    failed = audit_child(path, client, "E1")
    assert failed.returncode == 1 and message in failed.stderr


def test_inherited_uses_entire_cpu_mask():
    result = observed_placement(
        placement_client("inherited"), {"clock": {"pid": 11}}, "E2", ORDER, leaf_end_ns=DRAIN + 3
    )
    assert result["verified"]


def comparison_fixture():
    gates = {
        "overhead_ttft_ns": 2_000_000,
        "overhead_ttft_rel": 0.05,
        "overhead_throughput_rel": 0.05,
    }
    clients = {}
    runs = {}
    for leaf, p95 in (("E1", 605), ("E2", 800), ("E3", 610), ("E4", 608)):
        requests = []
        timing = {}
        for index in range(60):
            identity = f"request-{index:06d}"
            start = T0 + (index * 160_000_000 if index < 18 else 6_000_000_000 + index * 1_000_000)
            stride = 35_000_000 if leaf == "E2" else 22_000_000
            chunks = [{"at_ns": start + i * stride + 1, "token_ids": [3]} for i in range(16)]
            first = chunks[0]["at_ns"]
            requests.append(
                {
                    "request_id": identity,
                    "submit_ns": start,
                    "first_token_ns": first,
                    "cached_tokens": 0,
                    "token_ids": [3] * len(chunks),
                    "chunk_tokens": chunks,
                }
            )
            timing[identity] = {
                "submit_ns": start,
                "planned_ns": start,
                "first_components_ns": {
                    "submit_to_native_queued": 0,
                    "native_queued_to_first_scheduled": 0,
                },
            }
        clients[leaf] = {
            "phases": {"measure_start_ns": T0, "drain_end_ns": DRAIN},
            "requests": requests,
        }
        runs[leaf] = {
            "endpoint_evidence_complete": True,
            "native_timing": {"status": "complete", "clock": {"verified": True}, "per_id": timing},
            "placement": {"verified": True},
            "metrics": {
                "strata": {"all": {"ttft_ns": {"p95": p95 * 1_000_000}}},
                "throughput": {
                    "last_terminal_requests_per_s": 7.3,
                    "last_terminal_tokens_per_s": 116.8,
                },
            },
        }
    return runs, clients, gates


def test_d10_strict_improvement_and_inconclusive_guards():
    runs, clients, gates = comparison_fixture()
    supported = placement_assessment(runs, clients, gates, ORDER, thermal_verified=True)
    assert supported["status"] == "supported"
    assert supported["checks"]["control_early_slow"] == ["E2"]
    assert supported["pairs"]["E2/E3"]["ttft_within_tolerance"] is False
    assert supported["pairs"]["E1/E4"]["within_tolerance"] is True

    modified = copy.deepcopy(runs)
    modified["E3"]["metrics"]["strata"]["all"]["ttft_ns"]["p95"] = 790_000_000
    assert (
        placement_assessment(modified, clients, gates, ORDER, thermal_verified=True)["status"]
        == "inconclusive"
    )

    modified = copy.deepcopy(runs)
    modified["E4"]["metrics"]["throughput"]["last_terminal_tokens_per_s"] = 110
    assert (
        placement_assessment(modified, clients, gates, ORDER, thermal_verified=True)["status"]
        == "not_supported"
    )

    assert (
        placement_assessment(runs, clients, gates, ORDER, thermal_verified=False)["status"]
        == "unresolved"
    )
    assert (
        placement_assessment(
            {k: v for k, v in runs.items() if k != "E4"},
            {k: v for k, v in clients.items() if k != "E4"},
            gates,
            ORDER,
            thermal_verified=True,
        )["status"]
        == "unresolved"
    )
