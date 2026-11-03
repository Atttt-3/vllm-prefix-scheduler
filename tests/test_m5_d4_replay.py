"""Exercise D4 rate and heat routing without an engine or allocation."""

import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_prefix_scheduler.m5_campaign import D4_ROLE, D5_ROLE, D5R_ROLE, TASK, digest, read_json
from vllm_prefix_scheduler.m5_d2 import D3_PROPOSAL_NAME
from vllm_prefix_scheduler.m5_d4 import (
    D4_INPUT_6_NAME,
    D4_INPUT_8_NAME,
    D4_PREVIOUS_ANALYSIS,
    D4_PROPOSAL_NAME,
)
from vllm_prefix_scheduler.m5_d5 import D5_PREVIOUS_ANALYSIS, D5_PROPOSAL_NAME, d5_profile
from vllm_prefix_scheduler.m5_d5r import D5R_PREVIOUS_ANALYSIS, D5R_PROPOSAL_NAME

ROOT = Path(__file__).resolve().parents[1]
ORDER = {"thermal-pilot": 8, "E1": 8, "E2": 6, "E3": 6, "E4": 8}


def write(path, value):
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def staged(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    config = ROOT / "configs/m5-performance.json"
    settings = read_json(config)
    base = {
        "task_id": TASK,
        "tokenizer_revision": settings["tokenizer_revision"],
        "scenario": "mixed",
        "seed": 51213,
        "mechanism": None,
        "requests": [{}],
        "sampling": {"max_tokens": 16},
    }
    tables = {
        rate: write(tmp_path / name, {**base, "rate_rps": rate})
        for rate, name in ((8, D4_INPUT_8_NAME), (6, D4_INPUT_6_NAME))
    }
    proposal = {
        "target": {
            "scenario": base["scenario"],
            "seed": base["seed"],
            "count_per_engine": len(base["requests"]),
            "preheat_rate_rps": 8,
            "measure_order_rps": [8, 6, 6, 8],
            "arrival_tables": {
                str(rate): {"sha256": digest(path)} for rate, path in tables.items()
            },
        },
    }
    inherited = {
        "target": {"fresh_engines": list(ORDER)},
        "native_timing": {"schema_version": 1, "source_sha256": {}},
    }
    return SimpleNamespace(
        config=config,
        tables=tables,
        proposal=write(tmp_path / D4_PROPOSAL_NAME, proposal),
        inherited=write(tmp_path / D3_PROPOSAL_NAME, inherited),
        previous=write(tmp_path / D4_PREVIOUS_ANALYSIS, {}),
        original=write(tmp_path / "original-c3-result.json", {}),
    )


def test_d4_cli_routes_measured_six_rps_and_frozen_eight_rps_preheat(staged, tmp_path, monkeypatch):
    replay = importlib.import_module("replay_m5")
    monkeypatch.setattr(replay, "os", SimpleNamespace(environ={}))
    monkeypatch.setattr(replay, "d4_plan", lambda *args: tuple(map(read_json, args)))
    monkeypatch.setattr(
        replay, "_binding", lambda *args: {"project_commit": "a" * 40, "job_id": "1"}
    )
    monkeypatch.setattr(replay, "bind_loaded_sources", lambda *args: object)
    calls = []

    async def fake_run(
        args, config, workload, client, *, native_stats_type, preheat_workload, d5_expected_profile
    ):
        assert d5_expected_profile is None
        calls.append((workload["rate_rps"], preheat_workload["rate_rps"], native_stats_type))

    monkeypatch.setattr(replay, "_run", fake_run)
    output = tmp_path / "raw"
    argv = [
        "replay_m5.py",
        "--config",
        str(staged.config),
        "--input",
        str(staged.tables[6]),
        "--model",
        str(tmp_path / "unused-model"),
        "--output",
        str(output),
        "--mode",
        "guarded",
        "--tier",
        "endpoint",
        "--phase",
        "calibration",
        "--pair-id",
        "E2",
        "--d2-role",
        "measure",
        "--d2-plan",
        str(staged.inherited),
        "--d2-samples",
        str(tmp_path / "unused-samples"),
        "--d2-target",
        str(tmp_path / "unused-pilot"),
        "--d4-plan",
        str(staged.proposal),
        "--d4-preheat-input",
        str(staged.tables[8]),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    replay.main()
    assert calls == [(6, 8, object)]
    client = read_json(output / "client.json")
    assert client["input_sha256"] == digest(staged.tables[6])
    assert client["native_metrics_schema_version"] == 1

    argv[argv.index("--d4-preheat-input") + 1] = str(staged.tables[6])
    with pytest.raises(SystemExit) as error:
        replay.main()
    assert error.value.code == 2
    argv[argv.index("--d4-preheat-input") + 1] = str(staged.tables[8])
    argv[argv.index("--pair-id") + 1] = "E4"
    with pytest.raises(SystemExit) as error:
        replay.main()
    assert error.value.code == 2
    assert calls == [(6, 8, object)]


@pytest.mark.parametrize("role", (D4_ROLE, D5_ROLE, D5R_ROLE))
def test_thermal_node_runs_abba_and_seals_diagnostic_inputs(staged, tmp_path, monkeypatch, role):
    runner = importlib.import_module("run_m5_allocation")
    role_name = role.lower()
    planner_name = role_name.replace("-", "")
    attempt = f"{role_name}-01"
    output = tmp_path / "runs" / TASK / attempt
    d5r = role == D5R_ROLE
    d5 = d5r or role == D5_ROLE
    proposal = staged.proposal
    previous = staged.previous
    if d5:
        proposal = tmp_path / D5_PROPOSAL_NAME
        proposal.write_bytes((ROOT / "configs/m5-d5.json").read_bytes())
        previous = write(tmp_path / D5_PREVIOUS_ANALYSIS, {})
        (tmp_path / "input.json").write_bytes(staged.tables[8].read_bytes())
        monkeypatch.setattr(runner, "D1_INPUT_SHA", digest(staged.tables[8]))
    d5_plan = proposal
    if d5r:
        proposal = tmp_path / D5R_PROPOSAL_NAME
        proposal.write_bytes((ROOT / "configs/m5-d5r.json").read_bytes())
        previous_d5 = write(tmp_path / D5R_PREVIOUS_ANALYSIS, {})
    authorization = write(
        tmp_path / "authorization.json", {f"{planner_name}_plan_sha256": digest(proposal)}
    )
    plan = read_json(d5_plan)
    specs = [
        {
            "id": identity,
            "rate_rps": 8 if d5 else rate,
            "mode": "guarded",
            "tier": "endpoint",
            "seed": 51213,
            "mechanism": None,
            **(d5_profile(plan, identity) if d5 else {}),
        }
        for identity, rate in ORDER.items()
    ]
    calls = []

    def popen(command, **kwargs):
        separator = command.index("--")
        collect, child = command[:separator], command[separator + 1 :]
        write(Path(child[2]), {})
        raw = Path(child[child.index("--output") + 1])
        raw.mkdir()
        status = "target-frozen" if raw.parent.name == "thermal-pilot" else "admitted"
        write(raw / "d2-thermal.json", {"status": status})
        calls.append((collect, child))
        return SimpleNamespace(pid=123, wait=lambda **_: 0)

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "check_authorization", lambda *args, **kwargs: [role])
    monkeypatch.setattr(runner, f"{planner_name}_specs", lambda *args: specs)
    monkeypatch.setattr(runner, "collect_sources", lambda *args: {})
    monkeypatch.setattr(runner, "owner_group", lambda *args: 124)
    monkeypatch.setattr(runner, "owned_processes", lambda *args: [])
    monkeypatch.setattr(runner, "snapshot", lambda **_: {})
    monkeypatch.setattr(
        runner,
        "subprocess",
        SimpleNamespace(
            check_output=lambda argv, **_: "a" * 40 if "rev-parse" in argv else b"",
            Popen=popen,
            STDOUT=None,
        ),
    )
    monkeypatch.setattr(
        runner, "signal", SimpleNamespace(SIGTERM=1, SIGINT=2, SIG_IGN=3, signal=lambda *args: None)
    )
    monkeypatch.setattr(
        runner,
        "os",
        SimpleNamespace(environ={"SLURM_JOB_ID": "123"}, fsencode=os.fsencode, uname=os.uname),
    )
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(
        runner,
        "auxiliary_command",
        lambda command, timeout: write(Path(command[command.index("--output") + 1]), {}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_m5_allocation.py",
            str(staged.config),
            str(authorization),
            "calibration",
            role,
            attempt,
            str(output),
            str(tmp_path),
            str(tmp_path),
            str(proposal),
        ],
    )
    runner.main()
    assert len(calls) == len(ORDER)
    for (collect, child), (identity, rate) in zip(calls, ORDER.items(), strict=True):
        leaf = output / identity
        assert digest(leaf / "input.json") == digest(staged.tables[8 if d5 else rate])
        assert child[child.index("--d2-plan") + 1] == str(staged.inherited)
        assert child[child.index(f"--{planner_name}-plan") + 1] == str(proposal)
        if d5r:
            assert child[child.index("--d5-plan") + 1] == str(d5_plan)
        if not d5:
            assert child[child.index("--d4-preheat-input") + 1] == str(staged.tables[8])
        configs = [
            Path(value)
            for previous, value in zip(collect[:-1], collect[1:], strict=True)
            if previous == "--config"
        ]
        assert configs == [
            staged.config,
            leaf / "input.json",
            proposal,
            *([d5_plan] if d5r else []),
            staged.inherited,
            *([] if d5 else [staged.tables[8]]),
            previous,
            *([previous_d5] if d5r else []),
            staged.original,
            *(
                []
                if identity == "thermal-pilot"
                else [output / "thermal-pilot/raw/d2-thermal.json"]
            ),
        ]
    result = read_json(output / f"{role_name}-result.json")
    assert result["status"] == "diagnostic-complete"
    assert result["audited_runs"] == 4
    if d5:
        assert result["retained_d4_analysis_sha256"] == digest(previous)
        if d5r:
            assert result["retained_d5_analysis_sha256"] == digest(previous_d5)
            assert result["inherited_d5_proposal_sha256"] == digest(d5_plan)
        index = read_json(output / "index.json")
        assert [row["max_num_active_seqs"] for row in index["runs"]] == [4, 4, 5, 5, 4]
    else:
        assert result["retained_d3_analysis_sha256"] == digest(previous)
        assert result["measure_order_rps"] == [8, 6, 6, 8]
