"""Exercise frozen formal table bindings without a model runtime or allocation."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_m5_workload import encode

from vllm_prefix_scheduler.m5_campaign import CALIBRATION, TASK, config_matrix, digest, read_json
from vllm_prefix_scheduler.m5_contract import AuditError, bind_formal_calibration
from vllm_prefix_scheduler.m5_workload import prepare_workload

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/m5-performance.json"
RATES = {"burst": 3, "mixed": 8, "ample": 8, "all-cold": 8}
SERIAL_MS = 120

TABLE_AUDIT = """
import sys
from pathlib import Path
from vllm_prefix_scheduler.m5_campaign import check_freeze, read_json
from vllm_prefix_scheduler.m5_contract import validate_formal_table

config_path, freeze_path, table_path = map(Path, sys.argv[1:4])
freeze = read_json(freeze_path)
try:
    matrix = check_freeze(freeze, config_path, freeze["execution_commit"])
    validate_formal_table(table_path, read_json(config_path), matrix[sys.argv[4]], freeze)
except (KeyError, OSError, TypeError, ValueError) as error:
    sys.exit(str(error))
"""


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def freeze_record(matrix):
    return {
        "task_id": TASK,
        "schema_version": 1,
        "execution_commit": "a" * 40,
        "config_sha256": digest(CONFIG),
        "tables": {role: f"{i:064x}" for i, role in enumerate(matrix)},
        "calibration_sha256": dict.fromkeys(CALIBRATION, "b" * 64),
        "rates_rps": RATES.copy(),
        "serial_interval_ms": SERIAL_MS,
    }


@pytest.mark.parametrize(
    ("scenario", "change", "error"),
    (
        ("burst", "none", ""),
        ("mixed", "none", ""),
        ("serial", "none", ""),
        ("mixed", "wrong_rate", "rate or interval differs from freeze"),
        ("serial", "wrong_spacing", "rate or interval differs from freeze"),
        ("mixed", "offset", "arrival offsets differ from absolute rate"),
        ("burst", "unbound_bytes", "formal table hash differs"),
        ("serial", "off_grid", "outside the calibrated grid"),
    ),
)
def test_formal_input_subprocess(tmp_path, scenario, change, error):
    config = read_json(CONFIG)
    matrix = config_matrix(config)
    freeze = freeze_record(matrix)
    role = f"{scenario}-01"
    pair = matrix[role]
    rate, interval = (None, SERIAL_MS) if scenario == "serial" else (RATES[scenario], None)
    if change == "wrong_rate":
        rate = 6
    elif change == "wrong_spacing":
        interval = 130
    elif change == "off_grid":
        interval = freeze["serial_interval_ms"] = 125
    table = prepare_workload(
        config, encode, scenario, pair["seed"], rate=rate, interval_ms=interval
    )
    if change == "offset":
        table["requests"][1]["offset_ns"] += 1
    path = write(tmp_path / f"{role}.json", table)
    if change != "unbound_bytes":
        freeze["tables"][role] = digest(path)
    freeze_path = write(tmp_path / "freeze.json", freeze)
    proc = subprocess.run(
        [sys.executable, "-c", TABLE_AUDIT, str(CONFIG), str(freeze_path), str(path), role],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == bool(error), proc.stderr
    assert error in proc.stderr


def test_formal_freeze_retains_calibrated_parameters_and_c6_hashes(tmp_path):
    freeze = freeze_record(config_matrix(read_json(CONFIG)))
    c2, c6 = tmp_path / "c2", tmp_path / "c6"
    write(c2 / "node/calibration-result.json", {"selected_rates": {"burst": 3, "mixed": 8}})
    write(
        c6 / "node/calibration-result.json",
        {"selected_rates": {"ample": 8, "all-cold": 8}, "serial_interval_ms": SERIAL_MS},
    )
    write(c6 / "node/allocation.json", {"formal_table_sha256": freeze["tables"]})
    bind_formal_calibration(freeze, c2, c6)
    with pytest.raises(AuditError, match="detached from C2/C6 calibration"):
        bind_formal_calibration({**freeze, "serial_interval_ms": 130}, c2, c6)
    with pytest.raises(AuditError, match="detached from C2/C6 calibration"):
        bind_formal_calibration({**freeze, "rates_rps": {**RATES, "mixed": 6}}, c2, c6)
    with pytest.raises(AuditError, match="detached from the C6 allocation"):
        bind_formal_calibration(
            {**freeze, "tables": {**freeze["tables"], "mixed-01": "c" * 64}}, c2, c6
        )
