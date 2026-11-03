"""Frozen M5 identities shared by the allocator and the resource auditor."""

import hashlib
import json
import re
from pathlib import Path

TASK = "M5-paired-performance"
BASELINE = "f4d94d3d22a967efeeb3a00bc13f6a6b8591835b"
MODES = ("native", "off", "prefix", "guarded")
CALIBRATION = tuple(f"C{i}" for i in range(1, 7))
SPLIT_C3_ROLES = ("C3-burst", "C3-mixed")
D1_ROLE = "C3-D1"
D2_ROLE = "D2-E-E"
D2R_ROLE = "D2-R"
D2R_PLAN_FILENAME = "m5-d2r.json"
D2R_PREVIOUS_ANALYSIS = "previous-d2-analysis.json"
D2R_OFFICIAL_ANALYSIS_SHA256 = "67ef91b0052f59289ec648f84ed88ce161046e737cc54fdf2668ff51062beb72"
D2R110_ROLE = "D2-R110"
D2R110_PLAN_FILENAME = "m5-d2r110.json"
D2R110_PREVIOUS_ANALYSIS = "previous-d2-r-analysis.json"
DIAGNOSTIC_ROLES = (D1_ROLE, D2_ROLE, D2R_ROLE, D2R110_ROLE)
REVIEWED_DIAGNOSTIC_PLANS = {
    D2R_ROLE: (8, "d2r_plan_sha256", D2R_PLAN_FILENAME),
    D2R110_ROLE: (9, "d2r110_plan_sha256", D2R110_PLAN_FILENAME),
}
SCENARIOS = {"burst": 5, "mixed": 5, "ample": 2, "all-cold": 2, "serial": 2}
JOB_PREFIX = "m5-paired-"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def unique(pairs):
        row = {}
        for key, value in pairs:
            require(key not in row, f"duplicate JSON key: {key}")
            row[key] = value
        return row

    def invalid_constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    return json.loads(
        Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid_constant
    )


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def is_hash(value, size=64):
    return type(value) is str and re.fullmatch(rf"[0-9a-f]{{{size}}}", value) is not None


def id_ok(value):
    return type(value) is str and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9-]{0,63}", value) is not None


def exact_int(value, low, high):
    return type(value) is int and low <= value <= high


def config_matrix(config):
    require(config["task_id"] == TASK and config["baseline_commit"] == BASELINE, "wrong M5 scope")
    require(config["schema_version"] == 1, "unsupported M5 config")
    require(config["modes"] == list(MODES) and config["request_count"] == 500, "wrong matrix scale")
    require(
        config["window"] == 16
        and config["max_wait_ms"] == 500
        and config["production_policy_change"] is False,
        "production scheduler freeze differs",
    )
    rows = config["matrix"]
    expected_ids = {
        f"{scenario}-{repeat:02d}": scenario
        for scenario, count in SCENARIOS.items()
        for repeat in range(1, count + 1)
    }
    require(len(rows) == len(expected_ids), "formal matrix incomplete")
    require({row["id"]: row["scenario"] for row in rows} == expected_ids, "formal matrix differs")
    seeds = {row["seed"] for row in rows}
    require(
        len(seeds) == len(rows) and all(exact_int(seed, 52000, 52999) for seed in seeds),
        "formal seed collision or scope mismatch",
    )
    for row in rows:
        require(len(row["order"]) == 4 and set(row["order"]) == set(MODES), "wrong pair mode order")
    return {row["id"]: row for row in rows}


def check_freeze(freeze, config_path, commit):
    config = read_json(config_path)
    matrix = config_matrix(config)
    require(
        freeze["task_id"] == TASK
        and freeze["schema_version"] == 1
        and freeze["execution_commit"] == commit
        and is_hash(commit, 40)
        and freeze["config_sha256"] == digest(config_path),
        "formal freeze source mismatch",
    )
    require(set(freeze["tables"]) == set(matrix), "frozen arrival tables incomplete")
    require(
        all(is_hash(value) for value in freeze["tables"].values())
        and len(set(freeze["tables"].values())) == len(matrix),
        "frozen table hashes absent or duplicated",
    )
    require(
        set(freeze["calibration_sha256"]) == set(CALIBRATION)
        and all(is_hash(value) for value in freeze["calibration_sha256"].values()),
        "formal freeze lacks reviewed calibration",
    )
    require(set(freeze["rates_rps"]) == set(SCENARIOS) - {"serial"}, "frozen rates incomplete")
    choices = config["calibration"]["candidate_rates"]
    require(
        all(type(value) is int and value in choices for value in freeze["rates_rps"].values()),
        "rate outside calibrated candidates",
    )
    require(exact_int(freeze["serial_interval_ms"], 10, 2000), "invalid serial spacing")
    return matrix


def check_authorization(record, config_path, commit, phase, *, freeze_path=None):
    config = read_json(config_path)
    matrix = config_matrix(config)
    require(phase in {"calibration", "formal"}, "unknown M5 phase")
    require(
        record["schema_version"] == 1
        and record["task_id"] == TASK
        and record["phase"] == phase
        and record["execution_commit"] == commit
        and is_hash(commit, 40)
        and record["config_sha256"] == digest(config_path),
        "authorization source or phase differs",
    )
    require(
        type(record["user_approval_ref"]) is str and bool(record["user_approval_ref"].strip()),
        "record the explicit user's budget approval reference",
    )
    roles = record["roles"]
    qualifying = qualification_roles(config)
    permitted = (*qualifying, *DIAGNOSTIC_ROLES) if phase == "calibration" else matrix
    require(
        type(roles) is list and roles and all(role in permitted for role in roles),
        "authorization roles outside this phase",
    )
    require(
        exact_int(record["max_attempts"], 1, 99)
        and len(roles) == len(set(roles))
        and len(roles) <= record["max_attempts"]
        and record["max_requested_minutes"] == record["max_attempts"] * 20
        and record["minutes_per_attempt"] == 20
        and record["gpus_per_attempt"] == 1,
        "authorization attempt/minute/device limit differs",
    )
    if D1_ROLE in roles:
        require(
            roles == [D1_ROLE] and record["max_attempts"] == 6,
            "D1 is limited to the last original calibration allocation",
        )
    if D2_ROLE in roles:
        require(
            roles == [D2_ROLE] and record["max_attempts"] == 7,
            "D2 is limited to one new diagnostic allocation",
        )
    for role, (limit, key, filename) in REVIEWED_DIAGNOSTIC_PLANS.items():
        if role in roles:
            require(
                roles == [role] and record["max_attempts"] == limit,
                f"{role} is limited to one new diagnostic allocation",
            )
            require(
                is_hash(record.get(key))
                and record[key] == digest(Path(config_path).with_name(filename)),
                f"{role} authorization is detached from the reviewed plan",
            )
        else:
            require(key not in record, f"{role} plan hash outside its exclusive authorization")
    if (
        phase == "calibration"
        and qualifying != CALIBRATION
        and any(role in qualifying for role in roles)
    ):
        require(
            tuple(roles) == qualifying
            and record["max_attempts"] == 16
            and is_hash(record.get("d2r110_analysis_sha256")),
            "split C3 requires its exclusive seven-slot qualification authorization",
        )
    else:
        require(
            "d2r110_analysis_sha256" not in record,
            "D2-R110 result hash outside split C3 qualification authorization",
        )
    if phase == "formal":
        require(freeze_path is not None, "formal authorization requires the frozen tables")
        require(record["freeze_sha256"] == digest(freeze_path), "authorization freeze hash differs")
        check_freeze(read_json(freeze_path), config_path, commit)
    else:
        require(freeze_path is None and record["freeze_sha256"] is None, "phase freeze mismatch")
    return roles


def qualification_roles(config):
    if config["calibration"].get("split_c3") is True:
        roles = (*CALIBRATION[:2], *SPLIT_C3_ROLES, *CALIBRATION[3:])
        require(
            config["budget"]["split_calibration_allocations"] == len(roles),
            "split C3 physical allocation count differs",
        )
        return roles
    return CALIBRATION


def bind_d2r_inputs(record, plan, previous):
    require(
        record["roles"] == [D2R_ROLE] and digest(plan) == record["d2r_plan_sha256"],
        "D2-R staged plan differs from authorization",
    )
    return bind_retained_d2_analysis(read_json(plan)["retained_D2"], previous, D2R_ROLE)


def bind_retained_d2_analysis(expected, previous, role):
    require(digest(previous) == expected["analysis_sha256"], f"{role} retained D2 analysis differs")
    predecessor = read_json(previous)
    require(
        expected["status"] == predecessor["status"] == "diagnostic-incomplete"
        and expected["proposal_sha256"] == predecessor["proposal_sha256"],
        f"{role} predecessor is not the retained incomplete D2",
    )
    return expected["analysis_sha256"]


def bind_d2r110_inputs(record, plan, previous, previous_d2r):
    require(
        record["roles"] == [D2R110_ROLE] and digest(plan) == record["d2r110_plan_sha256"],
        "D2-R110 staged plan differs from authorization",
    )
    proposal = read_json(plan)
    d2_sha = bind_retained_d2_analysis(proposal["retained_D2"], previous, D2R110_ROLE)
    retained = proposal["retained_D2R"]
    require(
        retained["analysis_sha256"] == digest(previous_d2r) == D2R_OFFICIAL_ANALYSIS_SHA256,
        "D2-R110 retained D2-R analysis differs",
    )
    predecessor = read_json(previous_d2r)
    require(
        predecessor["role"] == D2R_ROLE
        and retained["status"] == predecessor["status"] == "diagnostic-incomplete"
        and retained["proposal_sha256"] == predecessor["proposal_sha256"],
        "D2-R110 predecessor is not the retained incomplete D2-R",
    )
    return d2_sha, retained["analysis_sha256"]


def formal_specs(config, freeze, role):
    pair = config_matrix(config)[role]
    scenario = pair["scenario"]
    return [
        {
            "id": mode,
            "scenario": scenario,
            "seed": pair["seed"],
            "count": config["request_count"],
            "mode": mode,
            "tier": "light",
            "rate_rps": freeze["rates_rps"].get(scenario),
            "interval_ms": freeze["serial_interval_ms"] if scenario == "serial" else None,
            "mechanism": None,
        }
        for mode in pair["order"]
    ]
