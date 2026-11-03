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
QUALIFICATION_SUCCESSOR_PROTOCOL = "same-condition-ee-v1"
D1_ROLE = "C3-D1"
D2_ROLE = "D2-E-E"
D2R_ROLE = "D2-R"
D2R_PLAN_FILENAME = "m5-d2r.json"
D2R_PREVIOUS_ANALYSIS = "previous-d2-analysis.json"
D2R_OFFICIAL_ANALYSIS_SHA256 = "67ef91b0052f59289ec648f84ed88ce161046e737cc54fdf2668ff51062beb72"
D2R110_ROLE = "D2-R110"
D2R110_PLAN_FILENAME = "m5-d2r110.json"
D2R110_PREVIOUS_ANALYSIS = "previous-d2-r-analysis.json"
D2R110_OFFICIAL_ANALYSIS_SHA256 = "27b389a03dfa8df4f9a6b531663e9767910de6f51ffbceef8006f5039cfdd844"
D3_ROLE = "D3"
D3_PLAN_FILENAME = "m5-d3.json"
D3_PREVIOUS_ANALYSIS = "previous-d2-r110-analysis.json"
D4_ROLE = "D4"
D4_PLAN_FILENAME = "m5-d4.json"
D5_ROLE = "D5"
D5_PLAN_FILENAME = "m5-d5.json"
D5R_ROLE = "D5-R"
D5R_PLAN_FILENAME = "m5-d5r.json"
D6_ROLE = "D6"
D6_PLAN_FILENAME = "m5-d6.json"
D7_ROLE = "D7"
D7_PLAN_FILENAME = "m5-d7.json"
DIAGNOSTIC_ROLES = (
    D1_ROLE,
    D2_ROLE,
    D2R_ROLE,
    D2R110_ROLE,
    D3_ROLE,
    D4_ROLE,
    D5_ROLE,
    D5R_ROLE,
    D6_ROLE,
    D7_ROLE,
)
REVIEWED_DIAGNOSTIC_PLANS = {
    D2R_ROLE: (8, "d2r_plan_sha256", D2R_PLAN_FILENAME),
    D2R110_ROLE: (9, "d2r110_plan_sha256", D2R110_PLAN_FILENAME),
    D3_ROLE: (10, "d3_plan_sha256", D3_PLAN_FILENAME),
    D4_ROLE: (11, "d4_plan_sha256", D4_PLAN_FILENAME),
    D5_ROLE: (12, "d5_plan_sha256", D5_PLAN_FILENAME),
    D5R_ROLE: (13, "d5r_plan_sha256", D5R_PLAN_FILENAME),
    D6_ROLE: (14, "d6_plan_sha256", D6_PLAN_FILENAME),
    D7_ROLE: (15, "d7_plan_sha256", D7_PLAN_FILENAME),
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
        if qualification_authorization_version(config) == 1:
            require(
                tuple(roles) == qualifying
                and record["max_attempts"] == 16
                and is_hash(record.get("d2r110_analysis_sha256"))
                and "qualification_batch" not in record,
                "split C3 requires its exclusive seven-slot qualification authorization",
            )
        else:
            qualification_batch(record, config)
    else:
        require(
            "d2r110_analysis_sha256" not in record and "qualification_batch" not in record,
            "qualification evidence outside split C3 qualification authorization",
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


def qualification_authorization_version(config):
    version = config["calibration"].get("qualification_authorization_schema_version", 1)
    require(
        type(version) is int and version in (1, 2), "unknown qualification authorization schema"
    )
    return version


def qualification_batch(record, config):
    batch = record.get("qualification_batch")
    roles = qualification_roles(config)
    require(
        qualification_authorization_version(config) == 2
        and roles != CALIBRATION
        and tuple(record["roles"]) == roles
        and "d2r110_analysis_sha256" not in record
        and type(batch) is dict
        and set(batch)
        == {
            "schema_version",
            "prior_allocation_count",
            "prior_history_sha256",
            "new_max_attempts",
            "predecessor",
        }
        and batch["schema_version"] == 2,
        "split C3 requires its versioned qualification batch authorization",
    )
    prior, new = batch["prior_allocation_count"], batch["new_max_attempts"]
    require(
        exact_int(prior, 1, 99)
        and exact_int(new, len(roles), 99 - prior)
        and record["max_attempts"] == prior + new
        and record["max_requested_minutes"] == 20 * (prior + new)
        and is_hash(batch["prior_history_sha256"]),
        "qualification cumulative budget differs from prior history and new approvals",
    )
    predecessor = batch["predecessor"]
    permitted = config["calibration"].get("qualification_predecessor_roles", [])
    require(
        type(permitted) is list
        and all(
            id_ok(role) and role not in (D4_ROLE, D5_ROLE, D5R_ROLE, D6_ROLE, D7_ROLE)
            for role in permitted
        )
        and type(predecessor) is dict
        and set(predecessor)
        == {
            "protocol",
            "role",
            "attempt",
            "execution_commit",
            "proposal_file",
            "proposal_sha256",
            "review_directory",
            "analysis_sha256",
        }
        and predecessor["protocol"] == QUALIFICATION_SUCCESSOR_PROTOCOL
        and predecessor["role"] in permitted
        and id_ok(predecessor["attempt"])
        and id_ok(predecessor["review_directory"])
        and is_hash(predecessor["execution_commit"], 40)
        and is_hash(predecessor["proposal_sha256"])
        and is_hash(predecessor["analysis_sha256"])
        and type(predecessor["proposal_file"]) is str
        and re.fullmatch(r"m5-[a-z0-9-]+\.json", predecessor["proposal_file"]) is not None,
        "qualification requires an approved same-condition predecessor protocol",
    )
    return batch


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
    return _bind_d2r110_lineage(proposal, previous, previous_d2r, D2R110_ROLE)


def _bind_d2r110_lineage(proposal, previous, previous_d2r, role):
    d2_sha = bind_retained_d2_analysis(proposal["retained_D2"], previous, role)
    retained = proposal["retained_D2R"]
    require(
        retained["analysis_sha256"] == digest(previous_d2r) == D2R_OFFICIAL_ANALYSIS_SHA256,
        f"{role} retained D2-R analysis differs",
    )
    predecessor = read_json(previous_d2r)
    require(
        predecessor["role"] == D2R_ROLE
        and retained["status"] == predecessor["status"] == "diagnostic-incomplete"
        and retained["proposal_sha256"] == predecessor["proposal_sha256"],
        f"{role} predecessor is not the retained incomplete D2-R",
    )
    return d2_sha, retained["analysis_sha256"]


def bind_d3_inputs(record, plan, previous, previous_d2r, previous_d2r110):
    require(
        record["roles"] == [D3_ROLE] and digest(plan) == record["d3_plan_sha256"],
        "D3 staged plan differs from authorization",
    )
    proposal = read_json(plan)
    earlier = _bind_d2r110_lineage(proposal, previous, previous_d2r, D3_ROLE)
    retained = proposal["retained_D2R110"]
    require(
        retained["analysis_sha256"] == digest(previous_d2r110) == D2R110_OFFICIAL_ANALYSIS_SHA256,
        "D3 retained D2-R110 analysis differs",
    )
    predecessor = read_json(previous_d2r110)
    require(
        predecessor["role"] == D2R110_ROLE
        and retained["status"] == predecessor["status"] == "diagnostic-complete"
        and retained["proposal_sha256"] == predecessor["proposal_sha256"]
        and predecessor["decision"]["thermal_control"] == "verified"
        and predecessor["decision"]["endpoint_stability_under_control"] == "unstable"
        and predecessor["decision"]["qualification_started"] is False,
        "D3 predecessor is not the retained thermally verified, endpoint-unstable D2-R110",
    )
    return (*earlier, retained["analysis_sha256"])


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
