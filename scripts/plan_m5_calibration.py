"""Generate and evaluate frozen M5 calibration allocations."""

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vllm_prefix_scheduler.m5_campaign import is_hash, read_json  # noqa: E402
from vllm_prefix_scheduler.m5_d1 import C3_BLOCKED_SHA  # noqa: E402

STAGES = ("C1", "C2", "C3", "C4", "C5", "C6")
C3_SPLIT_STAGES = ("C3-burst", "C3-mixed")
C3_SPLIT_PROTOCOL = "c3-balanced-split-v1"
C3_SPLIT_COUNT = 60
C3_BLOCK_TIERS = (
    ("endpoint", "light", "light", "endpoint"),
    ("light", "endpoint", "endpoint", "light"),
)
C3_COMPLETION_FIELDS = ("planned", "submitted", "completed", "timely_first_token")
C3_TERMINAL_FIELDS = ("completed", "failed", "cancelled", "timeout", "unfinished")
C3_PROFILE_FIELDS = (
    "schedule",
    "window",
    "score",
    "sort_exclusive",
    "queue_commit",
    "schedule_envelope",
)
KEY_SCENARIOS = ("burst", "mixed")
CONTROL_SCENARIOS = ("ample", "all-cold", "serial")
MODES = ("native", "off", "prefix", "guarded")
ORDER = ("native", "off", "guarded", "prefix")
NS_PER_MS = 1_000_000
PAIR_MARGIN_SECONDS = 60


class CalibrationError(ValueError):
    """The frozen calibration plan or its evidence is incomplete."""


def require(condition, reason):
    if not condition:
        raise CalibrationError(reason)


def number(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def c3_split_gate(config):
    gates = config["gates"]
    require(
        config["calibration"].get("split_c3") is True, "C3 split is not enabled in configuration"
    )
    require(
        config.get("measurement_schema_version") == 2
        and config["calibration"]["pilot_count"] == C3_SPLIT_COUNT
        and (
            gates["overhead_ttft_ns"],
            gates["overhead_ttft_rel"],
            gates["overhead_throughput_rel"],
        )
        == (2_000_000, 0.05, 0.05)
        and (gates["submit_p99_ns"], gates["submit_max_ns"], gates["deadline_ns"])
        == (5_000_000, 25_000_000, 30_000_000_000)
        and gates["max_allocation_work_s"] == 1050,
        "C3 split measurement version, request count or original tolerance differs",
    )
    return gates


def plan_stage(config, stage, rates=None, serial_interval_ms=None):
    """Return the exact ordered run specs for a stage or the final C6 wave."""
    contract = config["calibration"]
    budget = config["budget"]
    require(
        stage in (*STAGES, *C3_SPLIT_STAGES)
        and tuple(contract["stages"]) == STAGES
        and tuple(config["modes"]) == MODES
        and budget["calibration_allocations"] == len(STAGES)
        and budget["minutes_per_allocation"] == 20,
        "M5 calibration stages, modes or allocation cap differ",
    )
    candidate_rates = contract["candidate_rates"]
    require(candidate_rates == [3, 4, 5, 6, 8], "M5 candidate rates differ")
    seed_base = contract["seed_base"]
    count = contract["pilot_count"]
    require(type(seed_base) is int and 51_000 <= seed_base <= 51_400, "invalid pilot seeds")
    require(type(count) is int and 20 <= count <= 100 and count % 20 == 0, "invalid pilot count")
    require(config["request_count"] == 500, "M5 full-length request count differs")
    if stage in {"C3", "C4", "C5", "C6", *C3_SPLIT_STAGES}:
        require(
            type(rates) is dict
            and set(rates) == set(KEY_SCENARIOS)
            and all(type(rate) is int and rate in candidate_rates for rate in rates.values()),
            "C2 qualified native rates required",
        )
    require(serial_interval_ms is None or stage == "C6", "serial interval applies only to C6")

    runs = []

    def add(
        identity,
        scenario,
        seed,
        mode="native",
        tier="light",
        *,
        rows=count,
        rate=None,
        interval=None,
        mechanism=None,
    ):
        runs.append(
            {
                "id": identity,
                "scenario": scenario,
                "seed": seed_base + seed,
                "count": rows,
                "mode": mode,
                "tier": tier,
                "rate_rps": rate,
                "interval_ms": interval,
                "mechanism": mechanism,
            }
        )

    if stage == "C1":
        for profile, order in (("pressure", ORDER), ("ample", ORDER[::-1])):
            for mode in order:
                add(
                    f"mechanism-{profile}-{mode}",
                    "mechanism",
                    1,
                    mode,
                    "profile",
                    rows=2,
                    mechanism=profile,
                )
    elif stage == "C2":
        for scenario, seed in zip(KEY_SCENARIOS, (101, 102), strict=True):
            for rate in candidate_rates:
                add(f"{scenario}-{rate}rps", scenario, seed, rate=rate)
    elif stage == "C3":
        for scenario, mode, seed in (("burst", "native", 201), ("mixed", "guarded", 211)):
            for repeat, tiers in enumerate((("endpoint", "light"), ("light", "endpoint")), 1):
                for tier in tiers:
                    add(
                        f"{scenario}-{repeat}-{tier}",
                        scenario,
                        seed + repeat,
                        mode,
                        tier,
                        rate=rates[scenario],
                    )
        add("mixed-stages", "mixed", 213, "guarded", "profile", rate=rates["mixed"])
    elif stage in C3_SPLIT_STAGES:
        c3_split_gate(config)
        scenario, mode, seed = (
            ("burst", "native", 220) if stage == "C3-burst" else ("mixed", "guarded", 230)
        )
        for block, order in enumerate(C3_BLOCK_TIERS, 1):
            for position, tier in enumerate(order, 1):
                add(
                    f"{scenario}-b{block}-{position}-{tier}",
                    scenario,
                    seed + block,
                    mode,
                    tier,
                    rate=rates[scenario],
                )
        if scenario == "mixed":
            add("mixed-split-stages", scenario, 233, mode, "profile", rate=rates[scenario])
    elif stage in {"C4", "C5"}:
        scenario, seed, order = (
            ("burst", 301, ORDER) if stage == "C4" else ("mixed", 401, ORDER[::-1])
        )
        for mode in order:
            add(
                f"{scenario}-full-{mode}",
                scenario,
                seed,
                mode,
                rows=config["request_count"],
                rate=rates[scenario],
            )
    elif serial_interval_ms is None:
        for scenario, seed in (("ample", 501), ("all-cold", 502)):
            add(
                f"{scenario}-full",
                scenario,
                seed,
                rows=config["request_count"],
                rate=rates["mixed"],
            )
        pilot_count = contract["serial_pilot_count"]
        pilot_interval = contract["serial_pilot_interval_ms"]
        require(pilot_count == 20 and pilot_interval == 1000, "serial pilot shape differs")
        add("serial-pilot", "serial", 503, rows=pilot_count, interval=pilot_interval)
    else:
        require(
            type(serial_interval_ms) is int
            and 0 < serial_interval_ms <= 1000
            and serial_interval_ms % 10 == 0,
            "invalid derived serial interval",
        )
        add("serial-full", "serial", 504, rows=config["request_count"], interval=serial_interval_ms)

    require(all(run["seed"] < 52_000 for run in runs), "calibration and formal seed ranges overlap")
    arrival_seconds = sum(
        run["count"] / run["rate_rps"]
        if run["rate_rps"] is not None
        else run["count"] * run["interval_ms"] / 1000
        if run["interval_ms"] is not None
        else 0
        for run in runs
    )
    return {
        "schema_version": 1,
        "stage": stage,
        "allocation_minutes": budget["minutes_per_allocation"],
        "work_timeout_s": config["gates"]["max_allocation_work_s"],
        "estimated_mode_work_seconds": math.ceil(75 * len(runs) + arrival_seconds),
        "allocation_reserve_seconds": PAIR_MARGIN_SECONDS,
        "runs": runs,
    }


def validate_run(spec, report, inputs):
    for key in ("scenario", "seed", "mode", "tier"):
        require(report.get(key) == spec[key], f"{spec['id']}: audit {key} differs")
    phase = "diagnostic" if spec["scenario"] == "mechanism" else "calibration"
    require(report.get("phase") == phase, f"{spec['id']}: audit phase differs")
    require(report.get("status") == "valid", f"{spec['id']}: audit is not valid")
    for key in ("scenario", "seed", "rate_rps", "interval_ms", "mechanism"):
        require(inputs.get(key) == spec[key], f"{spec['id']}: input {key} differs")
    require(len(inputs["requests"]) == spec["count"], f"{spec['id']}: input row count differs")
    metrics = report["metrics"]
    counts = metrics["strata"]["all"]["counts"]
    require(counts["planned"] == spec["count"], f"{spec['id']}: planned count differs")
    return metrics


def timely_completion(metrics):
    stratum = metrics["strata"]["all"]
    return min(stratum["completion_rate"], stratum["timely_first_token_rate"])


def serial_interval_from_pilot(report):
    """Freeze a conservative open-loop interval from the valid native serial pilot."""
    require(
        report.get("status") == "valid"
        and report.get("phase") == "calibration"
        and report.get("scenario") == "serial"
        and report.get("mode") == "native"
        and report.get("tier") == "light",
        "valid native serial pilot required",
    )
    metrics = report["metrics"]
    counts = metrics["strata"]["all"]["counts"]
    service_ns = metrics["duration_ns"]["max_submit_to_end"]
    require(
        counts["planned"] == counts["completed"] == 20
        and metrics["max_outstanding"] == 1
        and type(service_ns) is int
        and service_ns > 0,
        "serial pilot incomplete or overlapping",
    )
    # ceil to 10 ms after allowing 25% service variation and 25 ms of submit jitter.
    quantum = 10 * NS_PER_MS
    interval_ms = ((5 * service_ns + 100 * NS_PER_MS + 4 * quantum - 1) // (4 * quantum)) * 10
    require(interval_ms <= 1000, "serial pilot cannot establish a safe bounded interval")
    return interval_ms


def observer_comparison(metrics, gates):
    comparisons = []
    for scenario in KEY_SCENARIOS:
        for repeat in (1, 2):
            key = f"{scenario}-{repeat}"
            reference, light = (metrics[f"{key}-{tier}"] for tier in ("endpoint", "light"))
            base_ttft, light_ttft = (
                item["strata"]["all"]["ttft_ns"]["p95"] for item in (reference, light)
            )
            require(number(base_ttft, 1) and number(light_ttft, 1), f"{key}: TTFT missing")
            tolerance = max(gates["overhead_ttft_ns"], base_ttft * gates["overhead_ttft_rel"])
            throughput_delta = {}
            for name in ("requests", "tokens"):
                measure = f"last_terminal_{name}_per_s"
                base, observed = (item["throughput"][measure] for item in (reference, light))
                require(
                    number(base) and base > 0 and number(observed), f"{key}: throughput missing"
                )
                throughput_delta[name] = (observed - base) / base
            comparisons.append(
                {
                    "id": key,
                    "endpoint_ttft_p95_ns": base_ttft,
                    "light_ttft_p95_ns": light_ttft,
                    "ttft_delta_ns": light_ttft - base_ttft,
                    "ttft_tolerance_ns": tolerance,
                    "relative_throughput_delta": throughput_delta,
                    "qualified": abs(light_ttft - base_ttft) <= tolerance
                    and all(
                        abs(delta) <= gates["overhead_throughput_rel"]
                        for delta in throughput_delta.values()
                    ),
                }
            )
    return comparisons


def c3_observation(metrics):
    all_requests = metrics["strata"]["all"]
    counts = all_requests["counts"]
    return {
        "counts": {name: counts[name] for name in (*C3_COMPLETION_FIELDS, *C3_TERMINAL_FIELDS[1:])},
        "kv_blocks": metrics["kv_blocks"],
        "ttft_p95_ns": all_requests["ttft_ns"]["p95"],
        "submit_lag_ns": {name: metrics["submit_lag_ns"][name] for name in ("n", "p99", "max")},
        "last_terminal_throughput": {
            kind: metrics["throughput"][f"last_terminal_{kind}_per_s"]
            for kind in ("requests", "tokens")
        },
    }


def c3_comparison(block, kind, reference, observed, observations, evidence, gates):
    a, b = observations[reference["id"]], observations[observed["id"]]
    outcome = {
        "block": block,
        "kind": kind,
        "reference": reference["id"],
        "observed": observed["id"],
        "seed": reference["seed"],
        "input_sha256": evidence[reference["id"]]["input_sha256"],
        "reference_ttft_p95_ns": a["ttft_p95_ns"],
        "observed_ttft_p95_ns": b["ttft_p95_ns"],
        "ttft_delta_ns": None,
        "ttft_tolerance_ns": None,
        "relative_throughput_delta": None,
        "qualified": False,
    }
    if any(
        row["counts"]["completed"] != C3_SPLIT_COUNT
        or row["counts"]["timely_first_token"] != C3_SPLIT_COUNT
        for row in (a, b)
    ):
        return {**outcome, "status": "incomplete"}
    delta = b["ttft_p95_ns"] - a["ttft_p95_ns"]
    tolerance = max(gates["overhead_ttft_ns"], a["ttft_p95_ns"] * gates["overhead_ttft_rel"])
    throughput = {
        kind: (b["last_terminal_throughput"][kind] - base) / base
        for kind, base in a["last_terminal_throughput"].items()
    }
    qualified = abs(delta) <= tolerance and all(
        abs(value) <= gates["overhead_throughput_rel"] for value in throughput.values()
    )
    return {
        **outcome,
        "ttft_delta_ns": delta,
        "ttft_tolerance_ns": tolerance,
        "relative_throughput_delta": throughput,
        "qualified": qualified,
        "status": "qualified" if qualified else "outside-tolerance",
    }


def c3_split_report(config, stage, specs, evidence, observations, stages, config_sha256):
    gates = c3_split_gate(config)
    scenario = specs[0]["scenario"]
    order = [spec["id"] for spec in specs]
    require(list(evidence) == list(observations) == order, "C3 split report run inventory differs")
    incomplete = []
    for identity in order:
        run, observation = evidence[identity], observations[identity]
        require(
            is_hash(run["audit_sha256"])
            and is_hash(run["input_sha256"])
            and number(run["elapsed_seconds"], 1e-9),
            f"{identity}: split audit identity or elapsed time invalid",
        )
        counts = observation["counts"]
        complete = all(counts[name] == C3_SPLIT_COUNT for name in C3_COMPLETION_FIELDS)
        require(
            set(counts) == set((*C3_COMPLETION_FIELDS, *C3_TERMINAL_FIELDS))
            and all(
                type(value) is int and 0 <= value <= C3_SPLIT_COUNT for value in counts.values()
            )
            and counts["planned"]
            == sum(counts[key] for key in C3_TERMINAL_FIELDS)
            == C3_SPLIT_COUNT
            and max(counts["completed"], counts["timely_first_token"]) <= counts["submitted"],
            f"{identity}: split request denominators differ",
        )
        p95 = observation["ttft_p95_ns"]
        throughput = observation["last_terminal_throughput"]
        lag = observation["submit_lag_ns"]
        require(
            lag["n"] == counts["submitted"] == C3_SPLIT_COUNT
            and number(lag["p99"])
            and number(lag["max"])
            and lag["p99"] <= lag["max"]
            and lag["p99"] <= gates["submit_p99_ns"]
            and lag["max"] <= gates["submit_max_ns"],
            f"{identity}: split submission lag evidence invalid",
        )
        require(
            type(observation["kv_blocks"]) is int
            and observation["kv_blocks"] > 1
            and (number(p95, 1) if counts["completed"] else p95 is None)
            and set(throughput) == {"requests", "tokens"}
            and all(number(value, 1e-9 if complete else 0) for value in throughput.values()),
            f"{identity}: split TTFT, throughput or KV capacity evidence invalid",
        )
        if not complete:
            incomplete.append(identity)
    require(
        len({row["kv_blocks"] for row in observations.values()}) == 1,
        "C3 split observed different physical KV capacities",
    )
    comparisons = []
    blocks = []
    for block, start in enumerate((0, 4), 1):
        leaves = specs[start : start + 4]
        table_hashes = {evidence[spec["id"]]["input_sha256"] for spec in leaves}
        require(len(table_hashes) == 1, "C3 split block does not share a frozen arrival table")
        blocks.append(
            {
                "block": block,
                "seed": leaves[0]["seed"],
                "input_sha256": table_hashes.pop(),
                "ordered_runs": [spec["id"] for spec in leaves],
            }
        )
        for a, b in ((leaves[0], leaves[1]), (leaves[2], leaves[3])):
            reference, observed = (a, b) if a["tier"] == "endpoint" else (b, a)
            comparisons.append(
                c3_comparison(block, "E/L", reference, observed, observations, evidence, gates)
            )
        for kind, tier in (("E/E", "endpoint"), ("L/L", "light")):
            reference, observed = [spec for spec in leaves if spec["tier"] == tier]
            comparisons.append(
                c3_comparison(block, kind, reference, observed, observations, evidence, gates)
            )
    require(
        blocks[0]["input_sha256"] != blocks[1]["input_sha256"],
        "C3 split reused one input across blocks",
    )
    measured = sum(run["elapsed_seconds"] for run in evidence.values())
    reasons = []
    if incomplete:
        reasons.append("split leaves lack all 60 completed and timely requests")
    for kind in ("E/E", "L/L", "E/L"):
        if any(row["kind"] == kind and row["status"] == "outside-tolerance" for row in comparisons):
            reasons.append(f"{kind} comparison exceeds an original paired tolerance")
    if measured + PAIR_MARGIN_SECONDS > gates["max_allocation_work_s"]:
        reasons.append("split measured work plus reserve exceeds the working limit")
    if scenario == "mixed":
        if any(
            not number(stages.get(name, {}).get("n"), 1)
            or not number(stages.get(name, {}).get("max"), 1)
            for name in C3_PROFILE_FIELDS
        ):
            reasons.append("profile did not exercise all declared scheduler measurements")
    else:
        require(stages is None, "burst split cannot substitute an extra profile leaf")
    return {
        "schema_version": 2,
        "qualification_protocol": C3_SPLIT_PROTOCOL,
        "stage": stage,
        "scenario": scenario,
        "rate_rps": specs[0]["rate_rps"],
        "status": "blocked" if reasons else "qualified",
        "runs": evidence,
        "observations": observations,
        "blocks": blocks,
        "comparisons": comparisons,
        "incomplete_runs": incomplete,
        "measured_run_seconds": measured,
        "projected_work_seconds": measured + PAIR_MARGIN_SECONDS,
        "stage_costs_ns": stages,
        "reasons": reasons,
        **({"config_sha256": config_sha256} if config_sha256 is not None else {}),
    }


def aggregate_c3(config, stage_reports, rates, *, config_sha256, original_c3):
    """Reconstruct two decisions before binding them into the prospective logical C3 result."""
    require(is_hash(config_sha256), "C3 split aggregate requires a configuration hash")
    original_c3 = Path(original_c3)
    retained = file_sha256(original_c3)
    original = read_json(original_c3)
    require(
        retained == C3_BLOCKED_SHA
        and original.get("stage") == "C3"
        and original.get("status") == "blocked",
        "retained original C3 blocked result differs",
    )
    require(
        len(stage_reports) == len(C3_SPLIT_STAGES), "both ordered C3 split reports are required"
    )
    summaries = {}
    for stage, path in zip(C3_SPLIT_STAGES, stage_reports, strict=True):
        path = Path(path)
        report = read_json(path)
        require(
            report.get("schema_version") == 2
            and report.get("qualification_protocol") == C3_SPLIT_PROTOCOL
            and report.get("stage") == stage
            and report.get("config_sha256") == config_sha256,
            f"{stage}: split report source or order differs",
        )
        specs = plan_stage(config, stage, rates)["runs"]
        rebuilt = c3_split_report(
            config,
            stage,
            specs,
            report["runs"],
            report["observations"],
            report["stage_costs_ns"],
            config_sha256,
        )
        require(rebuilt == report, f"{stage}: split decisions differ from reconstruction")
        summaries[stage] = {
            "status": report["status"],
            "report_sha256": file_sha256(path),
            "rate_rps": report["rate_rps"],
            "runs": report["runs"],
            "comparisons": report["comparisons"],
            "reasons": report["reasons"],
        }
    reasons = [
        f"{stage} did not qualify"
        for stage, row in summaries.items()
        if row["status"] != "qualified"
    ]
    return {
        "schema_version": 2,
        "stage": "C3",
        "qualification_protocol": C3_SPLIT_PROTOCOL,
        "status": "blocked" if reasons else "qualified",
        "config_sha256": config_sha256,
        "retained_original_c3": {"status": "blocked", "sha256": retained},
        "selected_rates": rates,
        "split_reports": summaries,
        "reasons": reasons,
    }


def native_rate_selection(specs, metrics, gates):
    pilots = []
    selected = {}
    for spec in specs:
        values = metrics[spec["id"]]
        completion = timely_completion(values)
        competition = values["pressure"]["competing_windows"]
        lag = values["submit_lag_ns"]
        submit_valid = (
            lag["n"] == spec["count"]
            and number(lag["p99"])
            and lag["p99"] <= gates["submit_p99_ns"]
            and number(lag["max"])
            and lag["max"] <= gates["submit_max_ns"]
        )
        qualifies = (
            submit_valid and completion >= gates["min_calibration_completion"] and competition > 0
        )
        if qualifies:
            selected.setdefault(spec["scenario"], spec["rate_rps"])
        pilots.append(
            {
                "id": spec["id"],
                "rate_rps": spec["rate_rps"],
                "timely_completion_lower_bound": completion,
                "competing_windows": competition,
                "submit_lag_valid": submit_valid,
                "qualified": qualifies,
            }
        )
    return selected, pilots


def load_stage_evidence(config, stage, index_path, rates, config_sha256):
    index_path = Path(index_path)
    reader = read_json if stage in C3_SPLIT_STAGES else read
    index = reader(index_path)
    require(
        index.get("schema_version") == 1 and index.get("stage") == stage,
        "calibration index schema/stage differs",
    )
    rows = index["runs"]
    specs = plan_stage(config, stage, rates)["runs"]
    if stage == "C6":
        pilots = [row for row in rows if row["id"] == "serial-pilot"]
        require(len(pilots) == 1, "C6 serial pilot missing or duplicated")
        interval = serial_interval_from_pilot(read(index_path.parent / pilots[0]["audit"]))
        specs += plan_stage(config, stage, rates, interval)["runs"]
    require(
        [row["id"] for row in rows] == [spec["id"] for spec in specs],
        "calibration order or run inventory differs",
    )

    evidence, metrics, table_hashes = {}, {}, defaultdict(set)
    for spec, row in zip(specs, rows, strict=True):
        identity = spec["id"]
        elapsed = row["elapsed_seconds"]
        require(number(elapsed) and elapsed > 0, f"{identity}: invalid measured wall time")
        path = index_path.parent / row["audit"]
        input_path = index_path.parent / row["inputs"]
        report = reader(path)
        require(report.get("pair_id") == identity, f"{identity}: audit run identity differs")
        if stage in C3_SPLIT_STAGES:
            require(
                report.get("measurement_schema_version") == 2,
                f"{identity}: split audit measurement version differs",
            )
        inputs_hash = file_sha256(input_path)
        require(report.get("input_sha256") == inputs_hash, f"{identity}: audit input hash differs")
        if config_sha256 is not None:
            require(
                report.get("config_sha256") == config_sha256,
                f"{identity}: audit config hash differs",
            )
        measured = validate_run(spec, report, reader(input_path))
        require(
            sum(
                measured["duration_ns"][part]
                for part in ("load", "warmup", "measure_through_drain")
            )
            <= elapsed * 1e9,
            f"{identity}: wall time shorter than observed phases",
        )
        metrics[identity] = measured
        table_key = (spec["scenario"], spec["seed"], spec["rate_rps"], spec["mechanism"])
        table_hashes[table_key].add(inputs_hash)
        evidence[identity] = {
            "elapsed_seconds": elapsed,
            "audit_sha256": file_sha256(path),
            "input_sha256": inputs_hash,
        }
    require(
        all(len(values) == 1 for values in table_hashes.values()),
        "calibration pair does not share a frozen arrival table",
    )
    return specs, rows, evidence, metrics


def evaluate_stage(config, stage, index_path, rates=None, dress_reports=(), *, config_sha256=None):
    """Apply predeclared decisions to independently validated, indexed run reports."""
    specs, rows, evidence, metrics = load_stage_evidence(
        config, stage, index_path, rates, config_sha256
    )
    if stage in C3_SPLIT_STAGES:
        return c3_split_report(
            config,
            stage,
            specs,
            evidence,
            {identity: c3_observation(values) for identity, values in metrics.items()},
            metrics["mixed-split-stages"]["stage_costs_ns"] if stage == "C3-mixed" else None,
            config_sha256,
        )
    gates = config["gates"]
    measured = {key: row["elapsed_seconds"] for key, row in evidence.items()}
    result = {
        "schema_version": 1,
        "stage": stage,
        "status": "qualified",
        "runs": evidence,
        "measured_run_seconds": sum(measured.values()),
        "reasons": [],
    }
    if config_sha256 is not None:
        result["config_sha256"] = config_sha256
    reasons = result["reasons"]
    if result["measured_run_seconds"] > gates["max_allocation_work_s"]:
        reasons.append("allocation run time exceeds the working limit")
    if stage == "C1":
        result["mechanism"] = mechanism_gate(config, Path(index_path).parent, specs, rows)
    elif stage == "C2":
        selected, pilots = native_rate_selection(specs, metrics, gates)
        result.update(selected_rates=selected, pilots=pilots)
        if set(selected) != set(KEY_SCENARIOS):
            reasons.append("at least one key workload has no qualifying native rate")
    elif stage == "C3":
        comparisons = observer_comparison(metrics, gates)
        result["comparisons"] = comparisons
        if any(not row["qualified"] for row in comparisons):
            reasons.append("light capture perturbation exceeds a paired tolerance")
        stages = metrics["mixed-stages"]["stage_costs_ns"]
        result["stage_costs_ns"] = stages
        expected = ("schedule", "window", "score", "sort_exclusive", "queue_commit")
        if config.get("measurement_schema_version", 1) == 2:
            expected += ("schedule_envelope",)
        if any(stages.get(name, {}).get("n", 0) <= 0 for name in expected):
            reasons.append("profile did not exercise all declared scheduler measurements")
    elif stage in {"C4", "C5"}:
        scenario = specs[0]["scenario"]
        native = metrics[f"{scenario}-full-native"]
        pressure = native["pressure"]
        result.update(
            scenario=scenario,
            native_pressure=pressure,
            native_completion_lower_bound=timely_completion(native),
            mode_wall_seconds={spec["mode"]: measured[spec["id"]] for spec in specs},
            pair_work_seconds=sum(measured.values()) + PAIR_MARGIN_SECONDS,
        )
        if (
            pressure["competing_windows"] < gates["min_competing_windows"]
            or pressure["reusable_evictions"] < gates["min_reusable_evictions"]
            or timely_completion(native) < gates["min_calibration_completion"]
        ):
            reasons.append("native dress rehearsal missed pressure or completion gates")
        if result["pair_work_seconds"] > gates["max_allocation_work_s"]:
            reasons.append("full four-mode pair exceeds the projected work limit")
    else:
        result.update(
            control_gate(
                config,
                specs,
                metrics,
                measured,
                dress_reports,
                reasons,
                Path(index_path).parent / rows[0]["inputs"],
                config_sha256,
            )
        )
    if reasons:
        result["status"] = "blocked"
    return result


def control_gate(config, specs, metrics, measured, dress_reports, reasons, ample_path, config_hash):
    gates = config["gates"]
    require(len(dress_reports) == 2, "both full-length dress reports required for C6")
    factor = 1.1
    for expected, report in zip(("C4", "C5"), dress_reports, strict=True):
        require(
            report.get("stage") == expected and report.get("status") == "qualified",
            "qualified C4/C5 dress reports required",
        )
        require(
            config_hash is None or report.get("config_sha256") == config_hash,
            "dress report configuration differs",
        )
        wall = report["mode_wall_seconds"]
        require(
            set(wall) == set(MODES) and all(number(x) and x > 0 for x in wall.values()),
            "dress report mode wall times differ",
        )
        factor = max(factor, max(wall.values()) / wall["native"])
    projections = {}
    for scenario in CONTROL_SCENARIOS:
        identity = f"{scenario}-full"
        values = metrics[identity]
        projected = len(MODES) * measured[identity] * factor + PAIR_MARGIN_SECONDS
        projections[scenario] = projected
        if timely_completion(values) < gates["min_calibration_completion"]:
            reasons.append(f"{scenario}: native completion below threshold")
        if projected > gates["max_allocation_work_s"]:
            reasons.append(f"{scenario}: projected four-mode pair exceeds work limit")
    if metrics["ample-full"]["pressure"]["reusable_evictions"]:
        reasons.append("ample cache showed reusable-prefix eviction")
    ample = read(ample_path)
    block_size = config["block_size"]
    output = ample["sampling"]["max_tokens"]
    # Conservative resident working set even when none of the planned inputs share KV.
    ample_capacity_upper_bound = 1 + sum(
        (len(row["prompt_token_ids"]) + output + block_size - 1) // block_size
        for row in (*ample["warmup"], *ample["requests"])
    )
    if ample_capacity_upper_bound > metrics["ample-full"]["kv_blocks"]:
        reasons.append("ample cache conservative working set exceeds actual physical capacity")
    cold_metrics = metrics["all-cold-full"]
    if config.get("measurement_schema_version", 1) == 2:
        if cold_metrics.get("all_cold_zero_hit") is not True:
            reasons.append("all-cold control lacks per-request zero-hit evidence")
    else:
        cold = cold_metrics["strata"]["all"]
        if (
            cold["cached_tokens"]["max"] != 0
            or cold["cached_tokens"]["n"] < cold["counts"]["completed"]
        ):
            reasons.append("all-cold control showed a positive cache hit")
    if metrics["serial-full"]["max_outstanding"] != 1:
        reasons.append("serial control had overlapping requests")
    return {
        "selected_rates": {
            scenario: specs[i]["rate_rps"] for i, scenario in enumerate(CONTROL_SCENARIOS[:2])
        },
        "serial_interval_ms": specs[-1]["interval_ms"],
        "ample_capacity_upper_bound_blocks": ample_capacity_upper_bound,
        "control_wall_seconds": {
            scenario: measured[f"{scenario}-full"] for scenario in CONTROL_SCENARIOS
        },
        "control_projection_factor": factor,
        "pair_work_seconds": projections,
    }


def mechanism_gate(config, root, specs, index):
    """Verify the ordered B/C allocator and GPU execution chain from original records."""
    prefix_tokens = config["scenarios"]["mechanism"]["prefix_tokens"]
    block_size = config["block_size"]
    results = []
    for spec, ref in zip(specs, index, strict=True):
        run = root / ref["run"]
        client, engine = (read(run / f"{name}.json") for name in ("client", "engine"))
        audit = read(root / ref["audit"])
        require(
            audit.get("client_sha256") == file_sha256(run / "client.json")
            and audit.get("engine_sha256") == file_sha256(run / "engine.json"),
            f"{spec['id']}: diagnostic raw hash differs",
        )
        pressure = spec["mechanism"] == "pressure"
        native_order = spec["mode"] in {"native", "off"}
        tag = spec["id"]
        expected_blocks = config["scenarios"]["mechanism"][f"{spec['mechanism']}_blocks"]
        require(
            engine["profile"]["num_gpu_blocks"] == expected_blocks,
            f"{tag}: actual diagnostic KV capacity differs",
        )
        warm = [row for row in client["warmup"] if row["request_id"] == "prime-A"]
        require(
            len(warm) == 1 and warm[0]["status"] == "completed" and warm[0]["token_ids"],
            f"{tag}: A lacks actual generation",
        )
        result = {row["request_id"]: row for row in client["requests"]}
        require(
            set(result) == {"B", "C"} and all(r["status"] == "completed" for r in result.values()),
            f"{tag}: B or C lacks a completed request",
        )
        probes = engine["probes"]
        require(
            len(probes) == 2 and {row["request_id"] for row in probes} == {"B", "C"},
            f"{tag}: missing independent pre-admission B/C probe",
        )
        probes = {row["request_id"]: row for row in probes}
        prefix_blocks = set(probes["C"]["block_ids"])
        require(
            probes["B"]["hit_tokens"] == 0
            and not probes["B"]["block_ids"]
            and probes["C"]["hit_tokens"] == prefix_tokens
            and len(prefix_blocks) == prefix_tokens // block_size,
            f"{tag}: cold B or reusable C precondition failed",
        )
        windows = [row for row in engine["decisions"] if row["candidates"] == ["B", "C"]]
        require(bool(windows), f"{tag}: both requests never entered the same fresh head window")
        decision = min(windows, key=lambda row: row["batch_id"])
        require(
            decision["ordered"] == (["B", "C"] if native_order else ["C", "B"])
            and not decision["protected"],
            f"{tag}: actual window order differs",
        )
        require(
            all(row["at_ns"] <= decision["decision_ns"] for row in probes.values()),
            f"{tag}: B/C probe happened after the observed decision",
        )

        inputs = read(root / ref["inputs"])
        prompt_lengths = {
            row["request_id"]: len(row["prompt_token_ids"])
            for row in (*inputs["warmup"], *inputs["requests"])
        }
        first = {}
        prefill = defaultdict(int)
        for batch in engine["batches"]:
            runner = batch["runner"]
            if not runner:
                continue
            for actual in runner["rows"]:
                rid = actual["request_id"]
                if rid not in {"prime-A", "B", "C"}:
                    continue
                require(
                    batch["materialized"] is True
                    and batch["consumed"] is True
                    and actual["position_source"] == "runner_positions_d2h",
                    f"{tag}: successful GPU execution is missing",
                )
                first.setdefault(rid, (batch["batch_id"], actual["start"], actual["generation"]))
                prefill[rid] += max(0, min(actual["end"], prompt_lengths[rid]) - actual["start"])
        require(
            set(first) == {"prime-A", "B", "C"} and first["prime-A"][0] < decision["batch_id"],
            f"{tag}: A did not execute before the B/C decision",
        )
        b_batch, b_start, b_gen = first["B"]
        c_batch, c_start, c_gen = first["C"]
        loses_prefix = pressure and native_order
        valid_hit = (
            type(c_start) is int and 0 <= c_start < prefix_tokens and c_start % block_size == 0
            if loses_prefix
            else c_start == prefix_tokens
        )
        require(
            decision["batch_id"] <= min(b_batch, c_batch)
            and b_start == 0
            and valid_hit
            and result["C"]["cached_tokens"] == c_start,
            f"{tag}: C hit does not match actual runner position and client result",
        )
        admitted = {
            (row["request_id"], row["generation"], row["batch_id"], row["hit_tokens"])
            for row in engine["allocations"]
            if row["success"]
        }
        require(
            {("B", b_gen, b_batch, 0), ("C", c_gen, c_batch, c_start)} <= admitted,
            f"{tag}: B/C successful allocation chain missing",
        )
        victims = {
            row["block_id"]
            for row in engine["evictions"]
            if row["request_id"] == "B" and b_batch <= row["batch_id"] <= c_batch
        }
        evicted = prefix_blocks & victims
        require(
            bool(evicted) == loses_prefix,
            f"{tag}: B allocation did not establish the expected C-prefix eviction",
        )
        results.append(
            {
                "id": tag,
                "actual_kv_blocks": expected_blocks,
                "C_hit_tokens": c_start,
                "B_evicted_C_blocks": len(evicted),
                "B_prefill_tokens": prefill["B"],
                "C_prefill_tokens": prefill["C"],
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("plan", "evaluate"):
        command = commands.add_parser(action)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--stage", choices=(*STAGES, *C3_SPLIT_STAGES), required=True)
        command.add_argument("--rates", type=Path, help="Qualified C2 result JSON")
        command.add_argument("--output", type=Path, required=True)
        if action == "plan":
            command.add_argument("--serial-interval-ms", type=int)
        else:
            command.add_argument("--index", type=Path, required=True)
            command.add_argument("--dress-report", action="append", type=Path, default=[])
    aggregate = commands.add_parser("aggregate-c3")
    for name in ("config", "rates", "burst-report", "mixed-report", "original-c3", "output"):
        aggregate.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = read(args.config)
        config_sha256 = file_sha256(args.config)
        rates = None
        if args.rates:
            source = read(args.rates)
            require(
                source.get("stage") == "C2"
                and source.get("status") == "qualified"
                and source.get("config_sha256") == config_sha256,
                "qualified C2 result required",
            )
            rates = source["selected_rates"]
        if args.action == "plan":
            result = plan_stage(config, args.stage, rates, args.serial_interval_ms)
        elif args.action == "aggregate-c3":
            result = aggregate_c3(
                config,
                (args.burst_report, args.mixed_report),
                rates,
                config_sha256=config_sha256,
                original_c3=args.original_c3,
            )
        else:
            result = evaluate_stage(
                config,
                args.stage,
                args.index,
                rates,
                [read(path) for path in args.dress_report],
                config_sha256=config_sha256,
            )
        result["config_sha256"] = config_sha256
        write(args.output, result)
        return 2 if result.get("status") == "blocked" else 0
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(f"M5 calibration evidence rejected: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
