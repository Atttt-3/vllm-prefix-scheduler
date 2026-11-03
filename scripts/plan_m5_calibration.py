"""Generate and evaluate the six predeclared M5 calibration allocations."""

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

STAGES = ("C1", "C2", "C3", "C4", "C5", "C6")
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


def plan_stage(config, stage, rates=None, serial_interval_ms=None):
    """Return the exact ordered run specs for a stage or the final C6 wave."""
    contract = config["calibration"]
    budget = config["budget"]
    require(
        stage in STAGES
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
    if stage in {"C3", "C4", "C5", "C6"}:
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
    index = read(index_path)
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
        report = read(path)
        require(report.get("pair_id") == identity, f"{identity}: audit run identity differs")
        inputs_hash = file_sha256(input_path)
        require(report.get("input_sha256") == inputs_hash, f"{identity}: audit input hash differs")
        if config_sha256 is not None:
            require(
                report.get("config_sha256") == config_sha256,
                f"{identity}: audit config hash differs",
            )
        measured = validate_run(spec, report, read(input_path))
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
        command.add_argument("--stage", choices=STAGES, required=True)
        command.add_argument("--rates", type=Path, help="Qualified C2 result JSON")
        command.add_argument("--output", type=Path, required=True)
        if action == "plan":
            command.add_argument("--serial-interval-ms", type=int)
        else:
            command.add_argument("--index", type=Path, required=True)
            command.add_argument("--dress-report", action="append", type=Path, default=[])
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
