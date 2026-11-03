"""Fixed thermal-window selection for M5 D2 endpoint replays."""

import json
import math
from collections import Counter
from statistics import median

NS = 1_000_000_000
TARGET_FIELDS = ("temperature_c", "sm_clock_mhz", "memory_clock_mhz")


def read_samples(path, cutoff_ns):
    """Read only complete rows while the side sampler is appending."""
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.split(b"\n")[:-1]]
    last = 0
    uuid = None
    for row in rows:
        stamp = row["monotonic_ns"]
        if type(stamp) is not int or stamp <= last:
            raise ValueError("D2 telemetry clocks are not strictly increasing")
        last = stamp
        if uuid is None:
            uuid = row["gpu_uuid"]
        if not uuid or row["gpu_uuid"] != uuid:
            raise ValueError("D2 telemetry GPU identity changed")
        values = [row[field] for field in (*TARGET_FIELDS, "gpu_utilization_percent")]
        if (
            any(
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for value in values
            )
            or values[-1] > 100
        ):
            raise ValueError("D2 telemetry values are invalid")
    return [row for row in rows if row["monotonic_ns"] <= cutoff_ns]


def target_from_plateau(plateau):
    return {field: plateau[field] for field in TARGET_FIELDS}


def _close(value, reference, margin):
    return reference > 0 and abs(value - reference) <= reference * margin


def _matching(plateau, target, limits):
    return (
        abs(plateau["temperature_c"] - target["temperature_c"])
        <= limits["preheat_temp_median_from_target_c"]
        and _close(
            plateau["sm_clock_mhz"],
            target["sm_clock_mhz"],
            limits["preheat_active_sm_median_relative_from_target"],
        )
        and plateau["memory_clock_mhz"] == target["memory_clock_mhz"]
    )


def first_plateau(samples, started_ns, conditioning, target=None, *, pilot_temperature_range=None):
    """Select the earliest eligible trailing window, never the best window."""
    gate = conditioning["pilot_plateau"]
    minimum_end_ns = started_ns + conditioning["minimum_conditioning_s"] * NS
    width_ns = gate["window_s"] * NS
    max_gap_ns = conditioning["read_only_side_sampler"]["max_gap_s"] * NS
    eligible = [sample for sample in samples if sample["monotonic_ns"] >= started_ns]
    left = 0
    for right, sample in enumerate(eligible):
        end_ns = sample["monotonic_ns"]
        if end_ns < minimum_end_ns:
            continue
        start_ns = end_ns - width_ns
        while eligible[left]["monotonic_ns"] < start_ns:
            left += 1
        window = eligible[left : right + 1]
        stamps = [row["monotonic_ns"] for row in window]
        if (
            len(window) < gate["min_total_samples"]
            or stamps[0] - start_ns > max_gap_ns
            or any(
                next_ns - prev_ns > max_gap_ns
                for prev_ns, next_ns in zip(stamps[:-1], stamps[1:], strict=True)
            )
        ):
            continue
        active = [row for row in window if row["gpu_utilization_percent"] > 0]
        mid_ns = end_ns - width_ns // 2
        early = [row["sm_clock_mhz"] for row in active if row["monotonic_ns"] < mid_ns]
        late = [row["sm_clock_mhz"] for row in active if row["monotonic_ns"] >= mid_ns]
        temp = [row["temperature_c"] for row in window]
        if (
            min(len(early), len(late)) < gate["min_active_samples_per_half"]
            or max(temp) - min(temp) > gate["max_temp_span_c"]
            or not _close(
                median(late),
                median(early),
                gate["max_sm_median_relative_change_between_halves"],
            )
        ):
            continue
        memory, count = Counter(row["memory_clock_mhz"] for row in active).most_common(1)[0]
        if count / len(active) < gate["min_active_memory_clock_mode_share"]:
            continue
        temperature = median(temp)
        if pilot_temperature_range is not None and not (
            pilot_temperature_range[0] <= temperature <= pilot_temperature_range[1]
        ):
            continue
        plateau = {
            "start_ns": start_ns,
            "end_ns": end_ns,
            "sample_count": len(window),
            "active_sample_count": len(active),
            "temperature_c": temperature,
            "sm_clock_mhz": median(early + late),
            "memory_clock_mhz": memory,
            "gpu_uuid": sample["gpu_uuid"],
        }
        if target is None or _matching(plateau, target, conditioning["per_measured_engine"]):
            return plateau
    return None


def pre_t0_admitted(sample, measure_start_ns, target, limits):
    return sample is not None and (
        0
        <= measure_start_ns - sample["monotonic_ns"]
        <= limits["max_age_of_actual_pre_t0_device_sample_s"] * NS
        and abs(sample["temperature_c"] - target["temperature_c"])
        <= limits["actual_pre_t0_temp_from_target_c"]
    )
