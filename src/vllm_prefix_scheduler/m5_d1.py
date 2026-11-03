"""Validate the single approved C3-D1 diagnostic against retained C3 inputs."""

from .m5_campaign import digest, read_json, require

D1_PROPOSAL_SHA = "c5362bab5611a301f483363c909334165714a2c34d1352efe7299f5740dd79ec"
D1_INPUT_SHA = "e52cd3e00563b3e7827b089faa3e4a8aaffafa8a873867b2dfde8c6b5ea713ab"
C3_BLOCKED_SHA = "e8618fde6793d2c2d749e32b7f8052a495f695ab1ddc68dc015ac58012495866"


def d1_specs(config, proposal, table, original):
    require(digest(proposal) == D1_PROPOSAL_SHA, "D1 reviewed plan differs")
    require(digest(table) == D1_INPUT_SHA, "D1 archived input differs")
    require(
        digest(original) == C3_BLOCKED_SHA and read_json(original)["status"] == "blocked",
        "D1 requires the retained blocked C3",
    )
    reviewed = read_json(proposal)
    limits = reviewed["guardrails"]
    gates = config["gates"]
    require(
        limits["ttft_p95_absolute_floor_ns"] == gates["overhead_ttft_ns"]
        and limits["ttft_p95_relative_margin"] == gates["overhead_ttft_rel"]
        and limits["throughput_both_relative_margin"] == gates["overhead_throughput_rel"]
        and limits["allocation_work_deadline_seconds"] == gates["max_allocation_work_s"],
        "D1 thresholds differ from C3",
    )
    ordered = [
        {
            "id": run["id"],
            "scenario": block["scenario"],
            "seed": block["seed"],
            "count": block["count"],
            "mode": block["mode"],
            "tier": run["tier"],
            "rate_rps": block["rate_rps"],
            "interval_ms": None,
            "mechanism": None,
        }
        for block in reviewed["diagnostic"]["blocks"]
        for run in block["ordered_runs"]
    ]
    workload = read_json(table)
    require(
        all(
            workload["scenario"] == row["scenario"]
            and workload["seed"] == row["seed"]
            and workload["rate_rps"] == row["rate_rps"]
            and len(workload["requests"]) == row["count"]
            for row in ordered
        ),
        "D1 input and frozen run specs disagree",
    )
    return ordered
