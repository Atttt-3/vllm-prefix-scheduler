"""Independently check endpoint-stability intervention evidence for D10."""

from collections import Counter
from itertools import combinations

from check_m5_d1 import compare_pair, matched_requests
from m5_d7_audit import waiting_summary
from m5_d9_audit import EARLY, signature

from vllm_prefix_scheduler.m5_campaign import read_json, require

ENDPOINTS = ("E1", "E2", "E3", "E4")
STAT_FIELDS = ("runtime_ns", "runnable_wait_ns", "timeslices")


def core_partition(roles):
    """Reconstruct masks from recorded physical core groups."""
    allocated, client, engine = (roles[key] for key in ("allocated", "client", "engine"))
    groups = roles["physical_cores"]
    require(
        all(
            type(mask) is list
            and mask
            and all(type(cpu) is int and cpu >= 0 for cpu in mask)
            and mask == sorted(set(mask))
            for mask in (allocated, client, engine)
        )
        and len(allocated) == 8
        and set(client).isdisjoint(engine)
        and set(client) | set(engine) == set(allocated),
        "D10 role masks do not partition the allocated CPUs",
    )
    physical = set()
    domains = Counter()
    inventory = []
    client_cores = 0
    engine_cores = 0
    for group in groups:
        package, core, node, cpus = (group[key] for key in ("package", "core", "numa_node", "cpus"))
        require(
            type(package) is int
            and package >= 0
            and type(core) is int
            and core >= 0
            and (node is None or type(node) is int and node >= 0)
            and type(cpus) is list
            and cpus
            and all(type(cpu) is int and cpu >= 0 for cpu in cpus)
            and cpus == sorted(set(cpus))
            and (package, core) not in physical,
            "D10 physical core inventory is malformed",
        )
        physical.add((package, core))
        domains[(package, node)] += 1
        inventory.extend(cpus)
        if set(cpus) <= set(client):
            client_cores += 1
        elif set(cpus) <= set(engine):
            engine_cores += 1
        else:
            raise ValueError("D10 SMT sibling cores cross placement roles")
    require(
        sorted(inventory) == allocated
        and client_cores == 2
        and engine_cores >= 4
        and [(row["package"], row["core"]) for row in groups] == sorted(physical),
        "D10 physical core placement differs from the frozen split",
    )
    preferred = min(
        domains,
        key=lambda domain: (-domains[domain], domain[0], -1 if domain[1] is None else domain[1]),
    )
    selection = sorted(groups, key=lambda row: (row["package"], row["numa_node"]) == preferred)[:2]
    require(
        sorted(cpu for row in selection for cpu in row["cpus"]) == client,
        "D10 client mask differs from the frozen physical core selection",
    )
    return {"allocated": allocated, "client": client, "engine": engine, "physical_cores": groups}


def placement_snapshot(snapshot, *, pids, masks, previous_end, expected_group=None):
    start, end = (snapshot[key] for key in ("started_ns", "ended_ns"))
    require(
        type(start) is int and type(end) is int and previous_end <= start <= end,
        "D10 affinity snapshot chronology differs",
    )
    table = {}
    group = None
    for role in ("client", "engine"):
        process = snapshot[role]
        pid, parent, pgid, birth = (
            process[key] for key in ("pid", "parent_pid", "pgid", "start_ticks")
        )
        require(
            pid == pids[role]
            and all(type(x) is int and x > 0 for x in (pid, parent, pgid, birth))
            and (role != "engine" or parent == pids["client"])
            and (expected_group is None or pgid == expected_group)
            and (group is None or pgid == group),
            "D10 CPU process identity or group differs from the engine binding",
        )
        group = pgid
        tids = {}
        for thread in process["threads"]:
            tid, ticks = thread["tid"], thread["start_ticks"]
            stats = thread["schedstat"]
            require(
                type(tid) is int
                and tid > 0
                and type(ticks) is int
                and ticks >= birth
                and tid not in tids
                and thread["affinity"] == masks[role]
                and type(thread["processor"]) is int
                and thread["processor"] >= 0
                and set(stats) == set(STAT_FIELDS)
                and all(type(stats[key]) is int and stats[key] >= 0 for key in STAT_FIELDS),
                f"D10 {role} thread affinity or schedstat differs",
            )
            tids[tid] = thread
        require(tids.get(pid, {}).get("start_ticks") == birth, "D10 main thread identity missing")
        table[role] = {"birth": birth, "threads": tids}
    return end, group, table


def observed_placement(
    client, engine, identity, order, *, thermal=None, leaf_end_ns=None, pgid=None
):
    proof = client["d10_placement"]
    mode = order[identity]
    require(
        type(proof["schema_version"]) is int
        and proof["schema_version"] == 1
        and proof["mode"] == mode,
        "D10 placement role differs from the frozen ABBA order",
    )
    roles = core_partition(proof["roles"])
    if mode == "split":
        masks = roles
    else:
        masks = dict.fromkeys(("client", "engine"), roles["allocated"])
    pids = {role: item["clock"]["pid"] for role, item in (("client", client), ("engine", engine))}
    require(pids["client"] != pids["engine"], "D10 client and EngineCore process are the same")
    setup_end, group, setup = placement_snapshot(
        proof["setup"], pids=pids, masks=masks, previous_end=0, expected_group=pgid
    )
    if thermal and thermal["conditioning_start_ns"] is not None:
        require(
            setup_end <= thermal["conditioning_start_ns"],
            "D10 placement was installed after conditioning began",
        )
    before = proof["before_measure"]
    after = proof["after_measure"]
    phases = client["phases"]
    if "measure_start_ns" not in phases:
        require(after is None, "D10 untimed leaf has post-measure affinity evidence")
        if before is not None:
            before_end, _, prepared = placement_snapshot(
                before, pids=pids, masks=masks, previous_end=setup_end, expected_group=group
            )
            require(
                before_end <= leaf_end_ns
                and all(prepared[role]["birth"] == setup[role]["birth"] for role in pids),
                "D10 untimed leaf affinity evidence is outside the bound process",
            )
        return {"verified": True, "mode": mode, "roles": roles, "schedstat_delta": None}
    require(identity in ENDPOINTS, "D10 pilot contains measured data")
    before_end, _, prev = placement_snapshot(
        before, pids=pids, masks=masks, previous_end=setup_end, expected_group=group
    )
    require(
        before_end <= phases["measure_start_ns"],
        "D10 affinity was not verified before the first request",
    )
    if "drain_end_ns" not in phases:
        require(after is None, "D10 undrained leaf has a completed affinity snapshot")
        return {"verified": False, "mode": mode, "roles": roles, "schedstat_delta": None}
    after_end, _, nxt = placement_snapshot(
        after,
        pids=pids,
        masks=masks,
        previous_end=phases["drain_end_ns"],
        expected_group=group,
    )
    require(
        leaf_end_ns is not None and after_end <= leaf_end_ns, "D10 affinity snapshot exceeds leaf"
    )
    totals = {}
    for role in ("client", "engine"):
        require(
            setup[role]["birth"] == prev[role]["birth"] == nxt[role]["birth"],
            "D10 placement snapshots crossed a process generation",
        )
        left, right = prev[role]["threads"], nxt[role]["threads"]
        matching = [
            tid
            for tid in left.keys() & right.keys()
            if left[tid]["start_ticks"] == right[tid]["start_ticks"]
        ]
        changes = [
            {key: right[tid]["schedstat"][key] - left[tid]["schedstat"][key] for key in STAT_FIELDS}
            for tid in matching
        ]
        require(
            pids[role] in matching and all(delta >= 0 for row in changes for delta in row.values()),
            "D10 schedstat counters or main thread continuity differ",
        )
        totals[role] = {
            "matching_threads": len(matching),
            "before_threads": len(left),
            "after_threads": len(right),
            **{key: sum(row[key] for row in changes) for key in STAT_FIELDS},
        }
    return {"verified": True, "mode": mode, "roles": roles, "schedstat_delta": totals}


def placement_assessment(runs, clients, gates, order, *, thermal_verified):
    controls = tuple(leaf for leaf in ENDPOINTS if order[leaf] == "inherited")
    isolated = tuple(leaf for leaf in ENDPOINTS if order[leaf] == "split")
    require(len(controls) == len(isolated) == 2, "D10 placement pair inventory differs")
    control_pair = "/".join(controls)
    isolated_pair = "/".join(isolated)
    performance_pairs = (controls, isolated, *((a, b) for a in controls for b in isolated))
    pairs = {
        f"{a}/{b}": compare_pair(runs[a]["metrics"], runs[b]["metrics"], gates)
        if all(runs.get(key, {}).get("endpoint_evidence_complete") for key in (a, b))
        else None
        for a, b in performance_pairs
    }
    signatures = {
        leaf: signature(clients[leaf], EARLY) if leaf in clients else None for leaf in ENDPOINTS
    }
    waiting = {
        leaf: waiting_summary(clients[leaf], runs[leaf].get("native_timing", {}))
        for leaf in clients
    }
    verified = thermal_verified and all(
        runs.get(leaf, {}).get("endpoint_evidence_complete")
        and runs[leaf].get("placement", {}).get("verified")
        and runs[leaf].get("native_timing", {}).get("status") == "complete"
        and runs[leaf]["native_timing"]["clock"]["verified"]
        and signatures[leaf]["candidate"] != "unverified"
        for leaf in ENDPOINTS
    )
    slow = [leaf for leaf in controls if (signatures[leaf] or {}).get("candidate") == "detected"]
    checks = {
        "all_original_gates_observable": bool(verified),
        "control_a_a_ttft_unstable": verified and not pairs[control_pair]["ttft_within_tolerance"],
        "control_early_slow": slow,
    }
    if not verified:
        result, reason = (
            "unresolved",
            "required_placement_endpoint_clock_or_thermal_evidence_missing",
        )
    elif not checks["control_a_a_ttft_unstable"] or not slow:
        result, reason = (
            "inconclusive",
            "inherited_endpoint_instability_and_early_signature_not_reproduced",
        )
    else:
        starts = {
            at
            for leaf in slow
            for trigger in signatures[leaf]["trigger_starts_offset_ns"]
            for at in (trigger, trigger + 500_000_000)
        }
        matching_cadence = {
            leaf: signatures[leaf]["candidate"] == "not_detected"
            and starts
            <= {
                row["start_offset_ns"]
                for row in signatures[leaf]["bins"]
                if row["eligible"] and not row["above_threshold"]
            }
            for leaf in isolated
        }
        improved = {
            f"{a}/{b}": pairs[f"{a}/{b}"]["ttft_delta_ns"] < -pairs[f"{a}/{b}"]["ttft_tolerance_ns"]
            for a in slow
            for b in isolated
        }
        no_harm = {
            f"{a}/{b}": pair["ttft_delta_ns"] <= pair["ttft_tolerance_ns"]
            and all(
                value >= -gates["overhead_throughput_rel"]
                for value in pair["relative_throughput_delta"].values()
            )
            for a in controls
            for b in isolated
            if (pair := pairs[f"{a}/{b}"])
        }
        checks.update(
            isolated_b_b_stable=pairs[isolated_pair]["within_tolerance"],
            matched_early_cadence=matching_cadence,
            ttft_improved_against_slow_control=improved,
            no_harm_against_either_control=no_harm,
        )
        supported = (
            checks["isolated_b_b_stable"]
            and all(matching_cadence.values())
            and all(improved.values())
            and all(no_harm.values())
        )
        result = "supported" if supported else "not_supported"
        reason = (
            "matched_stability_improvement" if supported else "frozen_intervention_criterion_failed"
        )
    matched = {
        f"{a}/{b}": matched_requests(
            *({row["request_id"]: row for row in clients[leaf]["requests"]} for leaf in (a, b))
        )
        for a, b in performance_pairs
        if pairs[f"{a}/{b}"] is not None
    }
    return {
        "status": result,
        "reason": reason,
        "checks": checks,
        "pairs": pairs,
        "matched_requests": matched,
        "early_signatures": signatures,
        "waiting": waiting,
    }


def d10_report(
    node,
    lineage,
    commit,
    resource,
    history,
    target,
    runs,
    clients,
    gates,
    limits,
    order,
    compare_thermal,
):
    leaf_ok = {
        leaf: runs.get(leaf, {}).get("endpoint_evidence_complete") is True
        and runs[leaf].get("placement", {}).get("verified") is True
        and runs[leaf].get("native_timing", {}).get("status") == "complete"
        and runs[leaf]["native_timing"]["clock"]["verified"]
        for leaf in ENDPOINTS
    }
    for first, second in zip(ENDPOINTS, ENDPOINTS[1:], strict=False):
        require(
            second not in runs or leaf_ok[first],
            "D10 continued after incomplete endpoint or placement evidence",
        )
    node_result = node / "d10-result.json"
    if node_result.exists():
        require(
            read_json(node_result)
            == {
                **lineage,
                "status": "diagnostic-complete"
                if all(leaf_ok.values())
                else "diagnostic-incomplete",
                "audited_runs": sum("metrics" in run for run in runs.values()),
            },
            "D10 node result differs from independent evidence",
        )
    elif resource["complete"]:
        raise ValueError("Completed D10 lacks the node result")
    thermal_pairs = [
        {
            "reference": a,
            "observed": b,
            "thermal": compare_thermal(runs[a]["in_window"], runs[b]["in_window"], limits)
            if all("in_window" in runs.get(leaf, {}) for leaf in (a, b))
            else None,
        }
        for a, b in combinations(ENDPOINTS, 2)
    ]
    thermal_verified = (
        resource["complete"]
        and target is not None
        and all(
            runs.get(leaf, {}).get("pre_t0_verified") is True and runs[leaf]["prelude"]["verified"]
            for leaf in ENDPOINTS
        )
        and all(row["thermal"] and row["thermal"]["comparable"] for row in thermal_pairs)
    )
    assessment = placement_assessment(
        runs, clients, gates, order, thermal_verified=thermal_verified
    )
    return {
        **lineage,
        "retained_original_C3": {
            "status": "blocked",
            "stage_result_sha256": lineage["retained_c3_sha256"],
        },
        "status": "diagnostic-complete"
        if thermal_verified and all(leaf_ok.values())
        else "diagnostic-incomplete",
        "execution_commit": commit,
        "analysis_commit": commit,
        "resource_release": resource,
        "resource_ledger": history,
        "verified_thermal_target": target,
        "runs": runs,
        "thermal_pairs": thermal_pairs,
        "assessment": assessment,
        "decision": {
            "thermal_control": "verified" if thermal_verified else "unverified",
            "placement_stability_improvement": assessment["status"],
            "qualifies_C3": False,
            "qualification_started": False,
        },
    }
