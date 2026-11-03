"""Bound D10 host placement to the allocated CPUs and two owned processes."""

import os
import re
import time
from pathlib import Path

CPU_SYSFS = Path("/sys/devices/system/cpu")
PROCFS = Path("/proc")
CLIENT_CORES = 2
MIN_ENGINE_CORES = 4
SNAPSHOT_ATTEMPTS = 4


def _siblings(text):
    cpus = set()
    for part in text.strip().split(","):
        match = re.fullmatch(r"([0-9]+)(?:-([0-9]+))?", part)
        if match is None:
            raise ValueError("invalid D10 Linux CPU sibling list")
        first = int(match[1])
        last = int(match[2]) if match[2] is not None else first
        if not first <= last < 65_536:
            raise ValueError("invalid D10 Linux CPU sibling range")
        cpus.update(range(first, last + 1))
    return cpus


def choose_roles(allocated, *, sysfs=CPU_SYSFS):
    """Keep every allocated CPU and assign whole physical cores to one role."""
    allocated = sorted(allocated)
    if (
        not allocated
        or any(type(cpu) is not int or cpu < 0 for cpu in allocated)
        or len(allocated) != len(set(allocated))
    ):
        raise ValueError("invalid D10 allocation CPU mask")
    allowed = set(allocated)
    cores = {}
    for cpu in allocated:
        directory = sysfs / f"cpu{cpu}"
        topology = directory / "topology"
        package = int((topology / "physical_package_id").read_text())
        core = int((topology / "core_id").read_text())
        siblings = _siblings((topology / "thread_siblings_list").read_text())
        nodes = [int(path.name[4:]) for path in directory.glob("node[0-9]*")]
        if package < 0 or core < 0 or cpu not in siblings or len(nodes) > 1:
            raise ValueError("D10 CPU topology is incomplete")
        row = {
            "package": package,
            "core": core,
            "numa_node": nodes[0] if nodes else None,
            "cpus": sorted(siblings & allowed),
        }
        if cores.setdefault((package, core), row) != row:
            raise ValueError("D10 physical core topology is inconsistent")
    physical = [row for _, row in sorted(cores.items())]
    if sorted(cpu for row in physical for cpu in row["cpus"]) != allocated:
        raise ValueError("D10 physical cores do not partition the allocated CPUs")
    if len(physical) < CLIENT_CORES + MIN_ENGINE_CORES:
        raise ValueError("D10 requires two client and at least four engine physical cores")
    domains = {}
    for row in physical:
        domain = (row["package"], row["numa_node"])
        domains[domain] = domains.get(domain, 0) + 1
    preferred = min(
        domains,
        key=lambda domain: (-domains[domain], domain[0], -1 if domain[1] is None else domain[1]),
    )
    ordered = sorted(physical, key=lambda row: (row["package"], row["numa_node"]) == preferred)
    client = sorted(cpu for row in ordered[:CLIENT_CORES] for cpu in row["cpus"])
    engine = sorted(allowed.difference(client))
    return {"allocated": allocated, "client": client, "engine": engine, "physical_cores": physical}


def _stat(path):
    fields = (path / "stat").read_text().rpartition(")")[2].split()
    if len(fields) < 37:
        raise ValueError("D10 Linux proc stat lacks process identity")
    return {
        "parent_pid": int(fields[1]),
        "pgid": int(fields[2]),
        "start_ticks": int(fields[19]),
        "processor": int(fields[36]),
    }


def _status(path):
    fields = {}
    for line in (path / "status").read_text().splitlines():
        key, delimiter, value = line.partition(":")
        if delimiter:
            fields[key] = value.strip()
    return fields


def _owned(path, pid):
    status = _status(path)
    uids = [int(value) for value in status["Uid"].split()]
    if int(status["Tgid"]) != pid or uids[:2] != [os.getuid(), os.geteuid()]:
        raise ValueError("D10 affinity target is outside the owned thread group")


def _identity(pid, pgid):
    if type(pid) is not int or pid < 2:
        raise ValueError("D10 affinity target PID is invalid")
    path = PROCFS / str(pid)
    _owned(path, pid)
    values = _stat(path)
    if values["pgid"] != pgid or values["start_ticks"] <= 0:
        raise ValueError("D10 affinity target is outside the owned process group")
    return {"pid": pid, **{key: values[key] for key in ("parent_pid", "pgid", "start_ticks")}}


def _thread_ids(pid):
    return sorted(int(path.name) for path in (PROCFS / str(pid) / "task").iterdir())


def _snapshot_process(identity, mask, *, apply=False):
    pid = identity["pid"]
    path = PROCFS / str(pid) / "task"
    for _ in range(SNAPSHOT_ATTEMPTS):
        if _identity(pid, identity["pgid"]) != identity:
            raise ValueError("D10 affinity target identity changed")
        ids = _thread_ids(pid)
        threads = []
        try:
            for tid in ids:
                thread = path / str(tid)
                _owned(thread, pid)
                stat = _stat(thread)
                affinity = sorted(os.sched_getaffinity(tid))
                if apply and affinity != mask:
                    os.sched_setaffinity(tid, mask)
                    affinity = sorted(os.sched_getaffinity(tid))
                runtime, wait, timeslices = map(int, (thread / "schedstat").read_text().split())
                if _stat(thread)["start_ticks"] != stat["start_ticks"]:
                    raise ValueError("D10 affinity target thread identity changed")
                threads.append(
                    {
                        "tid": tid,
                        "start_ticks": stat["start_ticks"],
                        "affinity": affinity,
                        "processor": stat["processor"],
                        "schedstat": {
                            "runtime_ns": runtime,
                            "runnable_wait_ns": wait,
                            "timeslices": timeslices,
                        },
                    }
                )
        except (FileNotFoundError, ProcessLookupError):
            continue
        if ids != _thread_ids(pid):
            continue
        if _identity(pid, identity["pgid"]) != identity or pid not in ids:
            raise ValueError("D10 affinity target changed during the snapshot")
        if any(row["affinity"] != mask for row in threads):
            raise ValueError("D10 live-thread affinity differs from the assigned CPU mask")
        return {**identity, "threads": threads}
    raise ValueError("D10 live-thread CPU evidence did not converge")


class HostPlacement:
    """Own the D10 client/EngineCore placement and its passive boundary evidence."""

    def __init__(self, engine_pid, mode):
        if mode not in {"inherited", "split"}:
            raise ValueError("invalid D10 host placement mode")
        pgid = os.getpgrp()
        client = _identity(os.getpid(), pgid)
        engine = _identity(engine_pid, pgid)
        if (
            client["pid"] != pgid
            or engine["parent_pid"] != client["pid"]
            or engine["start_ticks"] < client["start_ticks"]
        ):
            raise ValueError("D10 native EngineCore is not the replay client's child")
        self._identities = {"client": client, "engine": engine}
        self._roles = choose_roles(os.sched_getaffinity(0))
        self._mode = mode
        self._client_changed = False
        self.evidence = {
            "schema_version": 1,
            "mode": mode,
            "roles": self._roles,
            "setup": None,
            "before_measure": None,
            "after_measure": None,
        }

    def setup(self):
        if self._mode == "split":
            # The clock-bound EngineCore PID belongs to this replay child group.
            # Verify both groups inherited the Slurm mask before changing either.
            for identity in self._identities.values():
                _snapshot_process(identity, self._roles["allocated"])
            _snapshot_process(self._identities["engine"], self._roles["engine"], apply=True)
            self._client_changed = True
            _snapshot_process(self._identities["client"], self._roles["client"], apply=True)
        self.observe("setup")

    def observe(self, boundary):
        started = time.monotonic_ns()
        rows = {
            role: _snapshot_process(
                identity, self._roles[role if self._mode == "split" else "allocated"]
            )
            for role, identity in self._identities.items()
        }
        self.evidence[boundary] = {"started_ns": started, "ended_ns": time.monotonic_ns(), **rows}

    def restore_client(self):
        if self._client_changed:
            _snapshot_process(self._identities["client"], self._roles["allocated"], apply=True)
            self._client_changed = False
