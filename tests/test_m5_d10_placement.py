"""Prove that D10 partitions only allocated CPUs and touches only owned threads."""

from types import SimpleNamespace

import pytest

from vllm_prefix_scheduler import m5_d10_placement as placement

ALLOCATED = list(range(8))
CLIENT_PID = 200
ENGINE_PID = 300
UID = 500


def fake_topology(root, rows):
    for logical, package, core, node, siblings in rows:
        cpu = root / f"cpu{logical}"
        path = cpu / "topology"
        path.mkdir(parents=True)
        (path / "physical_package_id").write_text(str(package))
        (path / "core_id").write_text(str(core))
        (path / "thread_siblings_list").write_text(siblings)
        if node is not None:
            (cpu / f"node{node}").mkdir()


def test_d10_prefers_engine_numa_without_splitting_siblings(tmp_path):
    rows = [(logical, 0, logical, 0 if logical <= 5 else 1, str(logical)) for logical in ALLOCATED]
    rows[0] = (0, 0, 0, 0, "0,8")
    rows += [(8, 0, 0, 0, "0,8")]
    fake_topology(tmp_path, rows)
    evidence = placement.choose_roles([*ALLOCATED, 8], sysfs=tmp_path)
    assert evidence["client"] == [6, 7]
    assert evidence["engine"] == [0, 1, 2, 3, 4, 5, 8]
    assert evidence["physical_cores"][0] == {
        "package": 0,
        "core": 0,
        "numa_node": 0,
        "cpus": [0, 8],
    }
    assert placement.choose_roles(reversed([*ALLOCATED, 8]), sysfs=tmp_path) == evidence


def test_d10_rejects_insufficient_physical_cores(tmp_path):
    fake_topology(
        tmp_path,
        [(cpu, 0, cpu % 4, 0, f"{cpu % 4},{cpu % 4 + 4}") for cpu in ALLOCATED],
    )
    with pytest.raises(ValueError, match="at least four engine physical cores"):
        placement.choose_roles(ALLOCATED, sysfs=tmp_path)


def proc_entry(path, *, tgid, ppid, uid=UID, ticks=100):
    path.mkdir(parents=True, exist_ok=True)
    fields = ["0"] * 37
    fields[0] = "S"
    fields[1] = str(ppid)
    fields[2] = str(CLIENT_PID)
    fields[19] = str(ticks)
    fields[36] = "7"
    (path / "stat").write_text(f"{path.name} (process with ) parens) " + " ".join(fields))
    (path / "status").write_text(f"Tgid:\t{tgid}\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n")
    (path / "schedstat").write_text("500 20 10\n")


@pytest.fixture
def topology_and_processes(tmp_path, monkeypatch):
    topology = tmp_path / "sys"
    proc = tmp_path / "proc"
    fake_topology(topology, [(cpu, 0, cpu, None, str(cpu)) for cpu in ALLOCATED])
    roles = placement.choose_roles(ALLOCATED, sysfs=topology)
    masks = {tid: set(ALLOCATED) for tid in (CLIENT_PID, 201, ENGINE_PID, 301)}
    changes = []
    for pid, ppid, ticks in ((CLIENT_PID, 100, 200), (ENGINE_PID, CLIENT_PID, 300)):
        proc_entry(proc / str(pid), tgid=pid, ppid=ppid, ticks=ticks)
        for tid in (pid, pid + 1):
            proc_entry(proc / str(pid) / "task" / str(tid), tgid=pid, ppid=ppid, ticks=ticks)

    def setaffinity(tid, mask):
        changes.append((tid, list(mask)))
        masks[tid] = set(mask)

    monkeypatch.setattr(placement, "PROCFS", proc)
    monkeypatch.setattr(placement, "choose_roles", lambda allocated: roles)
    monkeypatch.setattr(
        placement,
        "os",
        SimpleNamespace(
            getuid=lambda: UID,
            geteuid=lambda: UID,
            getpid=lambda: CLIENT_PID,
            getpgrp=lambda: CLIENT_PID,
            sched_getaffinity=lambda tid: masks[tid or CLIENT_PID],
            sched_setaffinity=setaffinity,
        ),
    )
    return proc, roles, masks, changes


def test_d10_inherited_never_calls_affinity_setter(topology_and_processes):
    _, roles, _, changes = topology_and_processes
    host = placement.HostPlacement(ENGINE_PID, "inherited")
    host.setup()
    host.observe("before_measure")
    host.observe("after_measure")
    host.restore_client()
    assert not changes
    assert host.evidence["setup"]["client"]["threads"][0]["affinity"] == roles["allocated"]


def test_d10_split_checks_all_live_threads_and_restores_client(topology_and_processes):
    proc, roles, masks, changes = topology_and_processes
    host = placement.HostPlacement(ENGINE_PID, "split")
    host.setup()
    assert changes == [
        (ENGINE_PID, roles["engine"]),
        (301, roles["engine"]),
        (CLIENT_PID, roles["client"]),
        (201, roles["client"]),
    ]
    host.observe("before_measure")
    masks[202] = set(roles["client"])
    proc_entry(proc / "200" / "task" / "202", tgid=200, ppid=100, ticks=400)
    host.observe("after_measure")
    threads = host.evidence["after_measure"]["client"]["threads"]
    assert [row["tid"] for row in threads] == [200, 201, 202]
    assert all(row["affinity"] == roles["client"] for row in threads)
    assert all(
        row["schedstat"] == {"runtime_ns": 500, "runnable_wait_ns": 20, "timeslices": 10}
        for row in threads
    )
    host.restore_client()
    assert all(masks[tid] == set(ALLOCATED) for tid in (200, 201, 202))
    assert all(masks[tid] == set(roles["engine"]) for tid in (300, 301))


def test_d10_refuses_foreign_thread_before_mutating_either_process(topology_and_processes):
    proc, _, _, changes = topology_and_processes
    proc_entry(proc / "300" / "task" / "301", tgid=999, ppid=200)
    with pytest.raises(ValueError, match="outside the owned thread group"):
        placement.HostPlacement(ENGINE_PID, "split").setup()
    assert not changes


def test_d10_refuses_same_group_non_child(topology_and_processes):
    proc, _, _, changes = topology_and_processes
    proc_entry(proc / "300", tgid=300, ppid=100, ticks=300)
    with pytest.raises(ValueError, match="not the replay client's child"):
        placement.HostPlacement(ENGINE_PID, "split")
    assert not changes


def test_d10_rejects_new_mask_drift_and_process_reuse(topology_and_processes):
    proc, roles, masks, _ = topology_and_processes
    host = placement.HostPlacement(ENGINE_PID, "split")
    host.setup()
    masks[301] = set(ALLOCATED)
    with pytest.raises(ValueError, match="live-thread affinity differs"):
        host.observe("before_measure")
    masks[301] = set(roles["engine"])
    proc_entry(proc / "300", tgid=300, ppid=200, ticks=350)
    with pytest.raises(ValueError, match="target identity changed"):
        host.observe("after_measure")
