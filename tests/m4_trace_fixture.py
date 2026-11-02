"""Synthetic contract-only raw M4 observations; never native/GPU execution evidence."""

import copy

from vllm_prefix_scheduler.m3_contract import EMPTY_LIST_RESOURCES, ZERO_RESOURCES
from vllm_prefix_scheduler.m4_contract import digest, expected_cases


def drained():
    return {
        **dict.fromkeys(EMPTY_LIST_RESOURCES, []),
        **dict.fromkeys(ZERO_RESOURCES, 0),
        "total_non_null_blocks": 1024,
        "free_count": 1024,
        "free_unique_count": 1024,
        "free_traversed_count": 1024,
        "guard_fault_code": None,
        "cached_block_count": 4,
    }


class Trace:
    def __init__(self, metadata, case_id, *, layer="native", bridge=False):
        self.metadata, self.case_id, self.layer = metadata, case_id, layer
        self.bridge = bridge
        self.events, self.pending = [], {}
        self.next_object, self.batch, self.guard = 0, 0, {}

    def emit(self, kind, **fields):
        seq = len(self.events) + 1
        self.events.append(
            {
                "schema": 1,
                "task_id": "M4-scheduler-stability",
                "event": kind,
                "engine_id": self.case_id,
                "case_id": self.case_id,
                "pid": 10,
                "seq": seq,
                "monotonic_ns": seq * 1000,
                "clock_domain": "monotonic_ns:contract-fixture",
                "layer": self.layer,
                **self.metadata,
                **copy.deepcopy(fields),
            }
        )

    def bound(self, rid="request", prompt=64):
        gen = self.next_object
        self.next_object += 1
        record = {"enqueued_ns": (len(self.events) + 1) * 1000, "sequence": gen}
        request = {
            "request_id": rid,
            "object_id": f"object:{gen}",
            "generation": gen,
            "prompt_length": prompt,
            "status": "WAITING",
            "num_computed_tokens": 0,
            "num_in_flight_tokens": 0,
            "num_stale_output_tokens": 0,
            "num_output_placeholders": 0,
            "num_output_tokens": 0,
            "num_preemptions": 0,
            "is_prefill_chunk": False,
            "guard": record if self.metadata["mode"] == "guarded" else None,
        }
        if request["guard"]:
            self.guard[request["object_id"]] = {"object_id": request["object_id"], **record}
        self.emit("m4_request_bound", request=request, bound=True)
        return request

    def schedule(self, request=None, start=0, end=0, *, notices=None):
        self.batch += 1
        batch = f"{self.case_id}:{self.batch}"
        before = list(self.guard.values())
        intervals = []
        if request:
            intervals = [
                {k: request[k] for k in ("request_id", "object_id", "generation", "prompt_length")}
                | {"start": start, "end": end}
            ]
            request.update(
                status="RUNNING",
                num_computed_tokens=end,
                num_in_flight_tokens=end - start,
                is_prefill_chunk=end < request["prompt_length"],
            )
            self.guard.pop(request["object_id"], None)
        row = {
            "batch_id": batch,
            "intervals": intervals,
            "finished_req_ids": notices or [],
            "preempted_req_ids": [],
            "states": [request] if request else [],
            "guard_before": before,
            "guard_after": list(self.guard.values()),
        }
        self.emit("m4_schedule", **row)
        self.pending[batch] = {
            key: row[key]
            for key in ("batch_id", "intervals", "finished_req_ids", "preempted_req_ids")
        }
        if self.bridge:
            counts = {} if request is None else {request["request_id"]: end - start}
            descriptor = {
                "num_scheduled_tokens": counts,
                "total_num_scheduled_tokens": sum(counts.values()),
                "new_req_ids": list(counts),
                "cached_req_ids": [],
                "finished_req_ids": notices or [],
            }
            descriptor["output_sha256"] = digest(descriptor)
            self.emit("schedule_success", batch_id=batch, intervals=intervals, **descriptor)
            self.emit("runner_execute_start", batch_id=batch, **descriptor)
            self.emit(
                "runner_execute_success",
                batch_id=batch,
                result_kind="none" if counts else "output",
                forward_observed=bool(counts),
                forward_tokens=sum(counts.values()),
                forward_tokens_padded=sum(counts.values()),
                forward_req_ids=list(counts),
            )
            if counts:
                self.emit("runner_sample_start", batch_id=batch)
                self.emit("runner_sample_success", batch_id=batch, result_kind="async")
            material = {
                "output_type": "vllm.v1.outputs.ModelRunnerOutput",
                "req_ids": list(counts),
                "sampled_token_ids": [[7] for _ in counts],
            }
            self.emit("model_output_materialized", batch_id=batch, **material)
            self.pending[batch]["_material"] = material
        return batch

    def consume(self, batch):
        row = self.pending[batch]
        self.emit("m4_consume_start", batch_id=batch, intervals=row["intervals"], states=[])
        if self.bridge:
            self.emit("update_from_output_start", batch_id=batch, **row["_material"])
            self.emit(
                "update_from_output_success", batch_id=batch, engine_outputs=[], **row["_material"]
            )
        self.emit(
            "m4_consume_success", batch_id=batch, intervals=row["intervals"], states=[], outputs=[]
        )
        del self.pending[batch]

    def terminal(self, request, aborted=False):
        request["status"] = "FINISHED_ABORTED" if aborted else "FINISHED_LENGTH_CAPPED"
        self.guard.pop(request["object_id"], None)
        request["guard"] = None
        self.emit("m4_request_terminal", request=request)

    def fence(self):
        self.emit("m4_generation_drained", resources=drained())

    def cancel(self, request):
        self.emit(
            "m4_cancel_start",
            requested_ids=request["request_id"],
            requests=[request],
            pending=[
                {k: v for k, v in row.items() if not k.startswith("_")}
                for row in self.pending.values()
            ],
        )
        self.terminal(request, True)
        self.emit("m4_cancel_success", requests=[request], pending=[])

    def complete(self, rid="request", *, chunks=2):
        request = self.bound(rid)
        for index in range(chunks):
            batch = self.schedule(request, index * 64 // chunks, (index + 1) * 64 // chunks)
            self.consume(batch)
        self.terminal(request)
        batch = self.schedule(notices=[rid])
        self.consume(batch)
        self.fence()
        return request


def gpu_fixture(config):
    all_events = []
    for case, meta in expected_cases(config, "gpu").items():
        trace = Trace(meta, case, layer="gpu", bridge=True)
        trace.emit("m4_case_begin", initial_resources=drained(), interventions=[])
        trace.fence()
        scenario = meta["scenario"]
        if scenario in {"waiting_cancel_reuse", "pending_cancel_reuse"}:
            request = trace.bound()
            batch = None
            if scenario == "pending_cancel_reuse":
                batch = trace.schedule(request, 0, 32)
            trace.cancel(request)
            if batch:
                trace.consume(batch)
            batch = trace.schedule(notices=["request"])
            trace.consume(batch)
            trace.fence()
            trace.complete()
        elif scenario == "capacity_preempt_resume":
            request = trace.bound()
            batch = trace.schedule(request, 0, 32)
            trace.consume(batch)
            before = copy.deepcopy(request)
            trace.emit("m4_preempt_start", request=before, pending=[])
            request.update(status="PREEMPTED", num_computed_tokens=0, num_preemptions=1)
            trace.emit("m4_preempt_success", before=before, request=request, pending=[])
            batch = trace.schedule(request, 0, 32)
            for event in trace.events:
                if event["event"] == "m4_schedule" and event["batch_id"] == batch:
                    event["preempted_req_ids"] = [request["request_id"]]
            trace.consume(batch)
            batch = trace.schedule(request, 32, 64)
            trace.consume(batch)
            trace.terminal(request)
            trace.fence()
        else:
            if scenario == "full_events_probe":
                for kind in ("m4_probe_before", "m4_probe_after"):
                    trace.emit(
                        kind,
                        probe_id="p",
                        flags=dict.fromkeys(("full", "events", "stats", "metrics"), True),
                        snapshot={"pool": {"blocks": [1, 2]}, "metrics": {"queries": 2}},
                        score=16,
                    )
            trace.complete()
        trace.emit("m4_case_end", status="passed", resources=drained())
        all_events.extend(trace.events)
    return all_events


def resource_fixture(directory):
    """Synthetic private resource originals for contract controls only."""
    import json

    from vllm_prefix_scheduler.m4_contract import sha256

    directory.mkdir(parents=True, exist_ok=True)
    control = directory / "control" / "attempt-01"
    control.mkdir(parents=True)
    domain, commit, config_hash, job = (
        "monotonic_ns:controller:fixture",
        "a" * 40,
        "b" * 64,
        "12345",
    )

    def write(path, value):
        path.write_text(json.dumps(value) + "\n")
        return {"path": str(path), "sha256": sha256(path)}

    ready = {
        "ready": True,
        "campaign_id": "M4-scheduler-stability",
        "campaign_sha256": config_hash,
        "attempt_id": "attempt-01",
        "project_commit": commit,
        "clock_domain": domain,
        "observed_monotonic_ns": 100,
        "pid": 1,
    }
    receipt = {
        "returncode": 0,
        "stdout": job + "\n",
        "command": ["sbatch", "fixture.sbatch"],
        "clock_domain": domain,
        "started_monotonic_ns": 200,
        "finished_monotonic_ns": 300,
    }
    terminal = {
        "returncode": 0,
        "command": ["ssh", f"scontrol show job {job}"],
        "stdout": f"JobId={job} JobState=COMPLETED ExitCode=0:0 TimeLimit=00:20:00 AllocTRES=gres/gpu=1 StartTime=2026-11-03T01:00:00 EndTime=2026-11-03T01:10:00",
        "clock_domain": domain,
        "observed_monotonic_ns": 400,
    }
    exited = {
        "returncode": 0,
        "pid": 1,
        "owned_pids_remaining": [],
        "clock_domain": domain,
        "observed_monotonic_ns": 500,
    }
    query = {
        "returncode": 0,
        "command": ["ssh", 'squeue -h -o "%i|%T|%j"'],
        "stdout": "",
        "clock_domain": domain,
        "observed_monotonic_ns": 450,
    }
    gpu = {
        "returncode": 0,
        "stdout": "NVIDIA GeForce RTX 3090, 24576, 100, 24476, 0\n",
        "monotonic_ns": 10,
    }
    checkpoints = {
        "job_id": job,
        "owned_pids_remaining": [],
        "gpu_before": gpu,
        "gpu_during": [gpu],
        "gpu_after": gpu,
    }
    exits = {
        "job_id": job,
        "project_commit": commit,
        "owned_pids_remaining": [],
        "native": 0,
        "gpu": 0,
    }
    steps = [
        {"name": name, "returncode": 0, "status": "exited", "remaining": []}
        for name in ("native", "cpu", "inputs", "gpu", "regressions")
    ]
    exits.update(returncode=0, steps=steps)
    checkpoints.update(launcher_steps=steps, cleanup_errors=[])
    binding = {"job_id": job, "project_commit": commit, "config_sha256": config_hash}
    paths = {
        "submission": write(control / "submission.json", receipt),
        "terminal": write(control / "terminal.json", terminal),
        "checkpoints": write(control / "checkpoints.json", checkpoints),
        "exit": write(control / "exit.json", exits),
        "binding": write(control / "binding.json", binding),
    }
    trace = control / "watch.jsonl"
    trace.write_text(json.dumps(terminal) + "\n")
    attempt = {
        "attempt_id": "attempt-01",
        "project_commit": commit,
        "config_sha256": config_hash,
        "status": "submitted_terminal",
        "job_id": job,
        "submission": paths["submission"],
        "monitor_ready": write(control / "ready.json", ready),
        "monitor_exit": write(control / "watcher-exit.json", exited),
        "monitor_trace": {"path": str(trace), "sha256": sha256(trace)},
    }
    ledger = {
        "task_id": "M4-scheduler-stability",
        "config_sha256": config_hash,
        "attempts": [attempt],
    }
    return {
        "task_id": "M4-scheduler-stability",
        "config_sha256": config_hash,
        "submission_ledger": write(directory / "ledger.json", ledger),
        "submission_catalog_dir": str(control.parent),
        "active_query": query,
        "active_query_evidence": write(directory / "active.json", query),
        "allocations": [
            {
                "job_id": job,
                "state": "COMPLETED",
                "exit_code": "0:0",
                "gpus": 1,
                "limit_minutes": 20,
                "evidence": paths,
            }
        ],
    }
