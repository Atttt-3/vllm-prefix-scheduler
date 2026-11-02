"""Complete SYNTHETIC contract fixture. Never represents GPU execution evidence."""

import copy
import hashlib
import json
from pathlib import Path

from vllm_prefix_scheduler.m3_contract import (
    MODEL_REVISION,
    canonical,
    case_catalog,
    sha256,
)

ROOT = Path(__file__).resolve().parents[1]


def build_fixture(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "configs/m3-real-model.json"
    config = json.loads(config_path.read_text())
    texts = [r["text"] for r in config["requests"].values()] + [
        f"Give one short fact about {s}." for s in ("GPUs", "oceans", "planets")
    ]
    inputs = {
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "by_text": {text: list(range(32)) for text in texts},
    }
    inputs_path = directory / "inputs.json"
    inputs_path.write_text(canonical(inputs))
    config_hash, input_hash = sha256(config_path), sha256(inputs_path)
    modules = {
        **json.loads((ROOT / "configs/native-m2b.json").read_text())["source_sha256"],
        **config["extra_upstream_sha256"],
    }
    source = {
        f"src/vllm_prefix_scheduler/{name}.py": "0" * 64
        for name in (
            "scheduler",
            "config",
            "probe",
            "wait_guard",
            "m3_contract",
            "m3_events",
            "m3_observer",
            "m3_worker",
        )
    }
    source["configs/m3-real-model.json"] = config_hash
    sources = {
        "project_commit": "1" * 40,
        "project_dirty": False,
        "python_version": "3.12.16",
        "packages": {
            "vllm": "0.32.0",
            "torch": "2.13.0+cu129",
            "transformers": "5.8.0",
            "tokenizers": "0.22.2",
            "safetensors": "0.6.2",
        },
        "upstream_commit": config["upstream_commit"],
        "source_sha256": source,
        "upstream_sha256": modules,
        "model": {
            "revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "directory": "/synthetic-contract-fixture-model",
            "tree_sha256": "2" * 64,
            "files": [
                {
                    "name": name,
                    "size": item["size"],
                    "revision": MODEL_REVISION,
                    "sha256": item["content_id"] if item["kind"] == "lfs_sha256" else "3" * 64,
                    "verified_content_id": item["content_id"],
                }
                for name, item in config["model_files"].items()
            ],
        },
    }
    all_events = []
    expected = case_catalog()
    native_names = (
        "vllm.v1.engine.core",
        "vllm.v1.core.sched.scheduler",
        "vllm.v1.core.sched.async_scheduler",
        "vllm.v1.worker.gpu_worker",
        "vllm.v1.worker.gpu.model_runner",
        "vllm.v1.worker.gpu.async_utils",
        "vllm.v1.executor.uniproc_executor",
    )

    class Engine:
        def __init__(self, name, pid):
            self.name, self.pid = name, pid
            self.phase, self.mode = name.split("-")
            self.seq, self.tick, self.batch, self.next_guard = 0, 1_000_000_000, 0, 0
            self.guards, self.finished, self.cached = {}, {}, 0
            self.closed = False
            self.emit(
                "engine_start",
                production_class=self.production,
                scheduler_class=self.actual,
                project_commit=sources["project_commit"],
                source_sha256=source,
                upstream_sha256=modules,
                config_sha256=config_hash,
                inputs_sha256=input_hash,
                requested_config={
                    "model": sources["model"]["directory"],
                    "tokenizer": sources["model"]["directory"],
                },
            )
            if self.phase == "diagnostic":
                self.emit(
                    "scheduler_init",
                    actual_type=self.actual,
                    mro=[
                        self.actual,
                        self.production,
                        "vllm.v1.core.sched.async_scheduler.AsyncScheduler",
                    ],
                    modules={
                        **{
                            k: {
                                "path": "/synthetic/" + k,
                                "sha256": source[f"src/vllm_prefix_scheduler/{k}.py"],
                            }
                            for k in ("scheduler", "config", "probe", "wait_guard")
                        },
                        "observer": {"path": "/synthetic/observer", "sha256": "0" * 64},
                        "native_async_scheduler": {
                            "path": "/synthetic/native",
                            "sha256": modules["vllm/v1/core/sched/async_scheduler.py"],
                        },
                    },
                    effective_config={
                        **self.effective(),
                        "mode": self.mode,
                        "dtype": "torch.bfloat16",
                        "window": 16 if self.mode != "native" else None,
                        "max_wait_ms": 500 if self.mode != "native" else None,
                        "block_size": 16,
                        "hash_block_size": 16,
                        "num_gpu_blocks": 101,
                        "kv_spec_types": ["vllm.v1.kv_cache_interface.FullAttentionSpec"],
                        "max_num_active_reqs": 1,
                    },
                )
            self.emit("engine_ready", sources=sources, worker=[self.worker("boundary")])

        @property
        def production(self):
            return (
                "vllm.v1.core.sched.async_scheduler.AsyncScheduler"
                if self.mode == "native"
                else "vllm_prefix_scheduler.scheduler.PrefixAsyncScheduler"
            )

        @property
        def actual(self):
            if self.phase == "smoke":
                return self.production
            return (
                "vllm_prefix_scheduler.m3_observer.ObserverNativeScheduler"
                if self.mode == "native"
                else "vllm_prefix_scheduler.m3_observer.ObserverPrefixScheduler"
            )

        def emit(self, kind, **fields):
            self.seq += 1
            self.tick += 1_000_000
            event = {
                "schema": 1,
                "event": kind,
                "pid": self.pid,
                "seq": self.seq,
                "monotonic_ns": self.tick,
                "clock_domain": "monotonic_ns:synthetic-contract-fixture",
                "engine_id": self.name,
                "mode": self.mode,
                "phase": self.phase,
                **copy.deepcopy(fields),
            }
            all_events.append(event)
            return event

        def effective(self):
            return {
                "async_scheduling": True,
                "enable_prefix_caching": True,
                "enable_chunked_prefill": True,
                "use_v2_model_runner": True,
                "enforce_eager": True,
                "max_num_seqs": 4 if self.phase == "smoke" else 1,
                "max_num_active_seqs": None,
                "max_num_batched_tokens": 2048,
                "max_model_len": 16384,
                "gpu_memory_utilization": 0.85,
                "executor_backend": "uni",
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "max_concurrent_batches": 2,
            }

        def resources(self):
            return {
                **{
                    name: []
                    for name in (
                        "requests",
                        "waiting",
                        "running",
                        "holding",
                        "deferred",
                        "finished_req_ids",
                        "reset_preempted_req_ids",
                        "inflight_prefills",
                        "pending_batches",
                        "observation_errors",
                    )
                },
                **{
                    name: 0
                    for name in (
                        "deferred_frees",
                        "request_blocks",
                        "non_null_ref_sum",
                        "inflight_tokens",
                        "stale_tokens",
                        "output_placeholders",
                    )
                },
                "guard_records": list(self.guards.values()),
                "guard_closed": self.closed if self.mode == "guarded" else None,
                "guard_fault_code": None,
                "free_count": 100,
                "free_unique_count": 100,
                "free_traversed_count": 100,
                "total_non_null_blocks": 100,
                "cached_block_count": self.cached,
            }

        def worker(self, label):
            return {
                "pid": self.pid,
                "label": label,
                "runner_type": "vllm.v1.worker.gpu.model_runner.GPUModelRunner",
                "runner_observer_installed": self.phase == "diagnostic",
                "configured_scheduler_type": self.actual,
                "resolved_model_path": sources["model"]["directory"],
                "resolved_tokenizer_path": sources["model"]["directory"],
                "model_parameter_dtypes": {"torch.bfloat16": 4_000_000_000},
                "gpu_name": "NVIDIA GeForce RTX 3090",
                "compute_capability": [8, 6],
                "gpu_total_memory_bytes": 24 * 1024**3,
                "effective_config": self.effective(),
                "num_gpu_blocks": 101,
                "kv_tensors": [
                    {
                        "shape": [32, 64],
                        "numel": 2048,
                        "view_bytes": 4096,
                        "device": "cuda:0",
                        "storage_id": "synthetic-storage",
                        "storage_bytes": 4096,
                    }
                ],
                "kv_unique_storage_count": 1,
                "kv_unique_storage_bytes": 4096,
                "kv_cache_groups": [
                    {
                        "layer_names": ["layer0", "layer1"],
                        "spec": {
                            "type": "vllm.v1.kv_cache_interface.FullAttentionSpec",
                            "dtype": "torch.bfloat16",
                        },
                    }
                ],
                "modules": {
                    name: {
                        "path": "/synthetic/" + name,
                        "sha256": modules[name.replace(".", "/") + ".py"],
                    }
                    for name in native_names
                },
                "observer_modules": {
                    "events": {"sha256": "0" * 64},
                    "worker": {"sha256": "0" * 64},
                },
                "production_modules": {
                    name: {"sha256": "0" * 64}
                    for name in ("scheduler", "config", "probe", "wait_guard")
                },
                "observation_errors": [],
                "runner_request_ids": [],
                "runner_execute_state_pending": False,
                "scheduler": self.resources() if self.phase == "diagnostic" else None,
            }

        def drain(self, case, label):
            full = case + ":" + label
            if self.phase == "diagnostic":
                self.emit(
                    "scheduler_checkpoint", case_id=case, label=full, resources=self.resources()
                )
            self.emit(
                "client_drain",
                case_id=case,
                label=label,
                unfinished=0,
                snapshots=[self.worker(full)],
            )

        def add(self, case, role, value, waiting=None):
            rid = case + "::" + role
            self.emit("client_submit", case_id=case, request_id=rid)
            if self.phase == "smoke":
                return
            guard = None
            if self.mode == "guarded":
                guard = {
                    "request_id": rid,
                    "object_id": rid + "@fixture",
                    "enqueued_ns": self.tick,
                    "sequence": self.next_guard,
                }
                self.next_guard += 1
                self.guards[rid] = guard
            enqueue = self.emit(
                "enqueue",
                request_id=rid,
                object_id=rid + "@fixture",
                prompt_token_ids=value["token_ids"],
                num_computed_tokens=0,
                status="WAITING",
                bound=True,
                waiting=waiting or [rid],
                guard=guard,
            )
            if role in {"B", "C"}:
                self.emit(
                    "client_enqueue_confirmed",
                    request_id=rid,
                    engine_pid=self.pid,
                    engine_seq=enqueue["seq"],
                )

        def generate(self, case, role, cancel=False):
            rid = case + "::" + role
            output_ids = []
            count = 1 if cancel else 8
            for step in range(count):
                output_ids.append(7)
                if self.phase == "diagnostic":
                    self.batch += 1
                    batch = f"{self.pid}:{self.batch}"
                    amount = 32 if step == 0 else 1
                    start = 0 if step == 0 else 31 + step
                    descriptor = {
                        "num_scheduled_tokens": {rid: amount},
                        "total_num_scheduled_tokens": amount,
                        "new_req_ids": [rid] if step == 0 else [],
                        "cached_req_ids": [] if step == 0 else [rid],
                        "finished_req_ids": [],
                    }
                    descriptor["output_sha256"] = hashlib.sha256(
                        canonical(descriptor).encode()
                    ).hexdigest()
                    before = list(self.guards.values())
                    self.guards.pop(rid, None)
                    self.emit(
                        "schedule_success",
                        batch_id=batch,
                        **descriptor,
                        intervals=[
                            {
                                "request_id": rid,
                                "object_id": rid + "@fixture",
                                "start": start,
                                "end": start + amount,
                                "prompt_length": 32,
                            }
                        ],
                        guard_before=before,
                        guard_after=list(self.guards.values()),
                    )
                    self.emit("runner_execute_start", batch_id=batch, **descriptor)
                    self.emit(
                        "runner_execute_success",
                        batch_id=batch,
                        result_kind="none",
                        forward_observed=True,
                        forward_tokens=amount,
                        forward_tokens_padded=amount,
                        forward_req_ids=[rid],
                    )
                    self.emit("runner_sample_start", batch_id=batch)
                    self.emit("runner_sample_success", batch_id=batch, result_kind="async")
                    output = {
                        "output_type": "vllm.v1.outputs.ModelRunnerOutput",
                        "req_ids": [rid],
                        "sampled_token_ids": [[7]],
                    }
                    self.emit("model_output_materialized", batch_id=batch, **output)
                    self.emit("update_from_output_start", batch_id=batch, **output)
                    if step == count - 1 and not cancel:
                        self.emit(
                            "request_terminal",
                            request_id=rid,
                            object_id=rid + "@fixture",
                            status="FINISHED_LENGTH_CAPPED",
                            guard_before=list(self.guards.values()),
                            guard_after=list(self.guards.values()),
                        )
                    self.emit(
                        "update_from_output_success",
                        batch_id=batch,
                        **output,
                        engine_outputs=[
                            {
                                "request_id": rid,
                                "new_token_ids": [7],
                                "finish_reason": "LENGTH" if step == count - 1 else None,
                            }
                        ],
                    )
                self.emit(
                    "client_output",
                    case_id=case,
                    request_id=rid,
                    token_ids=output_ids,
                    finished=step == count - 1 and not cancel,
                    finish_reason="length" if step == count - 1 and not cancel else None,
                )
            if cancel:
                self.emit("client_abort", case_id=case, request_id=rid)
                self.emit(
                    "request_terminal",
                    request_id=rid,
                    object_id=rid + "@fixture",
                    status="FINISHED_ABORTED",
                    guard_before=list(self.guards.values()),
                    guard_after=list(self.guards.values()),
                )
            last = {"token_ids": output_ids, "finish_reason": "abort" if cancel else "length"}
            self.emit("client_done", case_id=case, request_id=rid, **last)
            self.finished[rid] = last
            self.cached = 2

        def case(self, case, metadata):
            scenario = metadata["scenario"]
            if scenario == "smoke":
                values = {
                    f"S{i}": {"text": text, "token_ids": inputs["by_text"][text]}
                    for i, text in enumerate(texts[3:])
                }
            elif scenario == "cancel":
                text = config["requests"]["B"]["text"]
                values = {"X": {"text": text, "token_ids": inputs["by_text"][text]}}
            else:
                values = {
                    role: {"text": value["text"], "token_ids": inputs["by_text"][value["text"]]}
                    for role, value in config["requests"].items()
                }
            sampling = {
                **config["sampling"],
                **({"max_tokens": 512} if scenario == "cancel" else {}),
            }
            self.emit("case_begin", case_id=case, **metadata, inputs=values, sampling=sampling)
            self.emit("cache_reset_success", case_id=case)
            self.cached = 0
            self.drain(case, "initial")
            if scenario == "mechanism":
                self.add(case, "A", values["A"])
                self.generate(case, "A")
                self.drain(case, "warm_drained")
                self.emit("pause_success", case_id=case)
                b, c = case + "::B", case + "::C"
                self.add(case, "B", values["B"])
                if metadata["variant"] == "guarded_expired":
                    self.tick += 750_000_000
                self.add(case, "C", values["C"], waiting=[b, c])
                self.emit("resume_success", case_id=case)
                order = (
                    ["C", "B"]
                    if metadata["variant"] in {"prefix", "guarded_unexpired"}
                    else ["B", "C"]
                )
                if self.mode in {"prefix", "guarded"}:
                    scores = {c: 16} if metadata["variant"] == "guarded_expired" else {b: 0, c: 16}
                    self.emit(
                        "decision",
                        before=[b, c],
                        after=[case + "::" + role for role in order],
                        scores=scores,
                        records=self.guards,
                        now_ns=self.tick if self.mode == "guarded" else None,
                        threshold_ns=500_000_000 if self.mode == "guarded" else None,
                    )
                for role in order:
                    self.generate(case, role)
            else:
                for role, value in values.items():
                    self.add(case, role, value)
                    self.generate(case, role, cancel=scenario == "cancel")
            self.drain(case, "drained")
            self.emit("case_end", case_id=case, **metadata, status="passed")

        def close(self):
            # Real RPC shutdown boundaries explicitly carry case_id=null.
            self.emit("worker_snapshot", case_id=None, **self.worker("before_shutdown"))
            if self.phase == "diagnostic":
                self.emit("scheduler_shutdown_before", resources=self.resources())
                self.closed = True
                self.emit("scheduler_shutdown_after", resources=self.resources())
            self.emit("engine_completed", request_count=len(self.finished), results=self.finished)
            self.emit("engine_shutdown", client_unfinished=0, observation_errors=[])
            self.emit(
                "process_exit",
                exitcode=0,
                timed_out=False,
                owned_pids_before_cleanup=[],
                owned_pids_remaining=[],
            )

    for index, name in enumerate(
        f"{phase}-{mode}"
        for phase in ("smoke", "diagnostic")
        for mode in ("native", "off", "prefix", "guarded")
    ):
        engine = Engine(name, 100 + index)
        for case, meta in expected.items():
            target = f"{'smoke' if meta['scenario'] == 'smoke' else 'diagnostic'}-{meta['mode']}"
            if target == name:
                engine.case(case, meta)
        engine.close()
    trace_path = directory / "trace.jsonl"
    trace_path.write_text("".join(canonical(event) + "\n" for event in all_events))
    gpu = {
        "returncode": 0,
        "stdout": "NVIDIA GeForce RTX 3090, 24576, 1, 24575, 0\n",
        "stderr": "",
        "monotonic_ns": 1,
    }
    raw_path = directory / "synthetic-resource.txt"
    raw_path.write_text("SYNTHETIC contract fixture; not real resource evidence\n")
    resources = {
        "allocations": [
            {
                "job_id": "12345",
                "limit_minutes": 20,
                "gpus": 1,
                "state": "COMPLETED",
                "exit_code": "0:0",
                "gpu_before": gpu,
                "gpu_during": [gpu],
                "gpu_after": gpu,
                "owned_pids_remaining": [],
                "slurm_final": {
                    "returncode": 0,
                    "stdout": "JobId=12345 JobState=COMPLETED ExitCode=0:0 TimeLimit=00:20:00 StartTime=2026-11-02T10:00:00 EndTime=2026-11-02T10:01:00",
                },
            }
        ],
        "active_query_success": True,
        "active_owned_jobs": [],
        "active_query": {"returncode": 0, "stdout": ""},
        "raw_artifacts": [{"path": str(raw_path), "sha256": sha256(raw_path)}],
    }
    resources_path = directory / "resources.json"
    resources_path.write_text(canonical(resources))
    return {
        "config_path": config_path,
        "inputs_path": inputs_path,
        "trace_path": trace_path,
        "resources_path": resources_path,
        "events": all_events,
        "config": config,
        "inputs": inputs,
        "resources": resources,
        "sources": sources,
    }
