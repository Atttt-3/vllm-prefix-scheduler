"""Bind private M5 run bytes to their collector, allocation and Git execution."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_prefix_scheduler.m5_campaign import TASK, digest, is_hash, read_json, require  # noqa: E402
from vllm_prefix_scheduler.run_evidence import check_run  # noqa: E402


def git_bytes(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def git_identity(commit):
    require(
        is_hash(commit, 40) and git_bytes("cat-file", "-t", commit) == b"commit\n", "invalid commit"
    )
    names = git_bytes("ls-tree", "-r", "--name-only", "-z", commit).decode().split("\0")
    return {
        name: hashlib.sha256(git_bytes("cat-file", "blob", f"{commit}:{name}")).hexdigest()
        for name in names
        if name
    }


def verify_analysis(execution, analysis):
    require(git_bytes("rev-parse", "HEAD").decode().strip() == analysis, "analysis HEAD differs")
    require(not git_bytes("status", "--porcelain"), "analysis checkout dirty")
    require(execution == analysis, "M5 analysis requires the frozen execution auditor")


def validate_profile(profile, run, config, source, git_files):
    baseline = read_json(ROOT / config["baseline_config"])
    expected = {
        key: baseline[key]
        for key in (
            "async_scheduling",
            "enable_prefix_caching",
            "enable_chunked_prefill",
            "enforce_eager",
            "tensor_parallel_size",
            "max_num_seqs",
            "max_num_batched_tokens",
            "max_model_len",
            "gpu_memory_utilization",
        )
    }
    expected.update(
        block_size=config["block_size"],
        attention_backend=[config["attention_backend"]],
        gpu_name="NVIDIA GeForce RTX 3090",
        compute_capability=[8, 6],
        model_type="vllm.model_executor.models.qwen3.Qwen3ForCausalLM",
        model_dtype="torch.bfloat16",
        use_v2_model_runner=True,
        pipeline_parallel_size=1,
        executor_backend="uni",
        scheduler_policy="fcfs",
        runner_type="vllm.v1.worker.gpu.model_runner.GPUModelRunner",
    )
    if run["mechanism"]:
        diagnostic = config["scenarios"]["mechanism"]
        expected.update(
            max_model_len=diagnostic["max_model_len"],
            num_gpu_blocks=diagnostic[f"{run['mechanism']}_blocks"],
        )
    native = run["mode"] == "native"
    if run["tier"] == "endpoint":
        expected["scheduler_type"] = (
            "vllm.v1.core.sched.async_scheduler.AsyncScheduler"
            if native
            else "vllm_prefix_scheduler.scheduler.PrefixAsyncScheduler"
        )
    else:
        expected["scheduler_type"] = (
            "vllm_prefix_scheduler.m5_observer.M5NativeScheduler"
            if native
            else "vllm_prefix_scheduler.m5_observer.M5PrefixScheduler"
        )
    require(
        all(
            type(profile[key]) is type(value) and profile[key] == value
            for key, value in expected.items()
        )
        and profile["model_device"].startswith("cuda")
        and type(profile["num_gpu_blocks"]) is int
        and profile["num_gpu_blocks"] > 1,
        "active runtime profile differs from the frozen engine",
    )
    modules = {
        "vllm.v1.core.sched.scheduler",
        "vllm.v1.core.sched.async_scheduler",
        "vllm.v1.worker.gpu.model_runner",
        "vllm.v1.worker.gpu.input_batch",
        "vllm.v1.worker.gpu.states",
        "vllm.v1.worker.gpu.async_utils",
        "vllm.v1.worker.gpu.model_states.default",
        "vllm.model_executor.models.qwen3",
        "vllm.v1.attention.backends.flash_attn",
    }
    require(
        set(profile["sources"]) == modules
        and all(
            row["sha256"] == source["upstream_sha256"][name.replace(".", "/") + ".py"]
            for name, row in profile["sources"].items()
        ),
        "active runtime module hashes differ from pinned files",
    )
    policy = profile["production_modules"]
    policy_names = {"scheduler", "config", "probe", "wait_guard"}
    require(
        set(policy) <= policy_names
        and (native and run["tier"] == "endpoint" or set(policy) == policy_names)
        and all(
            row["sha256"] == git_files[f"src/vllm_prefix_scheduler/{name}.py"]
            for name, row in policy.items()
        ),
        "active production scheduler hashes differ from the execution Git tree",
    )


def sealed_leaf(directory, remote_root, run, table, config_path, config, job, git_files, commit):
    leaf = directory / "node" / run["id"]
    raw = leaf / "raw"
    collector = leaf / "collector"
    status = check_run(collector, require_success=True)
    manifest = read_json(collector / "manifest.json")
    expected_layer = (
        "performance" if read_json(directory / "planned.json")["phase"] == "formal" else "gpu"
    )
    require(
        manifest["schema_version"] == 2
        and manifest["task_id"] == TASK
        and status["kind"] == expected_layer
        and manifest["seed"] == run["seed"]
        and manifest["allocation"]["slurm_job_id"] == job
        and manifest["project"]["commit"] == commit
        and manifest["project"]["dirty"] is False
        and manifest["source_changed_during_run"] is False,
        "collector identity/source mismatch",
    )
    before = read_json(collector / "source-before.json")
    after = read_json(collector / "source-after.json")
    catalog_hash = hashlib.sha256(json.dumps(git_files, sort_keys=True).encode()).hexdigest()
    require(
        before == after
        and before["commit"] == commit
        and before["dirty"] is False
        and before["files"] == git_files
        and before["files_sha256"] == catalog_hash
        and manifest["project"]["files_sha256"] == catalog_hash,
        "collector source fingerprint differs from actual execution Git tree",
    )

    def bind(path, source, group):
        wanted = digest(path)
        matches = [
            row
            for row in manifest["files"]
            if row["path"].startswith(group + "/")
            and row["source"] == source
            and row["sha256"] == wanted
        ]
        require(len(matches) == 1, f"raw {path.name} not sealed by collector")
        return wanted

    require(
        run["collector"] == f"{remote_root}/{run['id']}/collector"
        and run["run"] == f"{remote_root}/{run['id']}/raw"
        and run["audit"] == f"{remote_root}/{run['id']}/audit.json",
        "indexed raw location not bound to leaf",
    )
    require(
        run["inputs"].endswith(("/input.json", f"/tables/{run['pair_id']}.json")),
        "unexpected replay input provenance",
    )
    table_hash = bind(table, run["inputs"], "configs")
    config_matches = [
        entry
        for entry in manifest["files"]
        if entry["path"].startswith("configs/")
        and entry["source"].endswith("/configs/m5-performance.json")
        and entry["sha256"] == digest(config_path)
    ]
    require(len(config_matches) == 1, "collector M5 config differs")
    raw_hashes = {
        name: bind(raw / name, f"{run['run']}/{name}", "artifacts")
        for name in ("client.json", "engine.json", "sources.json")
    }
    client, engine = (read_json(raw / name) for name in ("client.json", "engine.json"))
    node = read_json(directory / "node/allocation.json")["node"].split(".")[0]
    require(
        client["execution_commit"] == commit
        and client["job_id"] == job
        and client["clock"]["host"].split(".")[0] == engine["clock"]["host"].split(".")[0] == node,
        "client and runner are not bound to the allocated node",
    )
    source = read_json(raw / "sources.json")
    require(
        source == read_json(directory / "node/allocation-sources.json")
        and source["job_id"] == job
        and source["project_commit"] == commit
        and source["python_version"] == "3.12.16"
        and source["project_dirty"] is False
        and source["packages"]["vllm"] == "0.32.0"
        and source["packages"]["torch"] == "2.13.0+cu129"
        and source["upstream_commit"] == config["upstream_commit"],
        "runtime source or job mismatch",
    )
    upstream = read_json(ROOT / "configs/native-m2b.json")["source_sha256"]
    require(
        source["upstream_sha256"] == {**upstream, **config["extra_upstream_sha256"]},
        "runtime upstream file hashes differ",
    )
    model = source["model"]
    require(
        model["revision"] == config["model_revision"]
        and model["tokenizer_revision"] == config["tokenizer_revision"],
        "runtime model/tokenizer revision mismatch",
    )
    require(
        {row["name"]: (row["size"], row["verified_content_id"]) for row in model["files"]}
        == {name: (row["size"], row["content_id"]) for name, row in config["model_files"].items()}
        and len(model["files"]) == len(config["model_files"]),
        "runtime model content differs",
    )
    source_files = {
        name: value
        for name, value in git_files.items()
        if name == "Makefile" or name.split("/")[0] in ("src", "scripts", "tests", "configs")
    }
    require(source["source_sha256"] == source_files, "bound runtime project sources differ")
    validate_profile(engine["profile"], run, config, source, git_files)
    audit = leaf / "audit.json"
    require(digest(audit) == run["audit_sha256"], "on-node audit record changed")
    return {
        "pair_id": run["pair_id"],
        "mode": run["mode"],
        "inputs": str(table),
        "run": str(raw),
        "audit": str(audit),
        "raw_hashes": raw_hashes,
        "input_sha256": table_hash,
        "collector_sha256": digest(collector / "manifest.json"),
    }
