"""Export complete M3 evidence through a dedicated, source-bound allowlist.

This command re-audits raw events, checks the saved complete audit, binds the
actual sources to the frozen Git commit, and only then creates public files.
Private paths, PIDs, clock domains, job IDs, prompts and generated text are never
copied. This exporter cannot publish partial, failed or calibration acceptance.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_prefix_scheduler import m3_contract  # noqa: E402

AUDIT_FIELDS = {
    "schema_version",
    "task_id",
    "scope",
    "accepted",
    "project_commit",
    "case_count",
    "cases",
    "engines",
    "batch_audit",
    "guard_lifecycle",
    "comparisons",
    "resource_release",
    "limitations",
}
ENGINE_FIELDS = {
    "kv_blocks",
    "kv_storage_bytes",
    "model_parameter_count",
    "project_commit",
    "smoke_direct_scheduler_object",
}
PACKAGES = {"vllm", "torch", "transformers", "tokenizers", "safetensors"}
SOURCE_ROOTS = ("src", "scripts", "configs", "tests", "Makefile")
VERSION = re.compile(r"[A-Za-z0-9.+_-]+\Z")


def require(condition, message):
    if not condition:
        raise ValueError(f"M3 export: {message}")


def digest(path):
    return m3_contract.sha256(path)


def raw_trace_files(path):
    """Match the auditor's canonical-file selection, including direct files."""
    path = Path(path)
    if path.is_file():
        return [path]
    if (path / "trace.jsonl").is_file():
        return [path / "trace.jsonl"]
    return sorted(path.rglob("*.jsonl"))


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"invalid public number: {name}")
    return value


def frozen_sources(root, sources):
    """Check all manifested code/config/test inputs against the tested Git tree."""
    commit = sources.get("project_commit")
    require(m3_contract.is_hash(commit, 40), "invalid tested commit")
    require(sources.get("project_dirty") is False, "tested tree was not clean")
    tree = subprocess.check_output(
        ["git", "-C", str(root), "ls-tree", "-r", "--name-only", commit, "--", *SOURCE_ROOTS],
        text=True,
    ).splitlines()
    hashes = sources.get("source_sha256")
    require(
        type(hashes) is dict and hashes and set(hashes) == set(tree),
        "source manifest differs from frozen Git tree",
    )
    for name in tree:
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts, "non-repository source path")
        require(m3_contract.is_hash(hashes[name]), "invalid source fingerprint")
        blob = subprocess.check_output(
            ["git", "-C", str(root), "cat-file", "blob", f"{commit}:{name}"]
        )
        require(
            hashlib.sha256(blob).hexdigest() == hashes[name],
            "source fingerprint differs from tested Git blob",
        )
    return {name: hashes[name] for name in sorted(tree)}


def normalized_audit(value):
    """Allow moving private evidence without ignoring its PID/line/seq links."""
    if type(value) is dict:
        if set(value) == {"file", "line", "pid", "seq"}:
            return {key: value[key] for key in ("line", "pid", "seq")}
        return {key: normalized_audit(item) for key, item in value.items()}
    if type(value) is list:
        return [normalized_audit(item) for item in value]
    return value


def public_source_manifest(sources, config, native_pins):
    native = sources["upstream_sha256"]
    require(type(native) is dict and native, "native source manifest missing")
    require(
        native == {**native_pins, **config["extra_upstream_sha256"]},
        "native source catalog differs from frozen public allowlist",
    )
    for name, value in native.items():
        require(
            re.fullmatch(r"vllm/[A-Za-z0-9_/]+\.py", name) is not None and ".." not in name,
            "unreviewed native source name",
        )
        require(m3_contract.is_hash(value), "invalid native content digest")
    require(
        all(native.get(name) == value for name, value in config["extra_upstream_sha256"].items()),
        "native engine/runner sources differ",
    )
    model = sources["model"]
    require(
        model["revision"] == model["tokenizer_revision"] == m3_contract.MODEL_REVISION,
        "model source revision differs",
    )
    files = model["files"]
    require(
        type(files) is list
        and len(files) == len(config["model_files"])
        and {value["name"] for value in files} == set(config["model_files"]),
        "model file catalog differs",
    )
    published = []
    for value in sorted(files, key=lambda item: item["name"]):
        name = value["name"]
        require(
            re.fullmatch(r"[A-Za-z0-9_.-]+", name) is not None and "/" not in name,
            "model filename outside whitelist",
        )
        require(m3_contract.is_hash(value["sha256"]), "model content hash invalid")
        official = config["model_files"][name]
        require(
            value["size"] == official["size"]
            and value["verified_content_id"] == official["content_id"],
            "model source identity differs",
        )
        if official["kind"] == "lfs_sha256":
            require(value["sha256"] == official["content_id"], "weight content hash differs")
        published.append(
            {
                "name": name,
                "bytes": integer(value["size"], "model file bytes", 1),
                "sha256": value["sha256"],
            }
        )
    return {
        "upstream_sha256": {name: native[name] for name in sorted(native)},
        "model_file_hashes": published,
    }


def public_summary(audit, sources, config):
    """Rebuild public values from explicit types/finite names, never copy raw dicts."""
    require(type(audit) is dict and set(audit) == AUDIT_FIELDS, "unknown or missing audit fields")
    require(
        audit["schema_version"] == 1 and type(audit["schema_version"]) is int,
        "audit schema differs",
    )
    require(audit["task_id"] == m3_contract.TASK_ID, "wrong task")
    require(
        audit["scope"] == "complete" and audit["accepted"] is True,
        "complete accepted raw audit required",
    )
    expected = m3_contract.expected_cases(config)
    require(
        set(audit["cases"]) == set(expected) and audit["case_count"] == len(expected),
        "complete matrix missing",
    )
    require(
        audit["project_commit"] == sources.get("project_commit")
        and m3_contract.is_hash(audit["project_commit"], 40),
        "source/audit commit differs",
    )
    expected_engines = {
        f"{phase}-{mode}" for phase in ("smoke", "diagnostic") for mode in m3_contract.MODES
    }
    require(set(audit["engines"]) == expected_engines, "complete engine catalog missing")
    engines = {}
    for name in sorted(expected_engines):
        engine = audit["engines"][name]
        require(
            type(engine) is dict and set(engine) == ENGINE_FIELDS, "unknown engine summary fields"
        )
        require(engine["project_commit"] == audit["project_commit"], "engine commit differs")
        direct = name.startswith("diagnostic-")
        require(
            engine["smoke_direct_scheduler_object"] is direct, "smoke/diagnostic boundary differs"
        )
        engines[name] = {
            "kv_blocks": integer(engine["kv_blocks"], "KV blocks", 1),
            "kv_storage_bytes": integer(engine["kv_storage_bytes"], "KV storage", 1),
            "model_parameter_count": integer(engine["model_parameter_count"], "parameters", 1),
            "direct_scheduler_object_observed": direct,
        }
    cases = {}
    for name, meta in expected.items():
        case = audit["cases"][name]
        roles = (
            {"S0", "S1", "S2"}
            if meta["scenario"] == "smoke"
            else {"X"}
            if meta["scenario"] == "cancel"
            else {"A", "B", "C"}
        )
        require(
            set(case["inputs"]) == roles and set(case["results"]) == roles,
            "public request roles differ",
        )
        counts = {}
        for role in sorted(roles):
            prompt = case["inputs"][role]["token_ids"]
            output = case["results"][role]["token_ids"]
            require(
                type(prompt) is list and prompt and all(type(v) is int and v >= 0 for v in prompt),
                "prompt token IDs invalid",
            )
            require(
                type(output) is list and output and all(type(v) is int and v >= 0 for v in output),
                "output token IDs invalid",
            )
            counts[role] = {
                "prompt_tokens": len(prompt),
                "output_tokens": len(output),
                "prompt_token_ids_sha256": hashlib.sha256(
                    m3_contract.canonical(prompt).encode()
                ).hexdigest(),
                "output_token_ids_sha256": hashlib.sha256(
                    m3_contract.canonical(output).encode()
                ).hexdigest(),
            }
        public_case = {**meta, "requests": counts}
        if meta["scenario"] == "mechanism":
            order = case["mechanism"]["first_order"]
            require(
                type(order) is list
                and len(order) == 2
                and set(order) == {name + "::B", name + "::C"},
                "mechanism order contains unknown request identity",
            )
            public_case["first_order"] = [value.rsplit("::", 1)[1] for value in order]
        cases[name] = public_case
    batches = audit["batch_audit"]
    require(
        set(batches)
        == {
            "batches",
            "positive_batches",
            "scheduled_tokens",
            "physical_gpu_work",
            "runner_forward_tokens",
            "runner_forward_tokens_padded",
            "dispatch_by_request",
        },
        "unknown batch audit fields",
    )
    require(
        batches["physical_gpu_work"]
        == "runner input token counts observed; FLOPs and speedup unmeasured",
        "unreviewed GPU workload claim",
    )
    forward = integer(batches["runner_forward_tokens"], "runner input tokens", 1)
    padded = integer(batches["runner_forward_tokens_padded"], "runner padded input tokens", 1)
    require(padded >= forward, "runner padded count is smaller than valid input count")
    dispatch = batches["dispatch_by_request"]
    expected_requests = {
        f"{name}::{role}"
        for name, metadata in expected.items()
        if metadata["scenario"] != "smoke"
        for role in cases[name]["requests"]
    }
    require(
        type(dispatch) is dict and set(dispatch) == expected_requests,
        "diagnostic dispatch request catalog differs",
    )
    for request_id, record in dispatch.items():
        case_name, role = request_id.rsplit("::", 1)
        require(
            type(record) is dict and set(record) == {"prompt", "decode", "repeated", "intervals"},
            "dispatch accounting fields differ",
        )
        cases[case_name]["requests"][role]["scheduler_dispatch"] = {
            key: integer(record[key], f"scheduler {key}")
            for key in ("prompt", "decode", "repeated")
        }
    guard = audit["guard_lifecycle"]
    require(set(guard) == {"diagnostic-guarded"}, "guard engine catalog differs")
    guard = guard["diagnostic-guarded"]
    require(set(guard) == {"registered", "first_dispatch_cleared"}, "unknown guard summary field")
    comparisons = audit["comparisons"]
    require(type(comparisons) is list and len(comparisons) == 3, "comparison repeats missing")
    for repeat, comparison in enumerate(comparisons):
        require(
            comparison
            == {
                "repeat": repeat,
                "native_off_exact_tokens": True,
                "unexpired_prefix_decision": True,
            },
            "comparison not accepted",
        )
    release = audit["resource_release"]
    require(
        type(release) is dict and set(release) == {"allocations", "released_job_ids"},
        "resource release audit missing",
    )
    allocation_count = integer(release["allocations"], "allocation count", 1)
    require(
        allocation_count <= 4 and len(release["released_job_ids"]) == allocation_count,
        "allocation budget/release differs",
    )
    packages = sources.get("packages")
    require(
        type(packages) is dict
        and set(packages) == PACKAGES
        and all(type(v) is str and VERSION.fullmatch(v) for v in packages.values()),
        "unreviewed package metadata",
    )
    require(sources.get("python_version") == "3.12.16", "Python pin differs")
    return {
        "schema_version": 1,
        "task_id": m3_contract.TASK_ID,
        "scope": "complete controlled real-model functionality",
        "accepted": True,
        "project_commit": audit["project_commit"],
        "upstream_commit": m3_contract.UPSTREAM_COMMIT,
        "model_id": "Qwen/Qwen3-4B",
        "model_revision": m3_contract.MODEL_REVISION,
        "tokenizer_revision": m3_contract.MODEL_REVISION,
        "python_version": "3.12.16",
        "packages": {name: packages[name] for name in sorted(PACKAGES)},
        "case_count": len(expected),
        "cases": cases,
        "engines": engines,
        "batch_counts": {
            key: integer(batches[key], key, 1)
            for key in ("batches", "positive_batches", "scheduled_tokens")
        },
        "runner_forward_input_tokens": forward,
        "runner_forward_input_tokens_padded": padded,
        "guard_lifecycle": {
            key: integer(guard[key], key, 1) for key in ("registered", "first_dispatch_cleared")
        },
        "native_off_exact_tokens_repeats": 3,
        "unexpired_prefix_decision_repeats": 3,
        "allocation_count": allocation_count,
        "owned_resources_released": True,
        "limits": {
            "natural_queue_fairness": "unmeasured",
            "ttft_or_throughput_improvement": "unmeasured",
            "flops_or_speedup": "unmeasured",
            "pause_and_wait_are_controlled_interventions": True,
            "raw_internal_evidence_published": False,
        },
    }


def export(args):
    m3_contract.require_assertions()
    require(not args.output.exists(), "existing public evidence must not be overwritten")
    bound_files = {
        name: getattr(args, name) for name in ("config", "sources", "inputs", "audit", "resources")
    }
    bound_hashes = {name: digest(path) for name, path in bound_files.items()}
    config = json.loads(args.config.read_text())
    sources = json.loads(args.sources.read_text())
    saved = json.loads(args.audit.read_text())
    require(
        saved.get("scope") == "complete" and saved.get("accepted") is True,
        "partial or failed audit cannot be exported",
    )
    source_hashes = frozen_sources(ROOT, sources)
    for name in ("scripts/export_m3.py", "src/vllm_prefix_scheduler/m3_contract.py"):
        require(
            source_hashes.get(name) == digest(ROOT / name),
            "executing exporter/auditor differs from tested source",
        )
    require(
        source_hashes.get("configs/m3-real-model.json") == digest(args.config),
        "config differs from frozen Git tree",
    )
    raw_files = raw_trace_files(args.trace_dir)
    require(raw_files, "raw trace catalog empty")
    raw_hashes = {path: digest(path) for path in raw_files}
    events = m3_contract.read_events(args.trace_dir)
    require(
        {Path(event["_source"]) for event in events} == set(raw_files),
        "audited raw trace catalog changed",
    )
    ready_sources = [
        event.get("sources") for event in events if event.get("event") == "engine_ready"
    ]
    require(
        ready_sources and all(value == sources for value in ready_sources),
        "provided source report differs from raw engine evidence",
    )
    resources = json.loads(args.resources.read_text())
    frozen_inputs = json.loads(args.inputs.read_text())
    rebuilt = m3_contract.audit_trace(
        events,
        config,
        resources=resources,
        config_sha256=digest(args.config),
        frozen_inputs=frozen_inputs,
        inputs_sha256=digest(args.inputs),
    )
    require(
        normalized_audit(rebuilt) == normalized_audit(saved),
        "saved audit differs from fresh raw-event audit",
    )
    summary = public_summary(rebuilt, sources, config)
    require(
        all(digest(path) == value for path, value in raw_hashes.items()),
        "raw trace changed during audit",
    )
    require(
        all(digest(path) == bound_hashes[name] for name, path in bound_files.items()),
        "bound input changed during audit",
    )
    native_config = json.loads(
        subprocess.check_output(
            [
                "git",
                "-C",
                str(ROOT),
                "cat-file",
                "blob",
                f"{sources['project_commit']}:configs/native-m2b.json",
            ]
        )
    )
    provenance = {
        "schema_version": 1,
        "project_commit": sources["project_commit"],
        "source_sha256": source_hashes,
        **public_source_manifest(sources, config, native_config["source_sha256"]),
        "config_sha256": digest(args.config),
        "frozen_inputs_sha256": digest(args.inputs),
        "private_sources_sha256": digest(args.sources),
        "private_complete_audit_sha256": digest(args.audit),
        "private_resources_sha256": digest(args.resources),
        "private_trace_files": sorted(
            ({"sha256": digest(path), "bytes": path.stat().st_size} for path in raw_files),
            key=lambda value: (value["sha256"], value["bytes"]),
        ),
        "export_boundary": "Only finite case labels, verified repository paths, numeric aggregates and content hashes; no raw request, machine, process or job identifiers.",
    }
    args.output.mkdir(parents=True)
    for name, value in (("summary.json", summary), ("provenance.json", provenance)):
        (args.output / name).write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "trace-dir", "resources", "sources", "inputs", "audit", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    summary = export(parser.parse_args())
    print(
        json.dumps(
            {
                "exported": True,
                "case_count": summary["case_count"],
                "project_commit": summary["project_commit"],
            }
        )
    )


if __name__ == "__main__":
    main()
