"""Source-bound public export for the explicitly authorized replacement campaign.

The original exporter and historical acceptance semantics remain unchanged.
Every output retains the unresolved historical terminal-evidence exception;
passing this campaign never changes the original full-history gate to passed.
"""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_prefix_scheduler import m3_campaign_contract, m3_contract  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "m3_legacy_export_for_campaign", ROOT / "scripts/export_m3.py"
)
legacy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(legacy)

CAMPAIGN_ID = "m3-campaign-20261103-v1"
SCOPE = "replacement_campaign_with_retained_history_exception"
PATH_ARGUMENTS = (
    "campaign_config",
    "authorization",
    "history_index",
    "config",
    "inputs",
    "trace_dir",
    "resources",
    "native_index",
)
EXCEPTION_FIELDS = {
    "historical_resource_status": "exception_unverified_terminal",
    "original_full_history_gate": "blocked",
    "history_terminal_unverifiable": True,
    "original_full_history_gate_blocked": True,
}


def require(condition, message):
    if not condition:
        raise ValueError(f"M3 campaign export: {message}")


def _hash(value, label):
    require(m3_contract.is_hash(value), f"missing valid {label} hash")
    return value


def _scope_fields(audit):
    require(type(audit) is dict, "campaign audit must be an object")
    require(
        type(audit.get("schema_version")) is int and audit["schema_version"] == 1,
        "campaign audit schema differs",
    )
    require(
        audit.get("scope") == SCOPE and audit.get("campaign_id") == CAMPAIGN_ID,
        "unrecognized campaign scope or identity",
    )
    require(
        audit.get("campaign_acceptance") == audit.get("campaign_resource_acceptance") == "passed",
        "new campaign is not fully passed",
    )
    for key, value in EXCEPTION_FIELDS.items():
        require(
            type(audit.get(key)) is type(value) and audit[key] == value,
            "historical exception was removed or relabeled",
        )
    require(audit.get("task_budget_status") == "within_8", "cumulative task budget not satisfied")
    return {
        "schema_version": 1,
        "task_id": m3_contract.TASK_ID,
        "scope": SCOPE,
        "campaign_id": CAMPAIGN_ID,
        "campaign_pass": True,
        "campaign_acceptance": "passed",
        "campaign_resource_acceptance": "passed",
        **EXCEPTION_FIELDS,
        "task_budget_status": "within_8",
    }


def public_campaign_summary(audit, sources, config):
    """Project only finite campaign labels, safe metrics and bound history hashes."""
    scoped = _scope_fields(audit)
    commit = audit.get("project_commit")
    require(
        m3_contract.is_hash(commit, 40) and commit == sources.get("project_commit"),
        "campaign source commit differs",
    )
    history = audit.get("history")
    require(type(history) is dict, "retained history audit missing")
    require(
        type(history.get("allocation_count")) is int and history["allocation_count"] == 4,
        "all four historical allocations must be retained",
    )
    require(
        type(history.get("unverified_terminal_count")) is int
        and history["unverified_terminal_count"] == 1,
        "historical terminal exception count differs",
    )
    safe_history = {
        "allocation_count": 4,
        "unverified_terminal_count": 1,
        "index_sha256": _hash(history.get("index_sha256"), "history index"),
        "ledger_sha256": _hash(history.get("ledger_sha256"), "history ledger"),
    }
    budget = audit.get("budget")
    require(type(budget) is dict, "cumulative budget audit missing")
    values = {
        name: legacy.integer(budget.get(name), name, 1)
        for name in (
            "historical_allocations",
            "new_allocations",
            "total_allocations",
            "cumulative_limit",
        )
    }
    require(
        values["historical_allocations"] == 4 and values["cumulative_limit"] == 8,
        "historical/cumulative allocation budget differs",
    )
    require(
        1 <= values["new_allocations"] <= 4
        and values["total_allocations"] == 4 + values["new_allocations"] <= 8,
        "new or cumulative allocation limit exceeded",
    )
    native = audit.get("native")
    require(
        type(native) is dict
        and type(native.get("compat_cases")) is int
        and native["compat_cases"] == 432
        and type(native.get("m1_cases")) is int
        and native["m1_cases"] == 12,
        "required native regressions missing",
    )
    gpu_audit = audit.get("gpu_audit")
    require(
        type(gpu_audit) is dict and gpu_audit.get("project_commit") == commit,
        "GPU campaign commit differs",
    )
    functional = legacy.public_summary(gpu_audit, sources, config)
    require(
        functional["allocation_count"] == values["new_allocations"],
        "GPU resource audit and campaign allocation catalog differ",
    )
    # These old fields describe the old export's acceptance scope. Rebuild a
    # campaign-scoped envelope rather than forwarding any generic pass label.
    excluded = {
        "schema_version",
        "task_id",
        "scope",
        "accepted",
        "project_commit",
        "allocation_count",
        "owned_resources_released",
    }
    functional = {key: value for key, value in functional.items() if key not in excluded}
    return {
        **scoped,
        "project_commit": commit,
        "history": safe_history,
        "budget": values,
        "native": {"compat_cases": 432, "m1_cases": 12},
        "campaign_owned_resources_released": True,
        "functional": functional,
    }


def _git_file(root, commit, name):
    return subprocess.check_output(["git", "-C", str(root), "cat-file", "blob", f"{commit}:{name}"])


def verified_artifact_hashes(audit):
    """Recheck indirect private originals; publish content identity without paths."""
    provenance = audit.get("provenance")
    require(type(provenance) is dict, "campaign artifact provenance missing")
    inventory = provenance.get("artifact_inventory")
    require(type(inventory) is list and inventory, "campaign indirect artifact inventory missing")
    seen, public = set(), []
    for artifact in inventory:
        require(
            type(artifact) is dict and set(artifact) == {"path", "sha256", "bytes"},
            "indirect artifact entry differs",
        )
        path = Path(artifact["path"])
        require(
            path.is_absolute() and path.is_file() and str(path) not in seen,
            "indirect artifact missing or duplicated",
        )
        seen.add(str(path))
        digest = _hash(artifact["sha256"], "indirect artifact")
        size = legacy.integer(artifact["bytes"], "indirect artifact bytes")
        require(
            path.stat().st_size == size and legacy.digest(path) == digest,
            "indirect history/native/resource artifact changed after audit",
        )
        public.append({"sha256": digest, "bytes": size})
    return sorted(public, key=lambda value: (value["sha256"], value["bytes"]))


def verify_frozen_campaign_sources(root, sources, *, campaign_config, authorization, config):
    """Retain all old source checks and bind new campaign code/config/authorization."""
    hashes = legacy.frozen_sources(root, sources)
    require(
        all(legacy.digest(root / name) == value for name, value in hashes.items()),
        "current campaign code/config/tests differ from frozen source",
    )
    for name in (
        "scripts/export_m3.py",
        "scripts/export_m3_campaign.py",
        "src/vllm_prefix_scheduler/m3_contract.py",
        "src/vllm_prefix_scheduler/m3_campaign_contract.py",
    ):
        require(
            hashes.get(name) == legacy.digest(root / name),
            "executing old/new exporter or auditor differs from tested source",
        )
    cfg = json.loads(Path(campaign_config).read_text())
    auth = json.loads(Path(authorization).read_text())
    require(
        cfg.get("campaign_id") == CAMPAIGN_ID and cfg.get("scope") == SCOPE,
        "campaign config scope differs",
    )
    bindings = {
        "configs/m3-campaign-v1.json": Path(campaign_config),
        cfg["runtime_config"]["path"]: Path(config),
        cfg["authorization"]["path"]: Path(authorization),
    }
    for name, path in bindings.items():
        require(
            name in hashes and hashes[name] == legacy.digest(path),
            "campaign configuration or authorization differs from frozen Git",
        )
    require(
        cfg["runtime_config"]["sha256"] == legacy.digest(config)
        and cfg["authorization"]["sha256"] == legacy.digest(authorization),
        "campaign configuration reference hash differs",
    )
    previous = auth.get("previous_authorization")
    require(
        type(previous) is dict
        and previous.get("path") in hashes
        and hashes[previous["path"]] == previous.get("sha256"),
        "previous authorization is not retained unchanged in frozen Git",
    )
    return hashes


def export(args):
    m3_contract.require_assertions()
    require(not args.output.exists(), "existing campaign public evidence must not be overwritten")
    bound = {
        name: Path(getattr(args, name))
        for name in (*PATH_ARGUMENTS, "sources", "audit")
        if name != "trace_dir"
    }
    bound_hashes = {name: legacy.digest(path) for name, path in bound.items()}
    saved = json.loads(args.audit.read_text())
    _scope_fields(saved)
    sources = json.loads(args.sources.read_text())
    config = json.loads(args.config.read_text())
    source_hashes = verify_frozen_campaign_sources(
        ROOT,
        sources,
        campaign_config=args.campaign_config,
        authorization=args.authorization,
        config=args.config,
    )
    raw_files = legacy.raw_trace_files(args.trace_dir)
    require(raw_files, "new campaign raw traces missing")
    raw_hashes = {path: legacy.digest(path) for path in raw_files}
    events = m3_contract.read_events(args.trace_dir)
    require(
        {Path(event["_source"]) for event in events} == set(raw_files),
        "raw campaign trace catalog changed",
    )
    ready = [event.get("sources") for event in events if event.get("event") == "engine_ready"]
    require(
        ready and all(value == sources for value in ready),
        "source report differs from actual campaign engines",
    )
    rebuilt = m3_campaign_contract.audit_campaign(
        root=ROOT,
        **{name: Path(getattr(args, name)) for name in PATH_ARGUMENTS},
    )
    require(
        legacy.normalized_audit(rebuilt) == legacy.normalized_audit(saved),
        "saved campaign audit differs from fresh raw/source audit",
    )
    summary = public_campaign_summary(rebuilt, sources, config)
    private_artifact_hashes = verified_artifact_hashes(rebuilt)
    require(
        all(legacy.digest(path) == value for path, value in raw_hashes.items()),
        "raw campaign trace changed during export",
    )
    require(
        all(legacy.digest(path) == bound_hashes[name] for name, path in bound.items()),
        "bound campaign/history/native input changed during export",
    )
    native_config = json.loads(
        _git_file(ROOT, sources["project_commit"], "configs/native-m2b.json")
    )
    provenance = {
        **_scope_fields(rebuilt),
        "project_commit": sources["project_commit"],
        "source_sha256": source_hashes,
        **legacy.public_source_manifest(sources, config, native_config["source_sha256"]),
        "history": summary["history"],
        "budget": summary["budget"],
        "private_bound_artifacts_sha256": bound_hashes,
        "private_indirect_artifacts": private_artifact_hashes,
        "private_trace_files": sorted(
            ({"sha256": value, "bytes": path.stat().st_size} for path, value in raw_hashes.items()),
            key=lambda value: (value["sha256"], value["bytes"]),
        ),
        "export_boundary": "New authorized campaign only; historical terminal exception retained. Public content excludes raw text, internal paths, account, machine, process and job identities.",
    }
    args.output.mkdir(parents=True)
    for name, data in (("summary.json", summary), ("provenance.json", provenance)):
        (args.output / name).write_text(
            json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*PATH_ARGUMENTS, "sources", "audit", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    result = export(parser.parse_args())
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "scope",
                    "campaign_id",
                    "campaign_pass",
                    "history_terminal_unverifiable",
                    "original_full_history_gate_blocked",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
