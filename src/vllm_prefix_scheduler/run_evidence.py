"""Collect private command evidence and check its integrity, without granting acceptance."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .config_evidence import check_config_inputs

KINDS = ("local", "native", "gpu", "performance")
COMMAND_STATES = {"succeeded", "failed", "timed_out", "interrupted", "launch_failed"}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def git_output(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, timeout=30
    ).stdout


def source_state(root: Path) -> dict:
    """Fingerprint tracked and nonignored sources; never traverse source symlinks."""
    names = git_output(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    files = {}
    for name in sorted(set(names.decode().split("\0")) - {""}):
        path = root / name
        if path.is_symlink():
            files[name] = "symlink:" + os.readlink(path)
        elif path.is_file():
            files[name] = digest(path)
        else:
            files[name] = "missing"
    return {
        "commit": git_output(root, "rev-parse", "HEAD").decode().strip(),
        "dirty": bool(git_output(root, "status", "--porcelain")),
        "files": files,
        "files_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
    }


def file_record(directory: Path, path: Path) -> dict:
    return {
        "path": path.relative_to(directory).as_posix(),
        "sha256": digest(path),
        "bytes": path.stat().st_size,
    }


def snapshot(directory: Path, source: Path, group: str, index: int) -> dict:
    if not source.is_file():
        raise ValueError(f"Evidence input is not a file: {source}")
    target = directory / group / f"{index:03d}-{source.name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as original, target.open("xb") as copied:
        while chunk := original.read(1024 * 1024):
            copied.write(chunk)
    return {**file_record(directory, target), "source": str(source)}


def stop_process(process: subprocess.Popen) -> list[str]:
    """Stop this command's process group, including children left by an exiting parent."""
    errors = []
    for force in (False, True):
        fallback = os.name != "posix"
        if not fallback:
            try:
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError as error:
                errors.append(f"Process-group cleanup could not be confirmed: {error}")
                fallback = True
        if fallback and process.poll() is None:
            try:
                process.kill() if force else process.terminate()
            except OSError as error:
                errors.append(f"Command cleanup could not be confirmed: {error}")
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            if force:
                errors.append("Command did not exit within the cleanup deadline")
    return errors


def group_exists(pid: int) -> bool:
    if os.name != "posix":
        return False
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # EPERM does not prove the group is gone; refuse a clean acceptance claim.
        return True
    return True


class RunSignals:
    """Remember the first termination request without interrupting cleanup or sealing."""

    def __init__(self):
        self.first_signal = None

    def request(self, signum, _frame=None):
        if self.first_signal is None:
            self.first_signal = signum


@contextmanager
def collect_signals():
    requested = RunSignals()
    previous = {}
    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, requested.request)
        yield requested
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def wait_for_command(process: subprocess.Popen, timeout: float, requested: RunSignals) -> tuple:
    deadline = time.monotonic() + timeout
    while True:
        if requested.first_signal is not None:
            return "interrupted", 128 + requested.first_signal
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "timed_out", 124
        try:
            returncode = process.wait(timeout=min(0.1, remaining))
        except subprocess.TimeoutExpired:
            continue
        return (
            "succeeded" if returncode == 0 else "failed",
            returncode if returncode >= 0 else 128 - returncode,
        )


def seal_run(output: Path, manifest: dict, requested: RunSignals, code: int) -> int:
    """Keep the manifest/review binding consistent if a signal arrives during sealing."""
    while True:
        observed_signal = requested.first_signal
        if observed_signal is not None:
            manifest.update(command_status="interrupted", termination_signal=observed_signal)
            code = 128 + observed_signal
        write_json(output / "manifest.json", manifest)
        acceptance = {
            "schema_version": 1,
            "manifest_sha256": digest(output / "manifest.json"),
            "decision": "pending",
            "reviewer": None,
            "criteria": [
                {**criterion, "status": "pending", "evidence": [], "note": ""}
                for criterion in manifest["criteria"]
            ],
            "limitations": [],
        }
        write_json(output / "acceptance.json", acceptance)
        report = [
            "# Command evidence (private)",
            "",
            f"Task: `{manifest['task_id']}`; declared layer: `{manifest['kind']}`.",
            f"Command status: **{manifest['command_status']}**; return code: `{manifest['returncode']}`.",
            "Acceptance: **pending**. See acceptance.json for the current review decision.",
            "",
            "A successful command is not semantic acceptance. Skipped tests, mock results and",
            "missing native/GPU artifacts do not establish their corresponding validation layer.",
            "Collector runtime is not necessarily the runtime of the child command.",
            "These files contain private commands, paths and logs; publish only through an explicit exporter.",
            "",
            "Exact argument vector:",
            "",
            "```json",
            json.dumps(manifest["argv"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Collection errors",
            "",
            *[f"- {error}" for error in manifest["collection_errors"]],
            "",
        ]
        (output / "report.md").write_text("\n".join(report))
        if requested.first_signal == observed_signal:
            return code if code else int(bool(manifest["collection_errors"]))


def collect_run(
    root: Path,
    output: Path,
    argv: list[str],
    *,
    task_id: str,
    kind: str,
    timeout: float,
    configs: list[Path],
    artifacts: list[Path],
    criteria: list[str],
    seed: int | None = None,
) -> tuple[dict, int]:
    if kind not in KINDS or not argv or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("A supported kind, command and finite positive timeout are required")
    if not task_id.strip() or any(not criterion.strip() for criterion in criteria):
        raise ValueError("Task ID and supplied criteria must be nonempty")
    if kind in {"gpu", "performance"} and not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Run GPU/performance collection inside an existing Slurm allocation")
    root, output = root.resolve(), output.resolve()
    if output.is_relative_to(root) and not output.is_relative_to(root / "runs"):
        raise ValueError("Private evidence inside the repository must stay under runs/")
    config_paths = list(
        dict.fromkeys(
            # Keep symlink paths so retargeting an input is checked as the command sees it.
            (path if path.is_absolute() else root / path).absolute()
            for path in [root / "configs/upstream.json", *configs]
        )
    )
    for path in config_paths:
        if not path.is_file():
            raise ValueError(f"Missing configuration: {path}")
    # Refuse reuse, including a directory left by a failed or interrupted collection.
    if output.exists():
        raise ValueError("Run directory already exists; use a new run ID")
    before = source_state(root)
    pins = json.loads((root / "configs/upstream.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    records = [snapshot(output, path, "configs", i) for i, path in enumerate(config_paths)]
    source_file = output / "source-before.json"
    write_json(source_file, before)
    records.append(file_record(output, source_file))
    versions = {}
    for package in ("vllm", "torch", "pytest"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest = {
        "schema_version": 2,
        "task_id": task_id,
        "criteria": [{"id": f"C{i}", "description": text} for i, text in enumerate(criteria, 1)],
        "kind": kind,
        "kind_scope": "Declared evidence layer, not automatic native/GPU/performance validation",
        "argv": argv,
        "cwd": str(root),
        "timeout_seconds": timeout,
        "seed": seed,
        "seed_scope": "Recorded metadata; the workload must explicitly use the same seed",
        "upstream": pins,
        "project": {key: value for key, value in before.items() if key != "files"},
        "collector_runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "packages": versions,
            "scope": "Collector environment; a child using another interpreter must archive its runtime",
        },
        "allocation": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "scope": "Environment metadata only; hardware and allocation release need run artifacts",
        },
        "started_at": datetime.now(UTC).isoformat(),
        "command_status": "running",
        "files": records,
        "collection_errors": [],
    }
    write_json(output / "manifest.json", manifest)
    started = time.monotonic()
    process = None
    with collect_signals() as requested:
        state, code = "launch_failed", 127
        with (
            (output / "stdout.log").open("wb") as stdout,
            (output / "stderr.log").open("wb") as stderr,
        ):
            try:
                process = subprocess.Popen(
                    argv,
                    cwd=root,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=os.name == "posix",
                )
                state, code = wait_for_command(process, timeout, requested)
            except KeyboardInterrupt:
                requested.request(signal.SIGINT)
                state, code = "interrupted", 128 + requested.first_signal
            except OSError as error:
                stderr.write(f"{type(error).__name__}: {error}\n".encode())
                state, code = "launch_failed", 127
            finally:
                if process is not None:
                    group_alive = group_exists(process.pid)
                    if group_alive and state in {"succeeded", "failed"}:
                        manifest["collection_errors"].append(
                            "Command left background processes; its process group was terminated"
                        )
                    if process.poll() is None or group_alive:
                        manifest["collection_errors"].extend(stop_process(process))
                    if group_exists(process.pid):
                        # A non-child zombie can briefly retain the group ID.
                        deadline = time.monotonic() + 0.5
                        while group_exists(process.pid) and time.monotonic() < deadline:
                            time.sleep(0.01)
        manifest.update(
            command_status=state,
            returncode=process.returncode if process is not None else None,
            timeout_expired=state == "timed_out",
            finished_at=datetime.now(UTC).isoformat(),
            duration_seconds=time.monotonic() - started,
        )
        for filename in ("stdout.log", "stderr.log"):
            records.append(file_record(output, output / filename))
        for index, path in enumerate(artifacts):
            try:
                source = path if path.is_absolute() else root / path
                records.append(snapshot(output, source, "artifacts", index))
            except (OSError, ValueError) as error:
                manifest["collection_errors"].append(str(error))
        config_checks, errors = check_config_inputs(records)
        manifest["config_checks"] = config_checks
        manifest["config_inputs_stable"] = bool(config_checks) and not errors
        manifest["collection_errors"].extend(errors)
        try:
            after = source_state(root)
            after_file = output / "source-after.json"
            write_json(after_file, after)
            records.append(file_record(output, after_file))
            manifest["source_changed_during_run"] = (
                before["files_sha256"] != after["files_sha256"]
                or before["commit"] != after["commit"]
            )
        except (OSError, subprocess.SubprocessError) as error:
            manifest["collection_errors"].append(str(error))
        code = seal_run(output, manifest, requested, code)
    return manifest, code


def contained_file(directory: Path, name: str) -> Path:
    relative = Path(name)
    path = (directory / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError(f"Missing or escaping evidence file: {name}")
    return path


def check_run(directory: Path, *, require_success=False, require_accepted=False) -> dict:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    schema = manifest.get("schema_version")
    if schema not in {1, 2} or manifest.get("kind") not in KINDS:
        raise ValueError("Unsupported run evidence schema/kind")
    state = manifest.get("command_status")
    if state not in COMMAND_STATES:
        raise ValueError("Run is unfinished or command status is invalid")
    if (state == "succeeded" and manifest.get("returncode") != 0) or (
        state == "failed" and manifest.get("returncode") in {0, None}
    ):
        raise ValueError("Command status contradicts its return code")
    records = manifest.get("files", [])
    by_path = {record["path"]: record for record in records}
    if len(by_path) != len(records) or not {"stdout.log", "stderr.log"}.issubset(by_path):
        raise ValueError("Missing logs or duplicate evidence paths")
    for name, record in by_path.items():
        path = contained_file(directory, name)
        if digest(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"Evidence hash/size mismatch: {name}")
    configs_verified = schema == 2 and manifest.get("config_inputs_stable") is True
    if schema == 2:
        configs = {name: record for name, record in by_path.items() if name.startswith("configs/")}
        checks = manifest.get("config_checks", [])
        by_config = {item["snapshot_path"]: item for item in checks}
        if not configs or set(configs) != set(by_config) or len(checks) != len(by_config):
            raise ValueError("Configuration checks must cover every registered input")
        for name, record in configs.items():
            item = by_config[name]
            if item["expected_sha256"] != record["sha256"] or item["source"] != record["source"]:
                raise ValueError("Configuration check differs from its archived input")
            if item["status"] not in {"unchanged", "changed", "missing", "unreadable"}:
                raise ValueError("Invalid configuration check status")
            if (item["status"] == "unchanged") != (item["actual_sha256"] == record["sha256"]):
                raise ValueError("Configuration check contradicts its hash")
        if configs_verified != all(item["status"] == "unchanged" for item in checks):
            raise ValueError("Configuration stability contradicts its checks")
    acceptance = json.loads((directory / "acceptance.json").read_text())
    if acceptance.get("schema_version") != 1 or acceptance.get("manifest_sha256") != digest(
        manifest_path
    ):
        raise ValueError("Acceptance is not bound to this manifest")
    decision = acceptance.get("decision")
    if decision not in {"pending", "accepted", "rejected", "blocked"}:
        raise ValueError("Invalid acceptance decision")
    original = {item["id"]: item["description"] for item in manifest["criteria"]}
    reviewed = acceptance.get("criteria", [])
    reviewed_criteria = {item["id"]: item["description"] for item in reviewed}
    if (
        original != reviewed_criteria
        or len(original) != len(manifest["criteria"])
        or len(reviewed_criteria) != len(reviewed)
    ):
        raise ValueError("Review criteria differ from the original manifest criteria")
    if any(
        item.get("status") not in {"pending", "passed", "failed", "skipped", "blocked"}
        for item in reviewed
    ):
        raise ValueError("Invalid criterion status")
    if decision == "accepted":
        if not configs_verified and (schema == 2 or require_accepted):
            raise ValueError("Acceptance requires verified stable configuration inputs")
        criteria = acceptance.get("criteria", [])
        if (
            state != "succeeded"
            or manifest.get("collection_errors")
            or manifest.get("source_changed_during_run") is not False
            or not isinstance(acceptance.get("reviewer"), str)
            or not acceptance["reviewer"].strip()
            or not criteria
        ):
            raise ValueError("Acceptance requires success, stable sources, reviewer and criteria")
        for criterion in criteria:
            if (
                not criterion.get("description", "").strip()
                or criterion.get("status") != "passed"
                or not criterion.get("evidence")
            ):
                raise ValueError("Every accepted criterion needs a pass and archived evidence")
            for name in criterion["evidence"]:
                if name not in by_path:
                    raise ValueError(f"Acceptance refers to unarchived evidence: {name}")
    if require_success and (
        state != "succeeded"
        or manifest.get("collection_errors")
        or (schema == 2 and not configs_verified)
    ):
        raise ValueError("Command success gate did not pass")
    if require_accepted and decision != "accepted":
        raise ValueError("Run has not been accepted")
    return {
        "integrity": "valid",
        "command_status": state,
        "acceptance": decision,
        "kind": manifest["kind"],
        "configuration_verification": "verified" if configs_verified else "not_verified",
        "semantic_scope": "Review assertions are checked structurally, not independently proven",
    }


def run_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--config", type=Path, action="append", default=[])
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--criterion", action="append", default=[])
    parser.add_argument("--seed", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    root = Path(__file__).resolve().parents[2]
    try:
        manifest, code = collect_run(
            root,
            args.output,
            command,
            task_id=args.task_id,
            kind=args.kind,
            timeout=args.timeout,
            configs=args.config,
            artifacts=args.artifact,
            criteria=args.criterion,
            seed=args.seed,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(2, f"Evidence collection error: {error}\n")
    print(json.dumps({"command_status": manifest["command_status"], "acceptance": "pending"}))
    raise SystemExit(code)


def check_main() -> None:
    parser = argparse.ArgumentParser(
        description="Check private evidence integrity and review gates"
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--require-command-success", action="store_true")
    parser.add_argument("--require-accepted", action="store_true")
    args = parser.parse_args()
    try:
        result = check_run(
            args.directory,
            require_success=args.require_command_success,
            require_accepted=args.require_accepted,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Evidence check failed: {error}\n")
    print(json.dumps(result, indent=2))
