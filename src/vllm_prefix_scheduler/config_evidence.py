"""Recheck registered configuration inputs against their immutable run snapshots."""

import hashlib
from pathlib import Path


def check_config_inputs(records: list[dict]) -> tuple[list[dict], list[str]]:
    """Read original config sources again without changing the archived snapshots.

    Accept the manifest's full file list; only ``configs/`` snapshot records are
    inputs. Paths come from the collector, which records absolute source paths.
    This observes the input at collection boundaries, not every read during a run.
    """
    checks = []
    errors = []
    for record in records:
        snapshot_path = record.get("path", "")
        if not isinstance(snapshot_path, str) or not snapshot_path.startswith("configs/"):
            continue
        source = record.get("source")
        result = {
            "snapshot_path": snapshot_path,
            "source": source,
            "expected_sha256": record.get("sha256"),
            "actual_sha256": None,
            "status": "unreadable",
            "error": None,
        }
        try:
            if not isinstance(source, str) or not Path(source).is_absolute():
                raise ValueError("Configuration source must be an absolute path")
            if not isinstance(result["expected_sha256"], str):
                raise ValueError("Configuration snapshot is missing its SHA-256")
            with Path(source).open("rb") as stream:
                result["actual_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            result["status"] = (
                "unchanged" if result["actual_sha256"] == result["expected_sha256"] else "changed"
            )
        except FileNotFoundError as error:
            result.update(status="missing", error=f"{type(error).__name__}: {error}")
        except (OSError, ValueError) as error:
            result["error"] = f"{type(error).__name__}: {error}"
        checks.append(result)
        if result["status"] != "unchanged":
            detail = f"; {result['error']}" if result["error"] else ""
            errors.append(
                f"Configuration input {result['status']}: {source} "
                f"(snapshot {snapshot_path}){detail}"
            )
    if not checks:
        errors.append("No archived configuration inputs were available for rechecking")
    return checks, errors
