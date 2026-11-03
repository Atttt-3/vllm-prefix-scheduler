"""Recompute M5 statistics from request and actual runner evidence."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_prefix_scheduler.m5_contract import (  # noqa: E402
    MODES,
    AuditError,
    audit_matrix,
    audit_run,
    read_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--inputs", required=True, type=Path)
    run.add_argument("--run", required=True, type=Path)
    run.add_argument("--pair")
    run.add_argument("--mode", choices=MODES)
    run.add_argument("--phase", choices=("formal", "calibration", "diagnostic"), default="formal")
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--index", required=True, type=Path)
    for command in (run, matrix):
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "run":
            result = audit_run(
                args.config,
                args.inputs,
                args.run,
                pair_id=args.pair,
                mode=args.mode,
                phase=args.phase,
            )
        else:
            result = audit_matrix(args.config, read_json(args.index))
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x") as stream:
                stream.write(payload)
        else:
            print(payload, end="")
    except (AuditError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"M5 evidence rejected: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
