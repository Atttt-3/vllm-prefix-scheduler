"""Verify complete M5 private allocation resource and terminal records."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vllm_prefix_scheduler.m5_resources import audit_attempt, audit_history  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    try:
        if sys.flags.optimize:
            raise ValueError("run unoptimized resource auditor")
        records = audit_history(args.path) if args.history else [audit_attempt(args.path)]
        print(json.dumps({"accepted": True, "attempts": records}, default=str))
    except (ValueError, OSError, TypeError, KeyError) as error:
        parser.exit(1, f"M5 resource gate: {error}\n")


if __name__ == "__main__":
    main()
