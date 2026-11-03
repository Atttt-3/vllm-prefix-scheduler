"""Audit all M4 submission attempts, independent budget and authoritative release."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_prefix_scheduler.m4_contract import audit_resources, require_assertions  # noqa: E402


def main():
    require_assertions()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--selected-commit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_resources(
        json.loads(args.resources.read_text()),
        json.loads(args.config.read_text()),
        selected_commit=args.selected_commit,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
