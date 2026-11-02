"""Independently audit source-bound M3 raw traces (standard library only)."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_prefix_scheduler.m3_contract import audit_trace, read_events  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--engine-id", help="Partial engine precheck only; never complete M3 acceptance")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_trace(
        read_events(args.trace_dir), json.loads(args.config.read_text()),
        resources=None if args.resources is None else json.loads(args.resources.read_text()),
        engine_id=args.engine_id,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("scope", "accepted", "case_count", "project_commit")}))


if __name__ == "__main__":
    main()
