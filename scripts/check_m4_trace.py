"""Audit complete M4 raw native/GPU/CPU observations independently."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_prefix_scheduler.m4_contract import audit, read_events, require_assertions, sha256  # noqa: E402


def main():
    require_assertions()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--layer", choices=("native", "gpu", "cpu"), required=True)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--partial", action="store_true", help="Precheck only; never complete acceptance"
    )
    parser.add_argument(
        "--contract-fixture", action="store_true", help="Synthetic local contract test only"
    )
    args = parser.parse_args()
    result = audit(
        json.loads(args.config.read_text()),
        read_events(args.trace),
        layer=args.layer,
        sources=None if args.sources is None else json.loads(args.sources.read_text()),
        resources=None if args.resources is None else json.loads(args.resources.read_text()),
        partial=args.partial,
        contract_fixture=args.contract_fixture,
        root=None if args.contract_fixture else ROOT,
        frozen_inputs=None if args.inputs is None else json.loads(args.inputs.read_text()),
        inputs_sha256=None if args.inputs is None else sha256(args.inputs),
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {key: result[key] for key in ("scope", "accepted", "project_commit", "case_count")}
        )
    )


if __name__ == "__main__":
    main()
