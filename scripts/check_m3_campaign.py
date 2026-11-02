"""Audit a new M3 campaign while retaining the blocked historical gate."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_prefix_scheduler.m3_campaign_contract import audit_campaign  # noqa: E402


def arguments(parser):
    for name in (
        "campaign-config",
        "authorization",
        "history-index",
        "config",
        "inputs",
        "trace-dir",
        "resources",
        "native-index",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main():
    args = arguments(argparse.ArgumentParser(description=__doc__)).parse_args()
    values = vars(args).copy()
    output = values.pop("output")
    result = audit_campaign(root=ROOT, **values)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "scope",
                    "campaign_acceptance",
                    "original_full_history_gate",
                    "project_commit",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
