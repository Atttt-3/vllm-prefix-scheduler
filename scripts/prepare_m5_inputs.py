"""Generate and freeze M5 paired arrival tables using the pinned tokenizer."""

import argparse
import json
from pathlib import Path

from vllm_prefix_scheduler.m5_workload import SCENARIOS, prepare_workload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("config", "model", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scenario", choices=SCENARIOS)
    source.add_argument("--batch", type=Path, help="Ordered list of table specifications")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--count", type=int)
    arrival = parser.add_mutually_exclusive_group()
    arrival.add_argument("--rate")
    arrival.add_argument("--interval-ms")
    parser.add_argument("--mechanism", choices=("pressure", "ample"))
    args = parser.parse_args()
    if args.batch is not None:
        if any(
            v is not None
            for v in (args.seed, args.count, args.rate, args.interval_ms, args.mechanism)
        ):
            parser.error("batch specs cannot be combined with single-table options")
        specs = json.loads(args.batch.read_text())
        if not isinstance(specs, list) or not specs:
            parser.error("batch requires a nonempty ordered specification list")
        ids = [spec["id"] for spec in specs]
        if len(ids) != len(set(ids)) or not all(
            value and all(ch in "abcdefghijklmnopqrstuvwxyz0123456789-" for ch in value)
            for value in ids
        ):
            parser.error("batch table IDs must be unique and path safe")
        args.output.mkdir(parents=True, exist_ok=True)
        outputs = [args.output / f"{name}.json" for name in ids]
        if any(path.exists() for path in outputs):
            parser.error("refusing to overwrite a frozen arrival table")
    else:
        if args.seed is None:
            parser.error("single-table generation requires --seed")
        specs = [
            {
                "scenario": args.scenario,
                "seed": args.seed,
                "rate_rps": args.rate,
                "interval_ms": args.interval_ms,
                "count": args.count,
                "mechanism": args.mechanism,
            }
        ]
        outputs = [args.output]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), local_files_only=True, trust_remote_code=False
    )
    config = json.loads(args.config.read_text())
    for spec, output in zip(specs, outputs, strict=True):
        result = prepare_workload(
            config,
            lambda text: tokenizer.encode(text, add_special_tokens=False),
            spec["scenario"],
            spec["seed"],
            rate=spec.get("rate_rps"),
            interval_ms=spec.get("interval_ms"),
            count=spec.get("count"),
            mechanism=spec.get("mechanism") or "pressure",
        )
        result["tokenizer_class"] = f"{type(tokenizer).__module__}.{type(tokenizer).__name__}"
        with output.open("x") as stream:
            json.dump(result, stream, separators=(",", ":"), allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
