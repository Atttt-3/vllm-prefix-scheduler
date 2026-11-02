"""Freeze exact-length public synthetic token inputs with the real pinned tokenizer."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def prepare(config, model):
    from transformers import AutoTokenizer

    if (
        len(
            {
                config["inputs"][key]
                for key in ("pressure_tokens_each", "chunk_tokens", "pending_cancel_tokens")
            }
        )
        != 1
    ):
        raise RuntimeError(
            "M4 shared long input requires equal declared chunk/pressure/cancel lengths"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model), local_files_only=True, trust_remote_code=False
    )
    phrases = {
        "long_a": "Public synthetic astronomy reference: planets orbit stars and telescopes observe distant galaxies. ",
        "long_b": "Public synthetic ocean reference: whales swim near islands and sailing boats follow changing tides. ",
        "short": "Public synthetic computer reference: memory stores data and processors execute instructions. ",
        "hot": "Public synthetic shared prefix: GPU inference can reuse identical cached prompt blocks. ",
        "cold": "Public synthetic garden reference: flowers grow in soil while birds visit trees. ",
    }
    lengths = {
        "long_a": config["inputs"]["pressure_tokens_each"],
        "long_b": config["inputs"]["pressure_tokens_each"],
        "short": config["inputs"]["short_tokens"],
        "hot": config["inputs"]["prefix_shared_tokens"] + 32,
        "cold": config["inputs"]["prefix_shared_tokens"] + 32,
    }
    values = {}
    for key, phrase in phrases.items():
        length = lengths[key]
        ids = tokenizer.encode(phrase * (length // 8 + 1), add_special_tokens=False)[:length]
        if len(ids) != length or any(type(token) is not int or token < 0 for token in ids):
            raise RuntimeError("Real tokenizer did not supply exact frozen token length")
        values[key] = {
            "token_ids": ids,
            "length": len(ids),
            "token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "construction": "crop real tokenizer encoding of repeated public synthetic phrase",
        }
    if values["long_a"]["token_ids"][:16] == values["long_b"]["token_ids"][:16]:
        raise RuntimeError("Pressure prompts share their first physical block")
    return {
        "schema_version": 1,
        "task_id": config["task_id"],
        "model_revision": config["model_revision"],
        "tokenizer_revision": config["tokenizer_revision"],
        "tokenizer_class": f"{type(tokenizer).__module__}.{type(tokenizer).__name__}",
        "requests": values,
    }


def main():
    if not __debug__ or not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise RuntimeError("M4 input preparation requires unoptimized allocated runtime")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = prepare(json.loads(args.config.read_text()), args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
