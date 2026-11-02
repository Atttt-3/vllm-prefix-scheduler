"""Freeze real tokenizer output before starting the inference evidence collector."""

import argparse
import json
from pathlib import Path


def main():
    from transformers import AutoTokenizer

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), local_files_only=True, trust_remote_code=False
    )
    texts = [v["text"] for v in config["requests"].values()] + [
        f"Give one short fact about {x}." for x in ["GPUs", "oceans", "planets"]
    ]
    by_text = {}
    for text in texts:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        if not ids or any(type(t) is not int or t < 0 for t in ids):
            raise RuntimeError("Expected nonempty real integer tokenizer IDs")
        by_text[text] = ids
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(
            {
                "model_revision": config["model_revision"],
                "tokenizer_revision": config["model_revision"],
                "by_text": by_text,
            },
            stream,
            indent=2,
        )
        stream.write("\n")


if __name__ == "__main__":
    main()
