"""Content verification for the fixed M3 environment; no installation or model loading."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_sources(root, model, config):
    root, model = Path(root).resolve(), Path(model).resolve()
    if platform.python_version() != "3.12.16":
        raise RuntimeError("M3 requires Python 3.12.16")
    versions = {
        n: importlib.metadata.version(n)
        for n in ("vllm", "torch", "transformers", "tokenizers", "safetensors")
    }
    if versions["vllm"] != "0.32.0" or versions["torch"] != "2.13.0+cu129":
        raise RuntimeError("M3 fixed native dependency mismatch")
    distribution = importlib.metadata.distribution("vllm")
    old = json.loads((root / "configs/native-m2b.json").read_text())["source_sha256"]
    expected = {**old, **config["extra_upstream_sha256"]}
    native = {}
    for name, expected_hash in expected.items():
        path = Path(distribution.locate_file(name))
        actual = sha256(path)
        if actual != expected_hash:
            raise RuntimeError(f"Fixed upstream source mismatch: {name}")
        native[name] = {"sha256": actual, "path": str(path.resolve())}
    revision = config["model_revision"]
    tree_path = model / ".cache/huggingface/trees" / f"{revision}.json"
    tree = json.loads(tree_path.read_text())["files"]
    index = json.loads((model / "model.safetensors.index.json").read_text())
    required = sorted(
        set(index["weight_map"].values())
        | {
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        }
    )
    files = []
    for name in required:
        path = model / name
        metadata = (
            (model / ".cache/huggingface/download" / f"{name}.metadata").read_text().splitlines()
        )
        expected_file = tree[name]
        frozen_file = config["model_files"][name]
        actual = sha256(path)
        if metadata[0] != revision or path.stat().st_size != expected_file["size"]:
            raise RuntimeError(f"Model revision/size mismatch: {name}")
        if "lfs_sha256" in expected_file:
            digest, wanted = actual, expected_file["lfs_sha256"]
        else:
            blob = path.read_bytes()
            digest = hashlib.sha1(f"blob {len(blob)}\0".encode() + blob).hexdigest()
            wanted = expected_file["blob_id"]
        if (
            digest != wanted
            or metadata[1] != wanted
            or wanted != frozen_file["content_id"]
            or path.stat().st_size != frozen_file["size"]
        ):
            raise RuntimeError(f"Model content mismatch: {name}")
        files.append(
            {
                "name": name,
                "size": path.stat().st_size,
                "sha256": actual,
                "revision": revision,
                "verified_content_id": wanted,
            }
        )
    project = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
    if dirty:
        raise RuntimeError("M3 run requires a clean frozen checkout")
    source_files = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "src", "scripts", "configs", "tests", "Makefile"],
        text=True,
    ).splitlines()
    return {
        "project_commit": project,
        "project_dirty": False,
        "python_version": platform.python_version(),
        "packages": versions,
        "upstream_commit": config["upstream_commit"],
        "upstream_modules": native,
        "source_sha256": {n: sha256(root / n) for n in source_files},
        "upstream_sha256": {n: v["sha256"] for n, v in native.items()},
        "model": {
            "revision": revision,
            "tokenizer_revision": revision,
            "directory": str(model),
            "tree_sha256": sha256(tree_path),
            "files": files,
        },
    }
