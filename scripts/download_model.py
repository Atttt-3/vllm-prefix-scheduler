"""Download a pinned model snapshot into the isolated remote project."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import model_info, snapshot_download


def main():
    root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
    pins = json.loads((root / "configs" / "upstream.json").read_text())
    revision = pins["model"]["revision"]
    model_dir = root / pins["model"]["directory"]
    info = model_info("Qwen/Qwen3-4B", revision=revision, files_metadata=True)
    assert info.sha == revision
    print(f"Downloading Qwen/Qwen3-4B revision {revision}", flush=True)
    snapshot_download(
        "Qwen/Qwen3-4B",
        revision=revision,
        local_dir=str(model_dir),
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"],
        max_workers=3,
    )
    weights = sorted(model_dir.glob("*.safetensors"))
    assert len(weights) == 3, [str(p) for p in weights]
    expected = {
        item.rfilename: item.size
        for item in info.siblings
        if item.rfilename.endswith(".safetensors")
    }
    for path in weights:
        assert path.stat().st_size == expected[path.name], path
    manifest = {
        "repo_id": "Qwen/Qwen3-4B",
        "revision": revision,
        "local_dir": str(model_dir),
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "files": [
            {"name": str(p.relative_to(model_dir)), "bytes": p.stat().st_size}
            for p in sorted(model_dir.iterdir())
            if p.is_file()
        ],
        "weight_file_bytes": sum(p.stat().st_size for p in weights),
    }
    (root / ".local").mkdir(exist_ok=True)
    (root / ".local" / "model-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
