"""Generate the persistent dashboard and status document from one progress file."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if generated files are stale")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from vllm_prefix_scheduler.progress import render_state, validate_progress

    data = json.loads((root / "progress/project-status.json").read_text())
    counts = validate_progress(data, root)
    template = (root / "progress/dashboard.template.html").read_text()
    marker = "__PROJECT_STATUS_JSON__"
    if template.count(marker) != 1:
        raise ValueError("Dashboard template must contain one data placeholder")
    serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    html = template.replace(marker, serialized)
    outputs = {
        root / "progress/index.html": html,
        root / "docs/state.md": render_state(data),
    }
    stale = []
    for path, content in outputs.items():
        if args.check:
            if not path.exists() or path.read_text() != content:
                stale.append(str(path.relative_to(root)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    if stale:
        raise SystemExit("Regenerate project progress: " + ", ".join(stale))
    print(json.dumps({"progress": "current", "task_states": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
