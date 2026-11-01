"""Run a bounded command and archive private evidence in a new run directory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vllm_prefix_scheduler.run_evidence import run_main  # noqa: E402

if __name__ == "__main__":
    run_main()
