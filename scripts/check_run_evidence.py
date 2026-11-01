"""Check a collected run without converting command success into acceptance."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vllm_prefix_scheduler.run_evidence import check_main  # noqa: E402

if __name__ == "__main__":
    check_main()
