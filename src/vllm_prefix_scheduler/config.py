"""Standard-library-only parsing of the prefix scheduler configuration."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PrefixSchedulerConfig:
    mode: str = "off"
    window: int = 16
    max_wait_ms: int | float = 500


def parse_prefix_config(additional_config: dict | None = None) -> PrefixSchedulerConfig:
    """Read without modifying vLLM's additional_config or importing vLLM."""
    if additional_config is None:
        additional_config = {}
    if not isinstance(additional_config, dict):
        raise ValueError("additional_config must be an object")
    raw = additional_config.get("prefix_scheduler", {})
    if not isinstance(raw, dict):
        raise ValueError("additional_config.prefix_scheduler must be an object")
    unknown = set(raw) - {"mode", "window", "max_wait_ms"}
    if unknown:
        raise ValueError("prefix_scheduler contains unknown configuration keys")
    mode = raw.get("mode", "off")
    if type(mode) is not str or mode not in {"off", "prefix", "guarded"}:
        raise ValueError("prefix_scheduler.mode must be off, prefix or guarded")
    window = raw.get("window", 16)
    if type(window) is not int or window <= 0:
        raise ValueError("prefix_scheduler.window must be a positive integer (not bool)")
    max_wait_ms = raw.get("max_wait_ms", 500)
    if (
        type(max_wait_ms) not in (int, float)
        or max_wait_ms < 0
        or (type(max_wait_ms) is float and not math.isfinite(max_wait_ms))
    ):
        raise ValueError("prefix_scheduler.max_wait_ms must be finite and nonnegative (not bool)")
    return PrefixSchedulerConfig(mode=mode, window=window, max_wait_ms=max_wait_ms)
