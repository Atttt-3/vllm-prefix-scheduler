"""Lightweight contract tests; no vLLM import or native behavior substitution."""

import copy
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from vllm_prefix_scheduler.config import PrefixSchedulerConfig, parse_prefix_config


def test_defaults_and_unrelated_additional_config():
    assert parse_prefix_config() == PrefixSchedulerConfig()
    assert parse_prefix_config({}) == PrefixSchedulerConfig()
    assert parse_prefix_config({"unrelated": 3}) == PrefixSchedulerConfig()


def test_parser_does_not_mutate_input_and_config_is_frozen():
    raw = {"prefix_scheduler": {"mode": "prefix", "window": 3, "max_wait_ms": 1.5}}
    before = copy.deepcopy(raw)
    config = parse_prefix_config(raw)
    assert config == PrefixSchedulerConfig("prefix", 3, 1.5)
    assert raw == before
    with pytest.raises(FrozenInstanceError):
        config.mode = "off"


@pytest.mark.parametrize("raw", [[], "off", 1, True])
def test_reject_invalid_additional_container(raw):
    with pytest.raises(ValueError, match="additional_config"):
        parse_prefix_config(raw)


@pytest.mark.parametrize("raw", [None, [], "off", 1, True])
def test_reject_invalid_prefix_container(raw):
    with pytest.raises(ValueError, match="prefix_scheduler"):
        parse_prefix_config({"prefix_scheduler": raw})


@pytest.mark.parametrize("value", [None, True, 0, "", "PREFIX", "unknown", []])
def test_reject_invalid_mode(value):
    with pytest.raises(ValueError, match="mode"):
        parse_prefix_config({"prefix_scheduler": {"mode": value}})


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, "16", None])
def test_reject_invalid_window_even_when_off(value):
    with pytest.raises(ValueError, match="window"):
        parse_prefix_config({"prefix_scheduler": {"mode": "off", "window": value}})


@pytest.mark.parametrize("value", [True, False, -1, -0.1, float("inf"), float("nan"), "0", None])
def test_reject_invalid_wait_contract(value):
    with pytest.raises(ValueError, match="max_wait_ms"):
        parse_prefix_config({"prefix_scheduler": {"max_wait_ms": value}})


@pytest.mark.parametrize("value", [0, 0.0, 2.5, 10**400])
def test_accept_nonnegative_finite_wait_contract(value):
    assert parse_prefix_config({"prefix_scheduler": {"max_wait_ms": value}}).max_wait_ms == value


@pytest.mark.parametrize("value", [0, 0.0, 2.5, 0.1, 10**400])
def test_guarded_accepts_wait_contract(value):
    assert parse_prefix_config(
        {"prefix_scheduler": {"mode": "guarded", "max_wait_ms": value}}
    ) == PrefixSchedulerConfig(mode="guarded", max_wait_ms=value)


def test_unknown_key_fails():
    with pytest.raises(ValueError, match="unknown"):
        parse_prefix_config({"prefix_scheduler": {"windwo": 16}})


def test_config_import_works_without_site_packages():
    source = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            f"import sys; sys.path.insert(0, {source!r}); "
            "from vllm_prefix_scheduler.config import parse_prefix_config; "
            "assert parse_prefix_config().mode == 'off'; assert 'vllm' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
