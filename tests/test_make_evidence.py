"""Exercise the public Make targets with real collectors and shell-quoted arguments."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make(target: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["make", target, "ARGS=" + shlex.join(args)],
        cwd=ROOT,
        env={**os.environ, "UV_OFFLINE": "true", "UV_PYTHON_DOWNLOADS": "never"},
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_make_preserves_literal_dollars_and_shell_quoted_command_arguments(tmp_path):
    # Both collector options and the checker path pass through ARGS. These values must
    # survive make expansion as well as shell parsing, without executing substitutions.
    directory = tmp_path / "run $5 $(not-a-command) ${MAKE_LITERAL} 中文"
    literal = "price $5 $(printf expanded) ${MAKE_LITERAL} spaces 中文 'single' \"double\""
    command = [sys.executable, "-c", "import sys; print(sys.argv[1])", literal]
    criterion = "保留 $5、$(printf altered)、${MAKE_LITERAL} 与 '引号'"
    result = make(
        "run-evidence",
        [
            "--task-id",
            "make-argument-regression",
            "--kind",
            "local",
            "--output",
            str(directory),
            "--timeout",
            "5",
            "--criterion",
            criterion,
            "--",
            *command,
        ],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["argv"] == command
    assert manifest["criteria"][0]["description"] == criterion
    assert (directory / "stdout.log").read_text().strip() == literal
    checked = make("check-run-evidence", [str(directory), "--require-command-success"])
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert '"integrity": "valid"' in checked.stdout


def test_make_accepts_existing_double_quoted_criterion_syntax(tmp_path):
    directory = tmp_path / "documented-example"
    # Preserve the ARGS='... --criterion "..." -- COMMAND' form in docs/workflows.md.
    args = (
        f"--task-id make-documentation --kind local --output {shlex.quote(str(directory))} "
        f'--timeout 5 --config configs/upstream.json --criterion "本地命令实际执行且通过" '
        f"-- {shlex.quote(sys.executable)} -c 'print(123)'"
    )
    result = subprocess.run(
        ["make", "run-evidence", "ARGS=" + args],
        cwd=ROOT,
        env={**os.environ, "UV_OFFLINE": "true", "UV_PYTHON_DOWNLOADS": "never"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["criteria"][0]["description"] == "本地命令实际执行且通过"
    assert (directory / "stdout.log").read_text().strip() == "123"
