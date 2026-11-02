UV ?= $(if $(wildcard .tools/bin/uv),.tools/bin/uv,uv)
NATIVE_PYTHON ?= .venv-gpu/bin/python
export UV_CACHE_DIR := $(CURDIR)/.cache/uv
export UV_PYTHON_INSTALL_DIR := $(CURDIR)/.tools/python
export UV_PROJECT_ENVIRONMENT := $(CURDIR)/.venv

.PHONY: dev check lint format test evidence build script-syntax run-evidence check-run-evidence native-replay native-trace-check

dev:
	"$(UV)" sync --locked --dev

lint:
	"$(UV)" run --frozen ruff check .
	"$(UV)" run --frozen ruff format --check .

format:
	"$(UV)" run --frozen ruff check --fix .
	"$(UV)" run --frozen ruff format .

test:
	"$(UV)" run --frozen pytest

evidence:
	@for record in evidence/*/validation.json; do "$(UV)" run --frozen prefix-evidence-check "$$record" || exit; done

# ARGS is shell-quoted argument text. Read its raw value so make cannot expand literal $.
# The recipe shell still parses ARGS; quote shell metacharacters for that shell.
run-evidence:
	"$(UV)" run --frozen python scripts/run_evidence.py $(value ARGS)

check-run-evidence:
	"$(UV)" run --frozen python scripts/check_run_evidence.py $(value ARGS)

# Use the existing Linux interpreter directly; never sync the dev lock into it.
native-replay:
	"$(NATIVE_PYTHON)" scripts/replay_bc.py $(value ARGS)

native-trace-check:
	"$(NATIVE_PYTHON)" tests/native/test_native_replay.py $(value ARGS)

build:
	"$(UV)" build --no-sources

script-syntax:
	@for script in scripts/*.sh scripts/*.sbatch; do bash -n "$$script" || exit; done

check: lint script-syntax test evidence build
