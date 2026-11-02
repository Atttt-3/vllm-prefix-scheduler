UV ?= $(if $(wildcard .tools/bin/uv),.tools/bin/uv,uv)
NATIVE_PYTHON ?= .venv-gpu/bin/python
export UV_CACHE_DIR := $(CURDIR)/.cache/uv
export UV_PYTHON_INSTALL_DIR := $(CURDIR)/.tools/python
export UV_PROJECT_ENVIRONMENT := $(CURDIR)/.venv

.PHONY: dev check lint format test evidence build script-syntax run-evidence check-run-evidence native-replay native-trace-check native-m2a native-m2a-trace-check native-m2b native-m2b-compat native-m2b-trace-check

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

# M2A has its own factory, complete matrix contract and independent trace auditor.
native-m2a:
	"$(NATIVE_PYTHON)" scripts/replay_m2a.py $(value ARGS)

native-m2a-trace-check:
	"$(NATIVE_PYTHON)" tests/native/m2a_trace_check.py $(value ARGS)

# M2A's historical guarded rejection stays versioned at its accepted commit.
# M2B has its own complete leaf matrix and current off/prefix compatibility suite.
native-m2b:
	"$(NATIVE_PYTHON)" scripts/replay_m2b.py $(value ARGS) --suite full

native-m2b-compat:
	"$(NATIVE_PYTHON)" scripts/replay_m2b.py $(value ARGS) --suite compat

native-m2b-trace-check:
	"$(NATIVE_PYTHON)" tests/native/m2b_trace_check.py $(value ARGS)

build:
	"$(UV)" build --no-sources

script-syntax:
	@for script in scripts/*.sh scripts/*.sbatch; do bash -n "$$script" || exit; done

check: lint script-syntax test evidence build

# M3 integration uses the existing runtime; these targets never install GPU dependencies.
.PHONY: m3-real-model m3-trace-check m3-fail-closed
m3-real-model:
	"$(NATIVE_PYTHON)" scripts/run_m3_suite.py $(value ARGS)

m3-trace-check:
	"$(NATIVE_PYTHON)" scripts/check_m3_trace.py $(value ARGS)

m3-fail-closed:
	"$(NATIVE_PYTHON)" scripts/check_m3_fail_closed.py $(value ARGS)
