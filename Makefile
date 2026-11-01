UV ?= $(if $(wildcard .tools/bin/uv),.tools/bin/uv,uv)
export UV_CACHE_DIR := $(CURDIR)/.cache/uv
export UV_PYTHON_INSTALL_DIR := $(CURDIR)/.tools/python
export UV_PROJECT_ENVIRONMENT := $(CURDIR)/.venv

.PHONY: dev check lint format test evidence build script-syntax progress progress-check run-evidence check-run-evidence

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

progress:
	"$(UV)" run --frozen python scripts/build_progress.py

progress-check:
	"$(UV)" run --frozen python scripts/build_progress.py --check

# ARGS supplies collector/checker arguments; commands run locally, without GPU allocation.
run-evidence:
	"$(UV)" run --frozen python scripts/run_evidence.py $(ARGS)

check-run-evidence:
	"$(UV)" run --frozen python scripts/check_run_evidence.py $(ARGS)

build:
	"$(UV)" build --no-sources

script-syntax:
	@for script in scripts/*.sh scripts/*.sbatch; do bash -n "$$script" || exit; done

check: lint script-syntax progress-check test evidence build
