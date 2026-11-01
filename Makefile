UV ?= $(if $(wildcard .tools/bin/uv),.tools/bin/uv,uv)
export UV_CACHE_DIR := $(CURDIR)/.cache/uv
export UV_PYTHON_INSTALL_DIR := $(CURDIR)/.tools/python
export UV_PROJECT_ENVIRONMENT := $(CURDIR)/.venv

.PHONY: dev check lint format test evidence build script-syntax

dev:
	$(UV) sync --locked --dev

lint:
	$(UV) run --frozen ruff check .
	$(UV) run --frozen ruff format --check .

format:
	$(UV) run --frozen ruff check --fix .
	$(UV) run --frozen ruff format .

test:
	$(UV) run --frozen pytest

evidence:
	$(UV) run --frozen prefix-evidence-check evidence/2026-11-01-rtx3090/validation.json

build:
	$(UV) build --no-sources

script-syntax:
	@for script in scripts/*.sh scripts/*.sbatch; do bash -n "$$script" || exit; done

check: lint script-syntax test evidence build
