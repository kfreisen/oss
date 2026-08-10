# Convenience fan-out over the five standalone packages.
#
# Everything here is a thin wrapper over uv — nothing in this file is required to
# build, test, or publish any package. CI calls the underlying commands directly.

PKGS := slatekit washbook venuematch mcharness impliedmove

.PHONY: help test test-all lint fmt typecheck bench bench-quick examples docs docs-build lock-all lowest-bounds clean

help:
	@echo "make test PKG=<name>     run one package's tests with coverage"
	@echo "make test-all            run every package's tests"
	@echo "make lint                ruff check + format check, whole repo"
	@echo "make fmt                 ruff format, whole repo"
	@echo "make typecheck PKG=<n>   mypy one package"
	@echo "make bench PKG=<name>    full benchmark run -> benchmarks/results/"
	@echo "make bench-quick PKG=<n> smoke-run benchmarks (no timings recorded)"
	@echo "make examples PKG=<name> execute every marimo example"
	@echo "make docs                serve the docs site on :8000"
	@echo "make lock-all            re-lock every package"
	@echo "make lowest-bounds       verify declared lower bounds resolve and pass"
	@echo ""
	@echo "packages: $(PKGS)"

_require_pkg:
	@test -n "$(PKG)" || { echo "error: set PKG=<name>, one of: $(PKGS)"; exit 1; }
	@test -d packages/$(PKG) || { echo "error: no such package: $(PKG)"; exit 1; }

test: _require_pkg
	cd packages/$(PKG) && uv run pytest --cov --cov-report=term-missing

test-all:
	@for p in $(PKGS); do echo "=== $$p ==="; $(MAKE) --no-print-directory test PKG=$$p || exit 1; done

lint:
	uvx ruff check .
	uvx ruff format --check .

fmt:
	uvx ruff format .
	uvx ruff check --fix .

typecheck: _require_pkg
	cd packages/$(PKG) && uv run mypy src

bench: _require_pkg
	cd packages/$(PKG) && uv run --group bench pytest benchmarks/ \
		--benchmark-json=.benchmark.json --benchmark-only
	uv run --no-project --python 3.12 --with psutil \
		tools/bench_report.py packages/$(PKG)/.benchmark.json --package $(PKG) --write

bench-quick: _require_pkg
	cd packages/$(PKG) && uv run --group bench pytest benchmarks/ -q --quick

examples: _require_pkg
	cd packages/$(PKG) && for nb in examples/*.py; do \
		echo "--- $$nb"; uv run marimo export html "$$nb" -o /dev/null || exit 1; \
	done

docs:
	uv run --project docs mkdocs serve

docs-build:
	uv run --project docs mkdocs build --strict

lock-all:
	@for p in $(PKGS); do echo "=== $$p ==="; (cd packages/$$p && uv lock) || exit 1; done

lowest-bounds:
	@for p in $(PKGS); do \
		echo "=== $$p ==="; \
		(cd packages/$$p && uv sync --resolution lowest-direct --all-extras && uv run pytest -q) || exit 1; \
	done

clean:
	find . -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
	find . -name '.pytest_cache' -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf site packages/*/.benchmark.json packages/slatekit/rust/target
