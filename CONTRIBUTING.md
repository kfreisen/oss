# Contributing

## Setup

You need [`uv`](https://docs.astral.sh/uv/) — it manages Python versions itself, so nothing
else is required. Working on `slatekit` also needs a Rust toolchain:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then, for whichever package you're touching:

```bash
cd packages/<name>
uv sync --all-extras
uv run pytest
```

Install the hooks once, at the repo root:

```bash
uvx pre-commit install
```

## Repository shape

Each package under `packages/` is standalone: its own `pyproject.toml`, its own `uv.lock`, its
own release tag. There is no uv workspace, and **packages must never import each other** —
`tools/check_no_cross_imports.py` fails the build if they do. If two packages want the same
helper, duplicate it; a shared internal package would defeat independent publishing.

Linting is configured once, in `/ruff.toml`. A package `pyproject.toml` must **not** contain a
`[tool.ruff]` section — ruff would then treat that file as the config root and silently detach
the package from the baseline. `tools/check_no_local_ruff.py` enforces this.

## The bar for a change

- **Tests.** Coverage thresholds are set per package in `[tool.coverage.report] fail_under`.
  They are not negotiable downward; if code is hard to cover, that is usually a design signal.
- **Types.** `mypy --strict` over `src/`. Every package ships `py.typed`.
- **Docstrings.** Google convention, enforced by ruff's `D` rules. mkdocstrings renders these
  directly into the published docs, so they are user-facing text.

## Benchmarks

Every package has a `benchmarks/baselines/` directory holding the *previous* implementation as
real, imported, tested code. This is the point of the exercise: a speedup claim only means
something if the slow version still runs and still produces the same answer.

Two rules follow from that:

1. **`tests/test_parity.py` is the most important test in each package.** It runs the baseline
   and the optimized path on the same inputs and asserts they agree. If you change the fast
   path, this is what catches you having changed its behavior rather than its speed.
2. **CI never asserts on wall-clock time.** Shared runners vary by roughly 2×, and a perf gate
   that flakes is a perf gate everyone learns to ignore. CI runs the benchmark suite in
   `--quick` mode only, to prove the benchmark code still executes.

Real numbers are produced on a known machine and committed by hand:

```bash
make bench PKG=<name>
```

That writes `packages/<name>/benchmarks/results/<hardware-id>/<date>-<sha>.json`, including a
hardware block. The docs site renders those files at build time, so published tables can never
drift from the committed data. Include the hardware you ran on in the PR.

## Examples

Examples are [marimo](https://marimo.io) notebooks stored as plain Python under
`packages/<name>/examples/`. CI executes them via `marimo export html`, which fails on any
exception — so an example must run in under a minute, must not touch the network, and must read
only from small committed fixtures in `examples/data/`.

## Releasing

Tag `<package>-v<version>`, e.g. `washbook-v0.2.0`. The release workflow verifies the tag
matches `project.version`, builds, and publishes to PyPI via Trusted Publishing. The `pypi`
GitHub Environment requires a manual approval, so a stray tag push cannot publish on its own.
