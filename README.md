# oss

Five independent Python libraries, each published separately to PyPI, developed in one
repository.

They are unrelated in subject matter. What they have in common is that each one was pulled out
of a working private system rather than written as a demo, and each ships a **committed
before/after benchmark against a real baseline implementation** — the "before" lives in the
repo as importable, tested code, not as a claim in a README.

| Package | What it does | Benchmarked against |
| --- | --- | --- |
| [`slatekit`](packages/slatekit) | Roster/lineup optimization — randomized greedy construction and lazy-greedy submodular portfolio selection, in Rust | MILP (PuLP/CBC, OR-Tools) |
| [`washbook`](packages/washbook) | After-tax backtest accounting — vectorized wash-sale blackout and an IRS §1091 deferral ledger | Serial per-symbol Python |
| [`venuematch`](packages/venuematch) | Blocked fuzzy entity resolution across data sources | Naive all-pairs O(n²) |
| [`mcharness`](packages/mcharness) | Reproducible batched Monte Carlo on Ray | Serial, and uncached-parallel |
| [`impliedmove`](packages/impliedmove) | Options-implied move and event probability, zero dependencies | Unadjusted prior (Brier score) |

Documentation: <https://kfreisen.github.io/oss/>

## Working in this repo

Each package under `packages/` is a fully standalone project with its own `pyproject.toml` and
its own `uv.lock`. There is deliberately **no uv workspace** — a workspace shares one lockfile
and one resolution, and these packages have incompatible dependency sets (`impliedmove` has no
dependencies at all and should not be developed inside Ray's resolution). Independent locks also
keep CI path-filtered and let each package verify its own declared lower bounds.

Packages never import each other; `tools/check_no_cross_imports.py` enforces that.

```bash
make test PKG=washbook      # one package
make test-all               # all five
make lint                   # ruff across the repo
make bench PKG=washbook     # full benchmark run, writes JSON
make docs                   # serve the docs site locally
make lock-all               # re-lock every package
```

You need [`uv`](https://docs.astral.sh/uv/); it manages the Python versions itself. `slatekit`
additionally needs a Rust toolchain (`rustup`).

## License

Apache-2.0. See [LICENSE](LICENSE).
