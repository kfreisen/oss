# oss

Five independent Python libraries, published separately to PyPI, developed in one repository.

They are unrelated in subject matter. What they share is provenance and method: each was pulled
out of a working private system rather than written as a demonstration, and each ships a
**committed before/after benchmark against a real baseline** — where the "before" lives in the
repository as importable, tested code, not as a claim in a README.

| Package | What it does | Measured against |
| --- | --- | --- |
| [slatekit](slatekit/index.md) | Roster optimization — randomized greedy construction and lazy-greedy submodular portfolio selection, in Rust | MILP (PuLP/CBC, OR-Tools) |
| [washbook](washbook/index.md) | After-tax backtest accounting — vectorized wash-sale blackout and an IRS §1091 deferral ledger | Serial per-symbol Python |
| [venuematch](venuematch/index.md) | Blocked fuzzy entity resolution | Naive all-pairs O(n²) |
| [mcharness](mcharness/index.md) | Reproducible batched Monte Carlo on Ray | Serial, and uncached-parallel |
| [impliedmove](impliedmove/index.md) | Options-implied move and event probability, zero dependencies | Unadjusted prior (Brier score) |

## The one idea these have in common

A performance claim is only meaningful next to the thing it beats, and only if both still
produce the same answer.

So every package keeps its predecessor alive. `benchmarks/baselines/` holds the previous
implementation as real code that is imported, executed, and tested; `tests/test_parity.py`
asserts that it and the current implementation agree on the same inputs. If the fast path ever
drifts, that test fails before any benchmark gets a chance to report a flattering number.

Timings are recorded on named hardware and committed as JSON. The tables on this site are
rendered from those files at build time, so a published figure cannot disagree with the data in
the repository. CI never times anything — see [How benchmarks work](benchmarks.md) for why.

## Status

All five are at `0.0.1.dev0`: the names are reserved and the build, test, docs, and release
pipelines are working end to end. Implementations land one at a time, in the order listed above.
