# slatekit

Fast roster optimization: randomized greedy lineup construction and lazy-greedy submodular
portfolio selection, implemented in Rust.

> **Status: pre-release.** `0.0.1.dev0` is a scaffold that reserves the name and exercises the
> build and release pipeline. The solver lands in `0.1.0`.

## The problem

Two different problems, usually conflated:

1. **Build one valid roster.** Pick players under a salary cap, filling positional slots, subject
   to group limits ("at most 6 from one team"). This is an integer program, and an ILP solver
   answers it exactly.
2. **Build a *portfolio* of rosters.** Pick 150 lineups that collectively do well across
   simulated outcomes. Optimality per lineup is close to worthless here — 150 optimal lineups are
   150 nearly identical lineups. What matters is diverse coverage of the outcome space.

For (2), asking an ILP for 150 solutions is both slow and the wrong objective. `slatekit`
generates a large randomized-greedy candidate pool and then selects from it by **lazy-greedy
submodular maximization** — Minoux's lazy evaluation, stochastic greedy for large pools, and a
CVaR-upside objective that scores a lineup by how much it improves the portfolio's *tail*, not
its mean.

## What's in it

- Randomized greedy construction with salary-repair backtracking, over an arbitrary roster
  specification — slots, position eligibility as bitmasks, salary cap and floor, and generic
  group constraints.
- Lazy-greedy submodular selection with a CVaR-upside objective, ownership/leverage discounting,
  and a diversity penalty.
- Sport presets (`slatekit.presets`) shipped as data, not hardcoded branches.
- Runtime AVX2 dispatch. Wheels are built portably; `slatekit.active_isa()` reports which path
  your machine took.

## Benchmarks

Measured against MILP formulations of the same problem (PuLP/CBC and OR-Tools), which live in
[`benchmarks/baselines/`](https://github.com/kfreisen/oss/tree/main/packages/slatekit/benchmarks/baselines) as real, tested, importable code — along with a
pure-Python transcription of the greedy algorithm that serves as the parity oracle.

Numbers and the hardware they were measured on: <https://kfreisen.github.io/oss/slatekit/benchmarks/>

## Install

```bash
pip install slatekit
```

Binary wheels are published for Linux (x86-64, aarch64), macOS (arm64, x86-64), and Windows
(x86-64). Installing from source requires a Rust toolchain:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## License

Apache-2.0.
