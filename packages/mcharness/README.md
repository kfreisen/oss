# mcharness

Reproducible batched Monte Carlo on [Ray](https://ray.io): streaming collection, per-batch
seeding, and a cache for the per-entity lookups that otherwise dominate the inner loop.

> **Status: pre-release.** `0.0.1.dev0` is a scaffold that reserves the name and exercises the
> build and release pipeline. The harness lands in `0.1.0`.

## Why this exists

Parallelizing a Monte Carlo study is easy to do and easy to do wrong. The three things that go
wrong are always the same:

**Reproducibility dies first.** Seeding each worker from a global RNG makes results depend on
scheduling order, so a run cannot be reproduced and a regression cannot be bisected. `mcharness`
derives each batch's seed from `(run_seed, batch_index)`, so results depend on the batch layout
and nothing else — the same run seed gives the same answer at any level of parallelism, including
serial.

**Waiting for everything.** `ray.get` on the full list of futures holds every result in memory
and reports no progress until the slowest batch lands. `mcharness` collects with `ray.wait`, reducing
incrementally as batches complete, so memory is bounded by the reducer's state rather than by the
number of trials.

**Parallelizing the wrong thing.** Distributing a loop that spends most of its time doing
DataFrame lookups per iteration buys a fraction of what it should. The entity-stats cache hoists
those lookups out of the inner loop into a plain array indexed once per batch. That change is
usually worth more than the parallelism, which is why the benchmarks measure the two separately:
serial, parallel-uncached, and parallel-cached, so the speedup is attributed rather than
asserted.

## What you supply

A `Simulation` protocol — how to run one trial and how to combine results. The harness owns
batching, seeding, dispatch, collection, and progress. Nothing about the domain is baked in.

## Benchmarks

See [`benchmarks/baselines/`](https://github.com/kfreisen/oss/tree/main/packages/mcharness/benchmarks/baselines) for the serial and uncached-parallel
implementations, both kept working and both asserted to agree with the fast path.

Numbers and hardware: <https://kfreisen.github.io/oss/mcharness/benchmarks/>

## Install

```bash
pip install mcharness
```

## License

Apache-2.0.
