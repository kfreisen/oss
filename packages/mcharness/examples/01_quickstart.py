import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import mcharness
    import numpy as np
    from mcharness import EntityCache, run
    from mcharness.seeding import spawn_generators
    from mcharness.testing import Coin, Draw

    return Coin, Draw, EntityCache, mcharness, mo, np, run, spawn_generators


@app.cell
def _(mcharness, mo):
    mo.md(
        f"""
        # mcharness — reproducible batched Monte Carlo

        Version `{mcharness.__version__}`.

        The harness owns batching, seeding, dispatch and streaming collection. You
        supply a simulation. Nothing about a domain appears in it.

        This notebook runs the serial backend so it works anywhere; the Ray backend
        is the same call with `backend="ray"`, and is required to produce identical
        results.
        """
    )
    return


@app.cell
def _(Coin, mo, run):
    result = run(Coin(0.3), 20_000, batch_size=1000, run_seed=7)
    mean = sum(result.value) / len(result.value)
    mo.md(
        f"""
        `{result.n_trials:,}` trials in `{result.n_batches}` batches →
        mean **{mean:.4f}** against a true 0.30.
        """
    )
    return mean, result


@app.cell
def _(mo):
    mo.md(
        """
        ## Seeding is the part that quietly goes wrong

        The obvious approach is `default_rng(run_seed + batch_index)`. Adjacent
        integer seeds are **not** guaranteed to produce independent streams, and
        correlated streams show up as variance that is quietly too low — the
        estimate looks more precise than it is, which is the worst way for a bug to
        present.

        `SeedSequence.spawn` is the supported primitive, and it is what this uses.
        """
    )
    return


@app.cell
def _(mo, np, spawn_generators):
    a, b = spawn_generators(run_seed=5, n_batches=2)
    correlation = float(np.corrcoef(a.random(5000), b.random(5000))[0, 1])
    mo.md(f"Correlation between two spawned batch streams: `{correlation:+.4f}`.")
    return a, b, correlation


@app.cell
def _(Draw, mo, run):
    same = run(Draw(), 500, batch_size=100, run_seed=11).value
    again = run(Draw(), 500, batch_size=100, run_seed=11).value
    other_batching = run(Draw(), 500, batch_size=250, run_seed=11).value

    mo.md(
        f"""
        - same seed, same batch size → identical: **{same == again}**
        - same seed, *different* batch size → identical: **{same == other_batching}**

        The second one is deliberate and documented. Streams are derived per batch,
        so batch size is part of the reproducibility contract, not a free tuning
        knob. Reproducing a run means matching the seed **and** the batch size.
        """
    )
    return again, other_batching, same


@app.cell
def _(mo):
    mo.md(
        """
        ## What the benchmarks actually found

        Two results, and neither is the one this package was expected to show.
        Measured at 100,000 trials:

        | implementation | median |
        | --- | ---: |
        | serial, DataFrame `.loc` per trial | 1167 ms |
        | serial, cached lookup | 40 ms |
        | serial, dict lookup | 43 ms |
        | Ray, cached lookup | 197 ms |
        | Ray, dict lookup | 220 ms |

        **Almost the entire win is DataFrame → dict, about 29×.** The cache adds
        ~6%. If a study is slow and does a `.loc` per trial, that is the whole
        problem and parallelism does not address it.

        **Ray is 5× slower than the serial loop here** — these trials are one draw
        each, so dispatch costs far more than the work.

        Give a trial real work to do and that reverses: at 400 inner operations per
        trial, Ray runs in **105 ms against 406 ms serial, 3.9×**. The rule is not
        "Ray is slow"; it is that per-trial work has to clear the dispatch cost, and
        one random draw does not come close.

        Fix the lookup first, measure, and reach for a cluster once a trial is heavy
        enough to earn one.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## And a smaller surprise

        Indexing a NumPy array one element at a time is **slower** than indexing a
        dict, because each access boxes a NumPy scalar. A plain Python list beats
        both:

        | container | 2000 scalar lookups + a draw |
        | --- | ---: |
        | dict of dicts | 0.87 ms |
        | numpy array | 0.94 ms |
        | python list | 0.77 ms |

        So "hoist the lookups into arrays" is only good advice when the consumer is
        vectorized. `EntityCache` stores arrays — they ship cheaply to workers and
        suit a vectorized simulation — and hands out lists via `field_list()` for
        scalar access in a loop.
        """
    )
    return


@app.cell
def _(EntityCache, mo):
    records = [{"player_id": f"p{i}", "mean": 10.0 + i, "stddev": 1.0 + i % 3} for i in range(5)]
    cache = EntityCache.from_records(records, key="player_id", fields=("mean", "stddev"))

    mo.md(
        f"""
        `{len(cache)}` entities, fields `{cache.fields}`.

        `cache.index_of("p3")` → `{cache.index_of("p3")}` — call this once, outside
        the loop. Calling it per trial puts the dict lookup back.
        """
    )
    return cache, records


if __name__ == "__main__":
    app.run()
