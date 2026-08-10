import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from datetime import datetime, timedelta, timezone

    import marimo as mo
    import venuematch
    from venuematch import Record, resolve
    from venuematch.records import block_by, block_size_report, iso_week_blocker
    from venuematch.resolve import Overrides
    from venuematch.scoring import CompositeScorer

    return (
        CompositeScorer,
        Overrides,
        Record,
        block_by,
        block_size_report,
        datetime,
        iso_week_blocker,
        mo,
        resolve,
        timedelta,
        timezone,
        venuematch,
    )


@app.cell
def _(mo, venuematch):
    mo.md(
        f"""
        # venuematch — matching entities across sources

        Version `{venuematch.__version__}`.

        Two systems describe the same thing in their own words, with no shared
        identifier. You want the clusters.
        """
    )
    return


@app.cell
def _(Record, datetime, timedelta, timezone):
    base = datetime(2026, 9, 30, tzinfo=timezone.utc)

    records = [
        Record("k1", "kalshi", "Will Acme Corp report revenue above $5B in Q3?", base, 0.42),
        Record("p1", "poly", "Acme Corp Q3 revenue over $5B?", base - timedelta(days=1), 0.44),
        Record("k2", "kalshi", "Will Zenith Corp report revenue above $5B in Q3?", base, 0.39),
        Record("p2", "poly", "Zenith Corp Q3 revenue over $5B?", base, 0.40),
        Record("k3", "kalshi", "Will the Iowa primary be won by Smith?", base, 0.61),
        Record("p3", "poly", "Will the Hampshire primary be won by Smith?", base, 0.58),
    ]
    records
    return base, records


@app.cell
def _(mo):
    mo.md(
        """
        ## The hard case is the last pair

        `k3` and `p3` share every token but one. Any generic string metric rates
        them nearly identical — and merging them would be a serious error, because
        Iowa and New Hampshire are different elections.

        The distinguishing information is in the **proper nouns**, so the scorer
        subtracts for disagreement there.
        """
    )
    return


@app.cell
def _(CompositeScorer, mo, records):
    with_penalty = CompositeScorer()
    without_penalty = CompositeScorer(proper_noun_penalty=0.0)

    iowa, hampshire = records[4], records[5]
    a = with_penalty(iowa, hampshire)
    b = without_penalty(iowa, hampshire)

    mo.md(
        f"""
        | scorer | confidence | proper nouns differing |
        | --- | ---: | ---: |
        | with penalty (default) | {a.confidence:.3f} | {a.proper_noun_diff} |
        | penalty disabled | {b.confidence:.3f} | {b.proper_noun_diff} |

        At the default threshold of 0.65, one of these merges two distinct
        elections and the other does not.
        """
    )
    return a, b, hampshire, iowa, with_penalty, without_penalty


@app.cell
def _(mo):
    mo.md("## Resolving")
    return


@app.cell
def _(mo, records, resolve):
    result = resolve(records)

    lines = "\n".join(
        f"- **{cid}** → {', '.join(members)}" for cid, members in result.clusters.items()
    )
    mo.md(
        f"""
{lines}

`{result.comparisons}` comparisons made. Iowa and New Hampshire stayed apart.
        """
    )
    return lines, result


@app.cell
def _(mo, result):
    near = "\n".join(
        f"| {p.left} | {p.right} | {p.score.confidence:.3f} | {p.score.proper_noun_diff} |"
        for p in result.near_misses
    )
    mo.md(
        f"""
        ## Near misses are where tuning starts

        | left | right | confidence | proper-noun diff |
        | --- | --- | ---: | ---: |
        {near if near else "| — | — | — | — |"}

        These are the decisions the threshold is actually making. Tuning against a
        single summary number is guesswork; tuning against this list is not.
        """
    )
    return (near,)


@app.cell
def _(mo):
    mo.md(
        """
        ## What blocking costs

        Blocking is what keeps this from being quadratic — but it buys speed by
        declining to compare some pairs, and a pair it never compares is lost
        **silently**.

        The default key is the ISO week of the timestamp, so two sources that place
        the same event either side of a week boundary will never be compared.
        Measured on a synthetic corpus, recall against exhaustive comparison falls
        roughly linearly with how far the two sources' timestamps disagree:

        | timestamp drift | recall |
        | ---: | ---: |
        | 0 days | 100% |
        | 1 day | 88% |
        | 2 days | 76% |
        | 3 days | 64% |
        | 5 days | 32% |

        Which is why the blocker is a parameter, why the benchmark reports recall
        next to speed, and why forced overrides exist.
        """
    )
    return


@app.cell
def _(block_by, block_size_report, iso_week_blocker, mo, records):
    mo.md(f"`{block_size_report(block_by(records, iso_week_blocker()))}`")
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Overrides beat scores

        Every real deduplication system accumulates a list of pairs a human has
        ruled on. Without somewhere to put them, the same wrong merge gets
        re-litigated every time a threshold moves.
        """
    )
    return


@app.cell
def _(Overrides, mo, records, resolve):
    pinned_apart = resolve(records, overrides=Overrides(blocked=frozenset({("k1", "p1")})))
    pinned_together = resolve(records, overrides=Overrides(forced=frozenset({("k3", "p3")})))

    mo.md(
        f"""
        - blocking `k1`–`p1`: {len(pinned_together.clusters) - len(pinned_apart.clusters) + len(pinned_apart.clusters)} clusters become **{len(pinned_apart.clusters)}**
        - forcing `k3`–`p3`: **{len(pinned_together.clusters)}** clusters, including the one no score would have made

        A forced pair is applied even if blocking never compared it, which is the
        repair for a blocking key that separates a true match.
        """
    )
    return pinned_apart, pinned_together


if __name__ == "__main__":
    app.run()
