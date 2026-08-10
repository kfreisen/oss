import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import impliedmove
    import marimo as mo
    from impliedmove import (
        Bridge,
        Expiry,
        Factor,
        OptionsSurface,
        bs_price,
        implied_view,
        implied_vol,
    )

    return (
        Bridge,
        Expiry,
        Factor,
        OptionsSurface,
        bs_price,
        implied_view,
        implied_vol,
        impliedmove,
        mo,
    )


@app.cell
def _(impliedmove, mo):
    mo.md(
        f"""
        # impliedmove

        Version `{impliedmove.__version__}`. **No dependencies** — this whole
        notebook is running in your browser.

        Read the market's opinion of a binary event out of an option chain: the
        expected move, the implied date, and the implied probability. Then check
        whether your own estimate was any good.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Its own implied volatility

        Vendor IV on an illiquid chain is frequently stale, inconsistent between
        strikes, or absent, and everything downstream inherits that. So this
        computes its own, by bisection.

        Bisection rather than Newton-Raphson on purpose: Newton is faster and
        diverges on exactly the inputs this exists to handle — deep out-of-the-money
        contracts where vega is near zero, and crossed quotes.
        """
    )
    return


@app.cell
def _(bs_price, implied_vol, mo):
    price = bs_price(100, 105, 0.5, 0.42, call=True)
    recovered = implied_vol(price, 100, 105, 0.5, call=True)
    mo.md(f"Priced at 42% vol → `{price:.4f}`, inverted back to `{recovered:.6f}`.")
    return price, recovered


@app.cell
def _(implied_vol, mo):
    # A quote below intrinsic value: 110 spot, 100 strike, so the call is worth at
    # least 10 and someone is showing 5.
    bad = implied_vol(5.0, 110, 100, 0.5, call=True)
    mo.md(
        f"""
        A quote below intrinsic returns `{bad}`, not a number.

        That refusal is the feature. A price below intrinsic is a bad quote, not a
        low-volatility signal, and substituting the nearest legal value turns a data
        problem into a confident trade.
        """
    )
    return (bad,)


@app.cell
def _(mo):
    mo.md(
        """
        ## The implied catalyst date

        When the market expects a binary event, at-the-money implied volatility
        peaks at the first expiry covering it. The argmax recovers the market's view
        of *when*, without anyone publishing a date.

        Expiries under 10 days are excluded from the hunt: front-week IV is inflated
        by near-dated gamma and routinely wins the argmax spuriously, pointing at a
        thin contract with nothing to do with the event.
        """
    )
    return


@app.cell
def _(Expiry, OptionsSurface):
    surface = OptionsSurface(
        symbol="BIOX",
        spot=20.0,
        expiries=(
            Expiry("2026-07-17", dte=5, atm_iv=2.10, atm_straddle=2.0),
            Expiry("2026-08-21", dte=40, atm_iv=0.95, atm_straddle=5.5),
            Expiry("2026-10-16", dte=96, atm_iv=1.55, atm_straddle=9.0),
            Expiry("2027-01-15", dte=187, atm_iv=1.05, atm_straddle=10.5),
        ),
    )
    surface
    return (surface,)


@app.cell
def _(implied_view, mo, surface):
    view = implied_view(surface, down_target=8.0, up_target=40.0)
    mo.md(
        f"""
        - implied catalyst expiry: **{view.hump_expiry}** (peak ATM IV {view.peak_iv:.0%})
        - implied move by then: **{view.implied_move:.0%}** of spot
        - two-outcome implied probability: **{view.implied_probability:.0%}**
        - trustworthy: **{view.trustworthy}**

        The 5-DTE expiry has the highest IV on the board and was correctly ignored.
        """
    )
    return (view,)


@app.cell
def _(mo):
    mo.md(
        """
        ## The refusal that matters

        `(spot - down) / (up - down)` always returns a number, including when the
        assumed outcomes are inconsistent with what options are pricing. Set the
        downside just below spot and the implied probability collapses toward zero —
        "the market says this is hopeless" — which looks like enormous edge and is
        really a bad input.

        So the downside case is cross-checked against the options-implied move.
        """
    )
    return


@app.cell
def _(implied_view, mo, surface):
    suspicious = implied_view(surface, down_target=19.0, up_target=40.0)
    caveats = "\n".join(f"> {c}" for c in suspicious.caveats)
    mo.md(
        f"""
        Downside set to 19 against a spot of 20:

        implied probability **{suspicious.implied_probability:.1%}**, trustworthy
        **{suspicious.trustworthy}**

        {caveats}
        """
    )
    return caveats, suspicious


@app.cell
def _(implied_view, mo, surface):
    outside = implied_view(surface, down_target=25.0, up_target=40.0)
    mo.md(
        f"""
        And with spot outside the bracket entirely, the answer is
        `{outside.implied_probability}` — never a clamped 0.0.

        > {outside.caveats[0]}

        Clamping is how a stale share count becomes a top position: a floor at or
        above spot yields probability zero, i.e. maximum apparent edge, out of a
        data bug.
        """
    )
    return (outside,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Bounded adjustment

        You want a model's judgement in a probability estimate, and you cannot let
        it produce the probability — you would get a number with no audit trail and
        no bound on how far it can move.

        So it proposes **named factors**, and code enforces the limits.
        """
    )
    return


@app.cell
def _(Bridge, Factor, mo):
    factors = [
        Factor("phase 2 hit its primary endpoint", 0.09, category="clinical"),
        Factor("open-label extension looked clean", 0.06, category="clinical"),
        Factor("management has never run a phase 3", -0.04, category="execution"),
        Factor("a model felt strongly about this", 0.80, category="vibes"),
    ]

    result = Bridge().apply(0.35, factors)
    rows = "\n".join(f"| {name} | {delta:+.3f} |" for name, delta in result.applied)

    mo.md(
        f"""
| factor | applied |
| --- | ---: |
{rows}

Base rate `{result.base_rate:.2f}` → **`{result.probability:.3f}`**, net movement
`{result.total_adjustment:+.3f}`.

Capped: `{", ".join(result.capped)}`.

The 0.80 factor was cut to the 0.10 per-factor limit, the clinical pair was held
to its category cap, and the total was capped again. The worst case is bounded
even when the reasoning is nonsense — which is the entire point, since no code can
check whether a factor is *sensible*.
        """
    )
    return factors, result, rows


@app.cell
def _(mo):
    mo.md(
        """
        ## Does any of it help?

        Measured over 4000 synthetic events, by Brier score (lower is better):

        | bad factors | prior | uncapped | bounded |
        | ---: | ---: | ---: | ---: |
        | 0% | 0.229 | **0.202** | 0.206 |
        | 17% | 0.229 | 0.260 | **0.213** |
        | 50% | **0.229** | 0.396 | 0.247 |

        At 17% the bridge wins and trusting factors outright is *worse than doing
        nothing*. At 0% the caps cost a little, holding back an adjustment that was
        right. At 50% the bridge still beats uncapped by a mile and is **worse than
        not adjusting at all**.

        Capping bounds the damage. It does not turn bad factors into good ones.
        """
    )
    return


if __name__ == "__main__":
    app.run()
