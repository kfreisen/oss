import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from datetime import date

    import marimo as mo
    import washbook
    from washbook import Trade, apply_wash_sales
    from washbook.defer import WashSaleConfig

    return WashSaleConfig, Trade, apply_wash_sales, date, mo, washbook


@app.cell
def _(mo, washbook):
    mo.md(
        f"""
        # washbook — what the wash-sale rule costs

        Version `{washbook.__version__}`.

        Sell at a loss, buy back within the window, and the loss is **disallowed** —
        not forgiven. It rolls into the replacement's cost basis and the old holding
        period tacks onto the new one. The deduction is deferred, not destroyed.

        Which matters, because a backtest that reports pre-tax returns on a
        high-turnover taxable strategy is reporting a number nobody can earn.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## A chain

        Three round trips in one symbol. The first two lose and are bought back
        quickly; the third finally runs.
        """
    )
    return


@app.cell
def _(Trade, date):
    trades = [
        Trade("AAPL", date(2026, 1, 5), date(2026, 2, 1), proceeds=900.0, basis=1000.0),
        Trade("AAPL", date(2026, 2, 6), date(2026, 3, 1), proceeds=850.0, basis=1000.0),
        Trade("AAPL", date(2026, 3, 4), date(2026, 12, 1), proceeds=1600.0, basis=1000.0),
    ]
    trades
    return (trades,)


@app.cell
def _(apply_wash_sales, mo, trades):
    result = apply_wash_sales(trades)

    rows = "\n".join(
        f"| {t.trade.closed} | {t.trade.pnl:+.0f} | {t.adjusted_basis:.0f} | "
        f"{'yes' if t.is_wash_sale else 'no'} | {t.allowed_loss:+.0f} | {t.gain:+.0f} |"
        for t in result.trades
    )

    mo.md(
        f"""
| closed | economic P&L | adjusted basis | wash sale | deductible loss | taxable gain |
| --- | ---: | ---: | :---: | ---: | ---: |
{rows}

**{result.wash_sale_count} wash sales.** The first two losses are disallowed and
their basis rolls forward, so the final leg is taxed on
`{result.total_gain:+.0f}` rather than on its raw `+600`.

Economic result: `{result.total_economic_pnl:+.0f}`. Nothing was created or
destroyed — only moved.
        """
    )
    return result, rows


@app.cell
def _(mo):
    mo.md(
        """
        ## The invariant

        Wash-sale treatment changes *when* a loss is deducted, never whether it
        happened. So deductible losses, plus gains, plus whatever is still deferred,
        must equal the economic result.

        That single property is worth more than any number of hand-computed
        examples — of the four implementations this package was merged from, the one
        that dropped washed rows entirely would have failed it immediately.
        """
    )
    return


@app.cell
def _(mo, result):
    result.check_invariant()
    mo.md("`result.check_invariant()` passes — the books balance.")
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## The symmetric window

        The rule is 30 days **before** through 30 days **after** the loss sale. Buy
        first, sell the older lot at a loss second, and it is still a wash sale.

        Every implementation this package was extracted from looked only forward,
        which understates wash sales for any strategy that scales into a position.
        Both behaviors are available; the statute is the default.
        """
    )
    return


@app.cell
def _(Trade, WashSaleConfig, apply_wash_sales, date, mo):
    scaled_in = [
        Trade("MSFT", date(2025, 11, 1), date(2026, 2, 1), proceeds=900.0, basis=1000.0),
        # Bought 12 days BEFORE the loss above was realized.
        Trade("MSFT", date(2026, 1, 20), date(2026, 9, 1), proceeds=1300.0, basis=1000.0),
    ]

    statutory = apply_wash_sales(scaled_in).wash_sale_count
    forward_only = apply_wash_sales(
        scaled_in, config=WashSaleConfig(symmetric=False)
    ).wash_sale_count

    mo.md(
        f"""
        - symmetric window (the statute): **{statutory}** wash sale
        - forward-only approximation: **{forward_only}** wash sales

        The forward-only reading misses it entirely.
        """
    )
    return forward_only, scaled_in, statutory


@app.cell
def _(mo):
    mo.md(
        """
        ## Positions still open at the end

        §1091 triggers on *acquisition*, not on the replacement's eventual sale. A
        position still open when the ledger ends is a replacement, and the loss it
        washed was economically suffered and never deducted.

        Omitting those overstates deductions for any strategy still holding at the
        end — which is most of them.
        """
    )
    return


@app.cell
def _(Trade, apply_wash_sales, date, mo):
    still_holding = apply_wash_sales(
        [Trade("NVDA", date(2026, 1, 1), date(2026, 2, 1), proceeds=900.0, basis=1000.0)],
        open_positions=[("NVDA", date(2026, 2, 10))],
    )

    mo.md(
        f"""
        Wash sales: **{still_holding.wash_sale_count}**.
        Deducted: `{still_holding.total_allowed_loss:+.0f}`.
        Still deferred at ledger end: `{still_holding.deferred_at_end:.0f}` — real
        money lost, no tax credit received inside the window studied.
        """
    )
    return (still_holding,)


if __name__ == "__main__":
    app.run()
