# washbook

After-tax backtest accounting: a vectorized wash-sale blackout and an IRS §1091 deferral ledger.

> **Status: pre-release.** `0.0.1.dev0` is a scaffold that reserves the name and exercises the
> build and release pipeline. The engine lands in `0.1.0`.

## Why

A backtest that reports pre-tax returns on a strategy that trades a taxable account is reporting
a number nobody can earn. Wash sales are the specific reason: realize a loss, buy back within 30
days, and the deduction is disallowed — deferred into the replacement lot's basis, with the
holding period tacked on. A high-turnover strategy that looks good gross can spend the year
generating losses it is never allowed to deduct.

There are two honest ways to model that, and they answer different questions:

**Cooldown** — refuse to re-enter within the window, so no wash sale ever occurs. This measures
what the strategy earns if you trade around the rule. It changes the equity curve, and the cost
shows up as suppressed entries (`wash_rejections`) rather than as tax. This is the
performance-critical path: it is a two-pass computation expressed as a single lazy query, with
the blackout applied by an as-of join partitioned across the whole symbol panel.

**Defer** — trade freely and account for §1091 as a taxable account actually experiences it.
Disallowed losses roll into the replacement lot's basis, holding periods tack per §1223(3),
chains roll through consecutive replacements, and a chain still open at the end of the ledger
surfaces as deferred-and-never-recovered. The economic equity curve is unchanged; only the
after-tax figures move.

`washbook` implements both, and treats the difference between them as a result worth reporting.

## Scope, stated honestly

Tax software this is not. It models §1091 mechanics for a backtest, and it is explicit about
where it stops — the boundaries are documented per function rather than left for you to discover.

## Benchmarks

The cooldown path is measured against the serial per-symbol implementation it replaced, which
lives in [`benchmarks/baselines/`](https://github.com/kfreisen/oss/tree/main/packages/washbook/benchmarks/baselines) as real, tested code, and is asserted
to produce identical output by `tests/test_parity.py`.

Numbers and hardware: <https://kfreisen.github.io/oss/washbook/benchmarks/>

## Install

```bash
pip install washbook
```

## License

Apache-2.0.
