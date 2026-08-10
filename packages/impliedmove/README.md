# impliedmove

Read the market's opinion of a binary event out of an option chain — the expected move, the
implied date, and the implied probability — then check whether your own estimate was any good.

**No dependencies.** Everything is stdlib `math`. That is a deliberate constraint, not an
oversight.

> **Status: pre-release.** `0.0.1.dev0` is a scaffold that reserves the name and exercises the
> build and release pipeline. The library lands in `0.1.0`.

## What it does

**Black-Scholes and implied volatility**, priced and inverted from scratch. Vendor IV on
illiquid chains is frequently stale, wrong, or absent, and the whole analysis is downstream of
it — so this computes its own rather than trusting a feed field.

**Implied event date.** When a binary catalyst is expected, ATM implied volatility peaks at the
expiry that first covers it. Taking the argmax of ATM IV across expiries recovers the market's
view of *when*, without anyone publishing a date. Near-dated expiries are excluded because
gamma noise produces a spurious peak there.

**Implied move and implied probability.** The ATM straddle gives the expected absolute move; a
two-outcome model turns an up case and a down case into the probability the market is assigning.

**A self-invalidation check.** Two-outcome implied probability is only meaningful when the
assumed outcomes are consistent with what options are actually pricing. When they are not, the
formula still returns a number, and that number manufactures edge out of nothing. This library
cross-checks the assumed downside against the options-implied move and refuses the estimate when
they disagree. Getting a *refusal* out of this calculation is the feature.

**A bounded-factor probability bridge.** Turn a base rate into an adjusted estimate by applying
named factors, each with a hard cap, under a per-category cap and a total-adjustment cap. The
caps are enforced in code. This exists because the natural way to use a language model here — ask
it for a probability — produces a number with no audit trail and no bound on how far it can move.
Letting it propose *factors* while code enforces the limits keeps the output explainable and
keeps the worst case bounded.

**Calibration.** Brier score, reliability curve, and edge-direction hit rate over a log of
resolved outcomes, because an uncalibrated probability is just a number with a percent sign.

## Not a timing benchmark

The other packages in this repo publish speedups. This one publishes **accuracy**: Brier score
and reliability of the bounded-factor bridge against the unadjusted prior, on the same resolved
outcomes. Pretending it had a performance story would be dishonest — it computes closed-form
arithmetic in microseconds.

## Install

```bash
pip install impliedmove
```

Try it in the browser — the examples run as interactive notebooks with no install:
<https://kfreisen.github.io/oss/impliedmove/examples/>

## License

Apache-2.0.
