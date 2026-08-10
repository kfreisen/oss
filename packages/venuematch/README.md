# venuematch

Blocked fuzzy entity resolution: decide which records from different sources describe the same
real-world thing, without comparing every pair to every other pair.

> **Status: pre-release.** `0.0.1.dev0` is a scaffold that reserves the name and exercises the
> build and release pipeline. The matcher lands in `0.1.0`.

## The problem

Two systems describe the same event, product, or company in their own words, with no shared
identifier. You want the clusters.

Comparing all pairs is O(n²) and stops being viable in the low tens of thousands of records. The
standard fix is **blocking**: partition records into buckets that a true match could not span,
and only score within a bucket. The interesting part is not the speedup — it's that a careless
blocking key silently drops true matches, and nothing in the output tells you it happened.

`venuematch` is built around that failure mode:

- **Blocking is a callable you supply**, and the benchmark reports **recall against the
  all-pairs baseline**, not just wall-clock. A scheme that is 100× faster and loses 4% of matches
  is a regression, and the tooling says so.
- **Discriminative-token penalty.** Generic string similarity scores "Iowa primary winner" and
  "New Hampshire primary winner" as near-identical — they share almost every token. The scorer
  subtracts a penalty for the symmetric difference of *proper nouns*, which is where the meaning
  actually lives. This one term does more work than any similarity-metric choice.
- **Manual overrides are first-class.** Real deduplication always accumulates a list of pairs a
  human has ruled on. Forced-match and blacklist entries take precedence over scores, so a
  correction is permanent rather than something the next threshold tweak undoes.
- **Clusters via union-find** over pairs above threshold, with the transitive closure made
  explicit rather than accidental.

## Benchmarks

Against naive all-pairs, reporting both speedup and recall — see
[`benchmarks/baselines/`](https://github.com/kfreisen/oss/tree/main/packages/venuematch/benchmarks/baselines).

Numbers and hardware: <https://kfreisen.github.io/oss/venuematch/benchmarks/>

## Install

```bash
pip install venuematch
```

## License

Apache-2.0.
