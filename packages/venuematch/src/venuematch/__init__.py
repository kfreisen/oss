"""Blocked fuzzy entity resolution.

Group records that describe the same real-world entity across sources, without
comparing every pair to every other pair.

```python
from venuematch import Record, resolve

result = resolve(records, threshold=0.7)
result.clusters  # cluster id -> member ids
result.near_misses  # what the threshold nearly caught — where tuning starts
```

The two things this gets right that generic string matching does not: it penalizes
disagreement on **proper nouns**, which is where meaning lives ("Iowa primary" is
not "New Hampshire primary" however many tokens they share); and it treats human
overrides as first-class, so a correction survives the next threshold change.
"""

from __future__ import annotations

from venuematch.records import Record, block_by, iso_week_blocker
from venuematch.resolve import Overrides, ResolutionResult, ScoredPair, resolve
from venuematch.scoring import CompositeScorer, PairScore

__all__ = [
    "CompositeScorer",
    "Overrides",
    "PairScore",
    "Record",
    "ResolutionResult",
    "ScoredPair",
    "__version__",
    "block_by",
    "iso_week_blocker",
    "resolve",
]

__version__ = "0.0.1.dev0"
