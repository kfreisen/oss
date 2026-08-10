"""Options-implied move, catalyst-date inference, and calibrated event probability.

Everything here is closed-form arithmetic over stdlib :mod:`math`. The absence of
dependencies is part of the contract: this is meant to drop into an environment
where adding scipy is not worth it, and to be readable end to end.

```python
from impliedmove import OptionsSurface, Expiry, implied_view

view = implied_view(surface, down_target=8.0, up_target=40.0)
view.hump_expiry  # the market-implied catalyst date
view.implied_move  # expected absolute move, as a fraction of spot
view.implied_probability  # or None, with view.caveats explaining why
```

The most valuable thing this computes is sometimes a refusal. A two-outcome implied
probability is arithmetic that always returns a number, including when the assumed
outcomes are inconsistent with what the options market is pricing — and in exactly
that case the number looks like enormous edge. See
:mod:`impliedmove.surface`.
"""

from __future__ import annotations

from impliedmove.blackscholes import bs_price, delta, implied_vol, norm_cdf
from impliedmove.bridge import Adjustment, Bridge, Factor
from impliedmove.calibration import (
    Bucket,
    Reliability,
    brier_score,
    directional_hit_rate,
    reliability,
)
from impliedmove.surface import Expiry, ImpliedView, OptionsSurface, implied_view

__all__ = [
    "Adjustment",
    "Bridge",
    "Bucket",
    "Expiry",
    "Factor",
    "ImpliedView",
    "OptionsSurface",
    "Reliability",
    "__version__",
    "brier_score",
    "bs_price",
    "delta",
    "directional_hit_rate",
    "implied_view",
    "implied_vol",
    "norm_cdf",
    "reliability",
]

__version__ = "0.0.1.dev0"
