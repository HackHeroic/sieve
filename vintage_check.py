"""
Is `err_lag24` actually available at forecast-filing time?

The issue
---------
EIA-930's day-ahead demand forecast is filed as ONE D-1 vintage covering all 24
hours of day D.  Our code treats every hour as having a uniform 24h lead via
`.shift(24)`.  It does not.  If the vintage is filed at hour T on D-1, then for
hour t on D the true lead is (t + 24 - T) hours, and a realised quantity at
lag 24 -- hour t on D-1 -- had only occurred by filing time if t < T.

For a mid-morning filing that is false for more than half the day.  So
`err_lag24` and the other realised lag-24 features are partly look-ahead with
respect to the filing instant, even though they are not look-ahead with respect
to hour t.

The test
--------
Rebuild the lattice using ONLY predicates that are unambiguously available:
the forecast itself and shapes derived from it (a forecast is known when filed,
whatever its vintage), calendar terms, and realised quantities at lag >= 48h.
Then ask whether the finding survives.
"""
import json
import numpy as np
import pandas as pd

from core import pack
from engine import Problem
from features import lattice

# realised quantities at lag 24: available at hour t, NOT necessarily at filing
SUSPECT = {"dem_lag24", "err_lag24", "interchange_lag24", "netgen_lag24",
           "dem_chg_24_48", "err_roll7d",
           "share_coal_lag24", "share_natural_lag24", "share_nuclear_lag24",
           "share_solar_lag24", "share_wind_lag24"}

X = pd.read_parquet("cache_X.parquet")
y = np.load("cache_y.npy")
months = np.asarray(pd.to_datetime(X.index).month)


def base(c):
    return c.split(">")[0].split("=")[0]


keep = [c for c in X.columns if base(c) not in SUSPECT]
print(f"full lattice predicates   : {X.shape[1]}")
print(f"strict (lag >= 48h only)  : {len(keep)}   (dropped {X.shape[1]-len(keep)})")
print(f"  kept bases: {sorted({base(c) for c in keep})}\n")

for name, cols in (("FULL (as shipped)", list(X.columns)),
                   ("STRICT (lag >= 48h)", keep)):
    r, lab = lattice(X[cols], max_arity=3, min_support=1, verbose=False)
    p = Problem(r, lab, y, months=months, alpha=0.05)
    nul = p.null(400, kind="block", seed=1)
    sig = p.significant(nul)
    ch, _ = p.cover(nul, top=3, min_support=800, min_new=40)
    print(f"{name}")
    print(f"  rules {r.shape[0]:,}   significant {len(sig):,}   cutoff {p.cutoff(nul):.1f}")
    for c in ch:
        print(f"    lift {c['lift']:5.2f}  sup {c['support']:6,}  {c['rule']}")
    if name.startswith("STRICT"):
        strict_lift = ch[0]["lift"] if ch else None
        json.dump(dict(strict_predicates=len(keep), full_predicates=int(X.shape[1]),
                       strict_rules=int(r.shape[0]), strict_significant=int(len(sig)),
                       strict_top_lift=strict_lift,
                       strict_top_rule=ch[0]["rule"] if ch else None),
                  open("vintage_check.json", "w"), indent=1)
    print()
