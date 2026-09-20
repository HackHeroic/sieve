"""
How much does the answer depend on the null specification?

The attack this answers
-----------------------
PJM's day-ahead forecast improved over 2019-2024, and slow predicates
(err_roll7d, err_lag*) carry that level shift.  Our null holds calendar MONTH
fixed but not YEAR, so 81% of rows change year under a stratified draw.  A
sceptic can reasonably ask whether we are detecting a year effect.

The honest response is not a single number, it is a range across defensible null
specifications, volunteered rather than extracted.
"""
import json, time
import numpy as np
import pandas as pd

from engine import Problem, load_cached

J = 1000
rules, labels, y, X = load_cached()
idx = pd.to_datetime(X.index)
month = np.asarray(idx.month)
year = np.asarray(idx.year)

SPECS = {
    "month only (shipped)":   month,
    "year x month":           year * 100 + month,
    "year only":              year,
    "none (unstratified)":    None,
}

out = {}
for name, strata in SPECS.items():
    t = time.time()
    p = Problem(rules, labels, y, months=strata, alpha=0.05)
    nul = p.null(J, kind="block", seed=1)
    n = len(p.significant(nul))
    est, sd = p.count_ci(nul)
    nblocks = None if strata is None else len(np.unique(strata))
    out[name] = dict(significant=n, boot_mean=est, boot_sd=sd,
                     cutoff=p.cutoff(nul), strata_levels=nblocks)
    print(f"  {name:24s} {n:7,}  [{est:,} +/- {sd:,}]  cutoff {p.cutoff(nul):7.1f}"
          f"  strata {nblocks}  {time.time()-t:.0f}s", flush=True)

# the naive comparator, unchanged by any of this
p0 = Problem(rules, labels, y, months=month, alpha=0.05)
nr = p0.null(J, kind="row", seed=1)
out["row shuffle (naive)"] = dict(significant=len(p0.significant(nr)),
                                  cutoff=p0.cutoff(nr))
print(f"  {'row shuffle (naive)':24s} {len(p0.significant(nr)):7,}"
      f"  cutoff {p0.cutoff(nr):7.1f}")

vals = [v["significant"] for k, v in out.items() if k != "row shuffle (naive)"]
print(f"\n  RANGE across defensible block-permutation specs: "
      f"{min(vals):,} to {max(vals):,}")
json.dump(out, open("sensitivity.json", "w"), indent=1, default=float)
print("  wrote sensitivity.json")
