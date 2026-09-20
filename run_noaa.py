"""SIEVE on NOAA GHCN-Daily. A different agency, domain, cadence and schema."""
import time, numpy as np, pandas as pd
from adapter import load_csv, build_generic
from features import lattice
from engine import Problem

t0 = time.time()
df = load_csv("wx/USW00013739.csv", "DATE",
              cols=["PRCP", "SNOW", "SNWD", "TMAX", "TMIN", "AWND"])
df = df[df.index.year >= 1970]
print(f"NOAA Philadelphia Intl: {len(df):,} days, {df.index[0].date()} to {df.index[-1].date()}")
print(f"  columns: {list(df.columns)}")

X, y, meta = build_generic(df, target="TMAX", horizon=1, q=0.90,
                           lags=(1, 2, 7), stratify="month")
print(f"\n  outcome: top decile of |TMAX(t) - TMAX(t-1)| within calendar month")
print(f"  {meta['n']:,} days, {meta['n1']:,} positive ({100*meta['base']:.1f}%)")
print(f"  {X.shape[1]} predicates")

rules, labels = lattice(X, max_arity=3, min_support=1, verbose=False)
print(f"  {rules.shape[0]:,} rules  ({rules.nbytes/1e6:.0f} MB)   {time.time()-t0:.0f}s")

p = Problem(rules, labels, y, months=meta["months"], alpha=0.05)
print(f"  testable {int(p.s.testable.sum()):,}  (Tarone prunes "
      f"{100*(1-p.s.testable.mean()):.2f}%)")

# daily data: autocorrelation horizon is far shorter than hourly power demand
import engine
engine.BLOCK = 30                      # ~1 month blocks for daily data
J = 400
t = time.time(); nb = p.null(J, kind="block", seed=1); tb = time.time()-t
t = time.time(); nr = p.null(J, kind="row",   seed=1); tr = time.time()-t
sb, sr = len(p.significant(nb)), len(p.significant(nr))
est, sd = p.count_ci(nb)
print(f"\n  block permutation (30d blocks): {sb:,} significant  [{est:,} +/- {sd:,}]"
      f"   cutoff {p.cutoff(nb):.1f}   {1000*tb/J:.0f} ms/perm")
print(f"  naive row shuffle             : {sr:,} significant"
      f"   cutoff {p.cutoff(nr):.1f}")
print(f"  inflation: {sr/max(sb,1):.2f}x")
print(f"  observed min log p: {p.lp.min():.1f}")

print(f"\n  FINDINGS")
ch, cum = p.cover(nb, top=5, min_support=200, min_new=20)
print(f"  {'lift':>5} {'sup':>7} {'new':>6} {'log p':>8}  rule")
for c in ch:
    print(f"  {c['lift']:5.2f} {c['support']:7,} {c['new_hours']:6,} {c['logp']:8.1f}  {c['rule']}")
print(f"    -> {cum:,} of {meta['n1']:,} covered")
np.savez("noaa_results.npz", nb=nb, nr=nr, lp=p.lp, sup=p.sup, a=p.a)
print(f"\n  total {time.time()-t0:.0f}s")
