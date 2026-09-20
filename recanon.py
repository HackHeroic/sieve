"""Regenerate the canonical J=1000 PJM null on the fixed block permutation."""
import time, numpy as np, pandas as pd
from engine import Problem, load_cached

rules, labels, y, X = load_cached()
p = Problem(rules, labels, y, months=pd.to_datetime(X.index).month.values)
old = np.load('results2.npz')
t0 = time.time()
nb = p.null(1000, kind="block", seed=1)
print(f"block null  {time.time()-t0:.0f}s", flush=True)
t0 = time.time()
nr = p.null(1000, kind="row", seed=1)
print(f"row null    {time.time()-t0:.0f}s", flush=True)
sb, sr = len(p.significant(nb)), len(p.significant(nr))
osb = int((old['lp'] <= np.quantile(old['nb'], .05)).sum())
osr = int((old['lp'] <= np.quantile(old['nr'], .05)).sum())
print(f"\n  before fix: block {osb:,}  row {osr:,}")
print(f"  after  fix: block {sb:,}  row {sr:,}   ({100*(sb-osb)/osb:+.1f}% on block)")
np.savez('results2.npz', nb=nb, nr=nr, lp=p.lp, sup=p.sup, a=p.a)
print("\nwrote results2.npz")
