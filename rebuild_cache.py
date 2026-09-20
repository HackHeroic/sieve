"""Rebuild the demo cache on the CORRECTED PJM series."""
import pickle, time
import numpy as np, pandas as pd, engine
from engine import Problem
from features import build, lattice

engine.BLOCK = 504
t0 = time.time()
X, y, meta = build("PJM", years=range(2019, 2027), q=.90, fc_shift=1)
r, lab = lattice(X, max_arity=3, min_support=1, verbose=False)
months = np.asarray(pd.to_datetime(X.index).month)
print(f"{meta['n']:,} hours, {meta['n1']:,} positive, {r.shape[0]:,} rules  {time.time()-t0:.0f}s")

np.save("cache_rules2.npy", r)
pickle.dump(lab, open("cache_labels2.pkl", "wb"))
np.save("cache_y.npy", y)
X.to_parquet("cache_X.parquet")
import json; json.dump(dict(ba="PJM", n=meta['n'], n1=meta['n1'], base=meta['base'],
    start=meta['start'], end=meta['end'], fc_shift=1, block=504),
    open("cache_meta.json","w"), indent=1)

p = Problem(r, lab, y, months=months, alpha=0.05)
t0 = time.time(); nb = p.null(1000, kind="block", seed=1); print(f"block null {time.time()-t0:.0f}s")
t0 = time.time(); nr = p.null(1000, kind="row",   seed=1); print(f"row null   {time.time()-t0:.0f}s")
np.savez("results2.npz", nb=nb, nr=nr, lp=p.lp, sup=p.sup, a=p.a)
sb, sr = len(p.significant(nb)), len(p.significant(nr))
print(f"\n  significant: block {sb:,}   row {sr:,}   inflation {sr/sb:.2f}x")
ch, _ = p.cover(nb, top=3, min_support=800, min_new=40)
for c in ch: print(f"    lift {c['lift']:.2f}  {c['rule']}")
