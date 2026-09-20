"""
120-null FWER calibration on the FIXED block permutation.  Run on the pod.

Why this needs re-running
------------------------
The 0.050 figure everywhere in the docs was measured BEFORE the ragged-tail
fix in core.block_permutation.  That fix changed which permutations get drawn
(only 3 of 20 draws previously kept every block inside its own month stratum;
now 20 of 20), so the headline calibration number is currently unverified on
the code we actually ship.  This re-measures it.

Design notes
------------
* The lattice depends only on X, so it is built ONCE and reused across the 120
  replicates.  reverify.py rebuilt it every call, which was pure waste.
* A "true null" is the real outcome circularly shifted away from its true
  alignment: the autocorrelation stays real, the relationship is destroyed.
* Both nulls run on the same permutations via backends.py.
"""
import json, time
import numpy as np
import pandas as pd

import backends
from core import Sieve, block_permutation, row_shuffle, pack, popcount
from fastp import logfac_table, logp_all, minp_scan
from features import build, lattice

NREP, J, BLOCK, ALPHA = 120, 300, 168, 0.05

print("building PJM features ...", flush=True)
t0 = time.time()
X, y, meta = build("PJM", years=range(2019, 2027), q=.90)
print(f"  {len(y):,} hours, {y.sum():,} positive ({y.mean():.1%})  {time.time()-t0:.0f}s", flush=True)

print("building lattice ...", flush=True)
t0 = time.time()
rules, labels = lattice(X, max_arity=3, min_support=1, verbose=False)
print(f"  {rules.shape[0]:,} rules  ({rules.nbytes/1e6:.0f} MB)  {time.time()-t0:.0f}s", flush=True)

months = np.asarray(pd.to_datetime(X.index).month)
n = len(y)
L = logfac_table(n)
bk = backends.make("auto", rules)
print(f"  backend: {bk.label}  batch {bk.batch}", flush=True)

sup = popcount(rules).astype(np.int64)


def null(yv, kind, seed):
    permuter = block_permutation if kind == "block" else row_shuffle
    strata = months if kind == "block" else None
    rng = np.random.default_rng(seed)
    out = np.empty(J); d = 0
    while d < J:
        B = min(bk.batch, J - d)
        idxs = [permuter(n, block=BLOCK, rng=rng, strata=strata) for _ in range(B)]
        C = bk.counts(yv, idxs)
        for k in range(B):
            out[d+k], _ = minp_scan(C[k], sup, n, int(yv.sum()), L, 0.0)
        d += B
    return out


hb = hr = 0
t0 = time.time()
for rep in range(NREP):
    yn = np.roll(y, int(n * (0.15 + 0.7 * rep / NREP)))
    n1 = int(yn.sum())
    a = popcount(rules & pack(yn)[None, :]).astype(np.int64)
    obs = logp_all(a, sup, n, n1, L).min()
    nb = null(yn, "block", 100 + rep)
    nr = null(yn, "row", 100 + rep)
    hb += (np.mean(nb <= obs) <= ALPHA)
    hr += (np.mean(nr <= obs) <= ALPHA)
    if (rep + 1) % 10 == 0:
        el = time.time() - t0
        print(f"  rep {rep+1:3d}/{NREP}   block {hb/(rep+1):.3f}   row {hr/(rep+1):.3f}"
              f"   {el:.0f}s  eta {el/(rep+1)*(NREP-rep-1):.0f}s", flush=True)

res = dict(block=hb/NREP, row=hr/NREP, n=NREP, J=J, rules=int(rules.shape[0]),
           hours=int(n), fixed_block_permutation=True)
print("\n=== CALIBRATION, FIXED CODE ===")
print(f"  circular block permutation  FPR = {res['block']:.3f}   (target {ALPHA})")
print(f"  naive row shuffle           FPR = {res['row']:.3f}")
json.dump(res, open("calib_fixed.json", "w"), indent=1)
print("\nwrote calib_fixed.json")
