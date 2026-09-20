"""
Calibrate the block length PER GRID.

The four comparison grids were never affected by the PJM timestamp defect, so
their autocorrelation is whatever it always was -- and we never measured which
block length calibrates them.  Replication swung 53/57 to 36/57 between 168h and
504h purely on that unmeasured choice, which is not acceptable.

Each grid gets its own block length, chosen by its own measured false positive
rate on 120 true nulls.
"""
import json, time
import numpy as np, pandas as pd
from core import BlockPermuter, pack, popcount
from fastp import logfac_table, logp_all, minp_scan
from features import build, lattice
from engine import Problem

J, NREP, ALPHA = 250, 100, 0.05
SE = np.sqrt(ALPHA*(1-ALPHA)/NREP)
OUT = {}
for ba, shift in (("PJM",1), ("CISO",0), ("ERCO",0), ("MISO",0), ("NYIS",0)):
    X, y, meta = build(ba, years=range(2019,2027), q=.90, fc_shift=shift)
    r, lab = lattice(X, max_arity=3, min_support=1, verbose=False)
    months = np.asarray(pd.to_datetime(X.index).month)
    p = Problem(r, lab, y, months=months, device="auto")
    n, L, sup, bk = p.n, logfac_table(p.n), p.sup, p.backend
    # outcome autocorrelation, lag 1, for context
    yy = y.astype(float) - y.mean()
    a1 = float(np.correlate(yy[:-1], yy[1:])[0] / np.dot(yy, yy))
    res = {}
    for B in (168, 336, 504):
        bp = BlockPermuter(n, B, months); hb = 0
        for rep in range(NREP):
            yn = np.roll(y, int(n*(0.15+0.7*rep/NREP))); n1 = int(yn.sum())
            obs = logp_all(popcount(r & pack(yn)[None,:]).astype(np.int64),
                           sup, n, n1, L).min()
            rng = np.random.default_rng(100+rep); o = np.empty(J); d = 0
            while d < J:
                Bc = min(bk.batch, J-d)
                C = bk.counts(yn, [bp.draw(rng) for _ in range(Bc)])
                for k in range(Bc): o[d+k], _ = minp_scan(C[k], sup, n, n1, L, 0.0)
                d += Bc
            hb += (np.mean(o <= obs) <= ALPHA)
        res[B] = hb/NREP
    best = min(res, key=lambda b: abs(res[b]-ALPHA))
    OUT[ba] = dict(fpr=res, chosen=best, acf1=a1, hours=meta['n'])
    print(f"  {ba:5s} acf1={a1:.3f}  " + "  ".join(f"{B}h:{res[B]:.3f}" for B in res)
          + f"   -> {best}h", flush=True)
json.dump(OUT, open("calib_grids.json","w"), indent=1, default=float)
print("\nwrote calib_grids.json")
