"""
Which block length calibrates on the CORRECTED series?

Correcting the PJM forecast alignment raised the outcome's autocorrelation
(lag-1 0.654 -> 0.716, lag-6 0.082 -> 0.218), so the 168-hour blocks chosen for
the old series no longer preserve enough structure: measured FWER came out
0.100 against a nominal 0.05, p = 0.019.

Block length should be chosen by measurement, not by assertion.  This sweeps it.
"""
import json, time
import numpy as np
import pandas as pd

from core import BlockPermuter, pack, popcount, row_shuffle
from fastp import logfac_table, logp_all, minp_scan
from features import build, lattice
from engine import Problem

J, NREP, ALPHA = 300, 120, 0.05
X, y, meta = build("PJM", years=range(2019, 2027), q=.90, fc_shift=1)
r, lab = lattice(X, max_arity=3, min_support=1, verbose=False)
months = np.asarray(pd.to_datetime(X.index).month)
p = Problem(r, lab, y, months=months, device="auto")
n, L, sup, bk = p.n, logfac_table(p.n), p.sup, p.backend
print(f"{meta['n']:,} hours, {r.shape[0]:,} rules, {bk.label}\n", flush=True)

res = {}
for BLOCK in (168, 336, 504, 720, 1008):
    bp = BlockPermuter(n, BLOCK, months)
    nblocks = n // BLOCK
    hb = 0
    t0 = time.time()
    for rep in range(NREP):
        yn = np.roll(y, int(n * (0.15 + 0.7 * rep / NREP)))
        n1 = int(yn.sum())
        a = popcount(r & pack(yn)[None, :]).astype(np.int64)
        obs = logp_all(a, sup, n, n1, L).min()
        rng = np.random.default_rng(100 + rep)
        o = np.empty(J); d = 0
        while d < J:
            B = min(bk.batch, J - d)
            C = bk.counts(yn, [bp.draw(rng) for _ in range(B)])
            for k in range(B):
                o[d+k], _ = minp_scan(C[k], sup, n, n1, L, 0.0)
            d += B
        hb += (np.mean(o <= obs) <= ALPHA)
    fpr = hb / NREP
    res[BLOCK] = dict(fpr=fpr, blocks=int(nblocks), hits=int(hb))
    flag = "OK" if abs(fpr - ALPHA) <= 2*np.sqrt(ALPHA*(1-ALPHA)/NREP) else "off"
    print(f"  block {BLOCK:5d}h ({BLOCK/24:5.1f}d)  {nblocks:4d} blocks  "
          f"FPR {fpr:.3f}  {flag}   {time.time()-t0:.0f}s", flush=True)

best = min(res, key=lambda b: abs(res[b]["fpr"] - ALPHA))
print(f"\n  closest to nominal: block = {best}h ({best/24:.1f} days), FPR {res[best]['fpr']:.3f}")
json.dump(res, open("block_sweep.json", "w"), indent=1)
print("  wrote block_sweep.json")
