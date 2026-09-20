"""
Everything, re-run with the PJM forecast timestamp corrected (fc_shift=+1).

Why
---
EIA-930's demand-forecast series for PJM is filed on an hour-shifted convention
relative to realised demand.  Sweeping the shift over 2019-2026:

    shift  -2     -1      0     +1     +2     +3
    MAPE  7.73   5.67   3.62   2.45   3.49   5.49

Our pipeline used shift 0, so roughly a third of what we were calling "forecast
error" was our own misalignment.  This re-runs the headline numbers on the
corrected series.
"""
import json, time
import numpy as np
import pandas as pd

import backends
from core import pack, popcount, block_permutation, row_shuffle, BlockPermuter
from fastp import logfac_table, logp_all, minp_scan
from features import build, lattice
from engine import Problem

OUT = {}
t0 = time.time()
X, y, meta = build("PJM", years=range(2019, 2027), q=.90, fc_shift=1)
r, lab = lattice(X, max_arity=3, min_support=1, verbose=False)
months = np.asarray(pd.to_datetime(X.index).month)
p = Problem(r, lab, y, months=months, device="auto")
print(f"corrected: {meta['n']:,} hours, {meta['n1']:,} positive, {r.shape[0]:,} rules")
print(f"backend {p.backend.label}   {time.time()-t0:.0f}s\n", flush=True)

# ---- canonical nulls -------------------------------------------------
nb = p.null(1000, kind="block", seed=1)
nr = p.null(1000, kind="row",   seed=1)
sb, sr = len(p.significant(nb)), len(p.significant(nr))
est, sd = p.count_ci(nb)
OUT["canonical"] = dict(sig_block=sb, sig_row=sr, inflation=sr/max(sb,1),
                        boot=est, boot_sd=sd, J=1000)
print(f"significant: block {sb:,} [{est:,} +/- {sd:,}]   row {sr:,}   "
      f"inflation {sr/max(sb,1):.2f}x", flush=True)
ch, cum = p.cover(nb, top=6, min_support=800, min_new=40)
OUT["findings"] = ch
print("\nFINDINGS")
for c in ch:
    print(f"  lift {c['lift']:4.2f}  n={c['support']:5,}  {c['rule']}")

# ---- calibration on 120 true nulls -----------------------------------
print("\ncalibration, 120 true nulls ...", flush=True)
n, L = p.n, logfac_table(p.n)
sup = p.sup
bk = p.backend
bp = BlockPermuter(n, 168, months)
J, NREP = 300, 120
hb = hr = 0
t0 = time.time()
for rep in range(NREP):
    yn = np.roll(y, int(n * (0.15 + 0.7 * rep / NREP)))
    n1 = int(yn.sum())
    a = popcount(r & pack(yn)[None, :]).astype(np.int64)
    obs = logp_all(a, sup, n, n1, L).min()
    def null(kind, seed):
        rng = np.random.default_rng(seed); o = np.empty(J); d = 0
        while d < J:
            B = min(bk.batch, J - d)
            idxs = [bp.draw(rng) if kind == "block" else row_shuffle(n, rng)
                    for _ in range(B)]
            C = bk.counts(yn, idxs)
            for k in range(B):
                o[d+k], _ = minp_scan(C[k], sup, n, n1, L, 0.0)
            d += B
        return o
    hb += (np.mean(null("block", 100+rep) <= obs) <= 0.05)
    hr += (np.mean(null("row",   100+rep) <= obs) <= 0.05)
    if (rep+1) % 20 == 0:
        print(f"  {rep+1}/{NREP}  block {hb/(rep+1):.3f}  row {hr/(rep+1):.3f}"
              f"  {time.time()-t0:.0f}s", flush=True)
OUT["calibration"] = dict(block=hb/NREP, row=hr/NREP, n=NREP, J=J)
print(f"\nCALIBRATION (corrected): block {hb/NREP:.3f}   row {hr/NREP:.3f}")
json.dump(OUT, open("corrected_results.json", "w"), indent=1, default=float)
print("wrote corrected_results.json")
