"""
Final PJM numbers: date-gated shift, corrupt rows removed, block length re-chosen.
"""
import json, time
import numpy as np
import pandas as pd

from core import BlockPermuter, pack, popcount, row_shuffle
from fastp import logfac_table, logp_all, minp_scan
from features import build, lattice
from engine import Problem
import engine

OUT = {}
X, y, meta = build("PJM", years=range(2019, 2027), q=.90, fc_shift=1)
r, lab = lattice(X, max_arity=3, min_support=1, verbose=False)
months = np.asarray(pd.to_datetime(X.index).month)
print(f"{meta['n']:,} hours, {meta['n1']:,} positive, {r.shape[0]:,} rules\n", flush=True)

for BLOCK in (168, 504):
    engine.BLOCK = BLOCK
    p = Problem(r, lab, y, months=months, device="auto")
    nb = p.null(1000, kind="block", seed=1)
    nr = p.null(1000, kind="row",   seed=1)
    sb, sr = len(p.significant(nb)), len(p.significant(nr))
    est, sd = p.count_ci(nb)
    ch, cum = p.cover(nb, top=6, min_support=800, min_new=40)
    OUT[f"block{BLOCK}"] = dict(sig_block=sb, sig_row=sr, inflation=sr/max(sb,1),
                                boot=est, boot_sd=sd, findings=ch, covered=cum)
    print(f"block {BLOCK}h: significant {sb:,} [{est:,} +/- {sd:,}]  row {sr:,}  "
          f"inflation {sr/max(sb,1):.2f}x", flush=True)
    for c in ch[:3]:
        print(f"    lift {c['lift']:4.2f}  n={c['support']:5,}  {c['rule']}")

# calibration at the chosen block length
print("\ncalibration at 504h, 120 true nulls ...", flush=True)
p = Problem(r, lab, y, months=months, device="auto")
n, L, sup, bk = p.n, logfac_table(p.n), p.sup, p.backend
bp = BlockPermuter(n, 504, months)
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
            idxs = [bp.draw(rng) if kind == "block" else row_shuffle(n, rng) for _ in range(B)]
            C = bk.counts(yn, idxs)
            for k in range(B):
                o[d+k], _ = minp_scan(C[k], sup, n, n1, L, 0.0)
            d += B
        return o
    hb += (np.mean(null("block", 100+rep) <= obs) <= 0.05)
    hr += (np.mean(null("row",   100+rep) <= obs) <= 0.05)
    if (rep+1) % 40 == 0:
        print(f"  {rep+1}/{NREP}  block {hb/(rep+1):.3f}  row {hr/(rep+1):.3f}  "
              f"{time.time()-t0:.0f}s", flush=True)
OUT["calibration_504h"] = dict(block=hb/NREP, row=hr/NREP, n=NREP, J=J, block_hours=504)
OUT["meta"] = dict(hours=meta['n'], positive=meta['n1'], rules=int(r.shape[0]),
                   fc_shift=1, date_gated="2019-01-30", corrupt_rows_removed=True)
print(f"\nFINAL CALIBRATION: block {hb/NREP:.3f}   row {hr/NREP:.3f}")
json.dump(OUT, open("final_results.json", "w"), indent=1, default=float)
print("wrote final_results.json")
