"""
Five-grid replication on the FIXED block permutation.

Why: the 57/58 replication and the per-grid counts in FINAL.md were all measured
BEFORE the ragged-tail fix in core.block_permutation.  The calibration has been
re-run (0.058); these have not.  Weather retention is NOT re-run here because
stratified_lift is a conditional lift computation, not a permutation test, so it
does not depend on the fix.
"""
import json, time
import numpy as np
import pandas as pd

import backends
from core import pack, popcount, block_permutation, row_shuffle
from fastp import logfac_table, logp_all, minp_scan
from features import build, lattice

J, BLOCK, ALPHA = 500, 168, 0.05
out = {}

for ba in ("PJM", "CISO", "ERCO", "MISO", "NYIS"):
    t0 = time.time()
    X, y, _ = build(ba, years=range(2019, 2027), q=.90)
    rules, labels = lattice(X, max_arity=3, min_support=1, verbose=False)
    months = np.asarray(pd.to_datetime(X.index).month)
    n, n1 = len(y), int(y.sum())
    L = logfac_table(n)
    sup = popcount(rules).astype(np.int64)
    bk = backends.make("auto", rules)

    def null(kind, seed):
        permuter = block_permutation if kind == "block" else row_shuffle
        strata = months if kind == "block" else None
        rng = np.random.default_rng(seed)
        o = np.empty(J); d = 0
        while d < J:
            B = min(bk.batch, J - d)
            idxs = [permuter(n, block=BLOCK, rng=rng, strata=strata) for _ in range(B)]
            C = bk.counts(y, idxs)
            for k in range(B):
                o[d+k], _ = minp_scan(C[k], sup, n, n1, L, 0.0)
            d += B
        return o

    a = popcount(rules & pack(y)[None, :]).astype(np.int64)
    lp = logp_all(a, sup, n, n1, L)
    nb, nr = null("block", 11), null("row", 11)
    cb, cr = np.quantile(nb, ALPHA), np.quantile(nr, ALPHA)
    sb, sr = int((lp <= cb).sum()), int((lp <= cr).sum())
    lift = (a / np.maximum(sup, 1)) / y.mean()
    cand = [i for i in np.flatnonzero(lp <= cb) if sup[i] > 800]
    top = sorted(cand, key=lambda i: -lift[i])[:10]
    out[ba] = dict(hours=n, rules=int(rules.shape[0]), sig_block=sb, sig_row=sr,
                   inflation=sr/max(sb, 1), mean_lift_top10=float(np.mean([lift[i] for i in top])),
                   top=" AND ".join(labels[top[0]]))
    print(f"  {ba}: {n:,}h {rules.shape[0]:,}r  sig {sb:,} vs {sr:,} "
          f"(infl {sr/max(sb,1):.2f}x)  lift {out[ba]['mean_lift_top10']:.2f}  "
          f"{time.time()-t0:.0f}s", flush=True)
    bk.close()

json.dump(out, open("replicate_fixed.json", "w"), indent=1, default=float)
print("\nwrote replicate_fixed.json")
