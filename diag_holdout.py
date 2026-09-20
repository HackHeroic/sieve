"""Why is the holdout 0/20 on corrected data? Diagnose, do not guess."""
import numpy as np, pandas as pd, engine
from engine import Problem
from features import build, lattice
engine.BLOCK = 168

for shift in (0, 1):
    X, y, _ = build("PJM", years=range(2019,2027), q=.90, fc_shift=shift, fit_until_year=2022)
    idx = pd.to_datetime(X.index); months = np.asarray(idx.month)
    early = np.asarray(idx.year <= 2022)
    def fit(mask, seed):
        r, lab = lattice(X[mask], max_arity=3, min_support=1, verbose=False)
        p = Problem(r, lab, y[mask], months=months[mask], alpha=0.05)
        return p, lab, p.null(400, kind="block", seed=seed)
    pA, labA, nA = fit(early, 1); pB, labB, nB = fit(~early, 2)
    sA, sB = pA.significant(nA), pB.significant(nB)
    cutB = pB.cutoff(nB)
    cand = sA[pA.sup[sA] > 500]; top = cand[np.argsort(-pA.lift[cand])][:20]
    ib = {l:i for i,l in enumerate(labB)}
    js = [ib.get(labA[i]) for i in top]
    found = [j for j in js if j is not None]
    lps = np.array([pB.lp[j] for j in found])
    print(f"\n=== fc_shift={shift} ===")
    print(f"  TRAIN significant {len(sA):,} of {len(labA):,}   TEST significant {len(sB):,} of {len(labB):,}")
    print(f"  TEST cutoff        {cutB:9.1f}")
    print(f"  the 20 train rules, their TEST log p:")
    print(f"     best {lps.min():9.1f}   median {np.median(lps):9.1f}   worst {lps.max():9.1f}")
    print(f"     how many clear the test cutoff: {(lps<=cutB).sum()}/{len(lps)}")
    print(f"  TEST null range    {nB.min():9.1f} .. {nB.max():9.1f}")
    print(f"  TEST observed min  {pB.lp.min():9.1f}")
    # how extreme would they need to be?
    print(f"  gap: best train rule is {lps.min()-cutB:+.1f} log-units from the cutoff")
