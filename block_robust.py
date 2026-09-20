"""
Is the holdout / replication conclusion robust to the block length?

The tension: the corrected full-sample PJM series calibrates at 504h, but the
holdout TEST window is 18 months, where 504h leaves 2.1 blocks per month
stratum. Rather than pick one and hope, run both and report whether the
conclusion moves.
"""
import json
import numpy as np, pandas as pd, engine
from engine import Problem
from features import build, lattice

OUT = {}
def fit(X, y, months, mask, seed, J=400):
    r, lab = lattice(X[mask], max_arity=3, min_support=1, verbose=False)
    p = Problem(r, lab, y[mask], months=months[mask], alpha=0.05)
    return p, lab, p.null(J, kind="block", seed=seed)

X, y, _ = build("PJM", years=range(2019,2027), q=.90, fc_shift=1, fit_until_year=2022)
idx = pd.to_datetime(X.index); months = np.asarray(idx.month)
early = np.asarray(idx.year <= 2022)

print("HOLDOUT at both block lengths")
for B in (168, 504):
    engine.BLOCK = B
    pA, labA, nA = fit(X, y, months, early, 1)
    pB, labB, nB = fit(X, y, months, ~early, 2)
    sA = pA.significant(nA); cand = sA[pA.sup[sA] > 500]
    top = cand[np.argsort(-pA.lift[cand])][:20]
    cutB = pB.cutoff(nB); ib = {l:i for i,l in enumerate(labB)}
    held, tr, te = 0, [], []
    for i in top:
        j = ib.get(labA[i])
        if j is None: continue
        held += int(pB.lp[j] <= cutB); tr.append(float(pA.lift[i])); te.append(float(pB.lift[j]))
    OUT[f"holdout_{B}"] = dict(held=held, tested=len(tr), test_significant=int(len(pB.significant(nB))),
                               lift_train=float(np.mean(tr)), lift_test=float(np.mean(te)))
    print(f"  block {B:4d}h: {held}/{len(tr)} held   test-set significant {len(pB.significant(nB)):,}"
          f"   lift {np.mean(tr):.2f} -> {np.mean(te):.2f}", flush=True)

print("\nREPLICATION at both block lengths")
def grid(ba, seed, shift, B, J=400):
    engine.BLOCK = B
    Xb, yb, _ = build(ba, years=range(2019,2027), q=.90, fc_shift=shift)
    r, lab = lattice(Xb, max_arity=3, min_support=1, verbose=False)
    m = np.asarray(pd.to_datetime(Xb.index).month)
    p = Problem(r, lab, yb, months=m, alpha=0.05)
    return p, lab, p.null(J, kind="block", seed=seed)
for B in (168, 504):
    pP, labP, nP = grid("PJM", 3, 1, B)
    sP = pP.significant(nP); cP = sP[pP.sup[sP] > 1000]
    topP = cP[np.argsort(-pP.lift[cP])][:15]
    okt, nt, per = 0, 0, {}
    for ba in ("CISO","ERCO","MISO","NYIS"):
        q, lq, nq = grid(ba, 4, 0, B)
        cq = q.cutoff(nq); iq = {l:i for i,l in enumerate(lq)}
        ok, cnt = 0, 0
        for i in topP:
            j = iq.get(labP[i])
            if j is None: continue
            cnt += 1; ok += int(q.lp[j] <= cq)
        per[ba] = f"{ok}/{cnt}"; okt += ok; nt += cnt
    OUT[f"replication_{B}"] = dict(replicated=okt, tested=nt, per_ba=per,
                                   pjm_significant=int(len(sP)))
    print(f"  block {B:4d}h: {okt}/{nt}   {per}", flush=True)

json.dump(OUT, open("block_robust.json","w"), indent=1, default=float)
print("\nwrote block_robust.json")
