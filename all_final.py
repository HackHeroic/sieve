"""Everything remaining, one pass, corrected data."""
import json, time
import numpy as np, pandas as pd
import engine
from engine import Problem
from features import build, lattice
from core import BlockPermuter

OUT = {}; T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

# ---------- 1. holdout, 168h ----------
engine.BLOCK = 168
X, y, meta = build("PJM", years=range(2019,2027), q=.90, fc_shift=1, fit_until_year=2022)
idx = pd.to_datetime(X.index); months = np.asarray(idx.month)
early = np.asarray(idx.year <= 2022)
log(f"holdout: train {early.sum():,} test {(~early).sum():,}")
def fit(mask, seed, J=500):
    r, lab = lattice(X[mask], max_arity=3, min_support=1, verbose=False)
    p = Problem(r, lab, y[mask], months=months[mask], alpha=0.05)
    return p, lab, p.null(J, kind="block", seed=seed)
pA, labA, nA = fit(early, 1)
sA = pA.significant(nA); cand = sA[pA.sup[sA] > 500]
top = cand[np.argsort(-pA.lift[cand])][:20]
pB, labB, nB = fit(~early, 2)
cutB = pB.cutoff(nB); ib = {l:i for i,l in enumerate(labB)}
held, tr, te = 0, [], []
for i in top:
    j = ib.get(labA[i])
    if j is None: continue
    held += int(pB.lp[j] <= cutB); tr.append(float(pA.lift[i])); te.append(float(pB.lift[j]))
OUT["holdout"] = dict(held=held, tested=len(tr), lift_train=float(np.mean(tr)),
                      lift_test=float(np.mean(te)), block=168)
log(f"HOLDOUT {held}/{len(tr)}  lift {np.mean(tr):.2f} -> {np.mean(te):.2f}")

# ---------- 2. cross-BA replication, 168h ----------
def grid(ba, seed, shift, J=500):
    Xb, yb, _ = build(ba, years=range(2019,2027), q=.90, fc_shift=shift)
    r, lab = lattice(Xb, max_arity=3, min_support=1, verbose=False)
    m = np.asarray(pd.to_datetime(Xb.index).month)
    p = Problem(r, lab, yb, months=m, alpha=0.05)
    return p, lab, p.null(J, kind="block", seed=seed)
pP, labP, nP = grid("PJM", 3, 1)
sP = pP.significant(nP); cP = sP[pP.sup[sP] > 1000]
topP = cP[np.argsort(-pP.lift[cP])][:15]
log(f"replication: PJM {len(sP):,} significant, top {len(topP)} tested elsewhere")
rep, okt, nt = {}, 0, 0
for ba in ("CISO","ERCO","MISO","NYIS"):
    q, lq, nq = grid(ba, 4, 0)
    cq = q.cutoff(nq); iq = {l:i for i,l in enumerate(lq)}
    ok, lifts, miss = 0, [], 0
    for i in topP:
        j = iq.get(labP[i])
        if j is None: miss += 1; continue
        lifts.append(float(q.lift[j])); ok += int(q.lp[j] <= cq)
    rep[ba] = dict(replicated=ok, tested=len(lifts), not_expressible=miss,
                   mean_lift=float(np.mean(lifts)) if lifts else None)
    okt += ok; nt += len(lifts)
    log(f"  {ba}: {ok}/{len(lifts)}  lift {np.mean(lifts):.2f}")
OUT["replication"] = dict(per_ba=rep, replicated=okt, tested=nt, block=168)
log(f"REPLICATION {okt}/{nt}")

# ---------- 3. sensitivity on corrected data, 504h ----------
engine.BLOCK = 504
Xf, yf, mf = build("PJM", years=range(2019,2027), q=.90, fc_shift=1)
rf, lf = lattice(Xf, max_arity=3, min_support=1, verbose=False)
ix = pd.to_datetime(Xf.index); mo = np.asarray(ix.month); yr = np.asarray(ix.year)
sens = {}
for nm, st in (("month only", mo), ("year x month", yr*100+mo),
               ("year only", yr), ("unstratified", None)):
    p = Problem(rf, lf, yf, months=st, alpha=0.05)
    nb = p.null(500, kind="block", seed=1)
    sens[nm] = int(len(p.significant(nb)))
    log(f"  sensitivity {nm:16s} {sens[nm]:7,}")
p = Problem(rf, lf, yf, months=mo, alpha=0.05)
sens["row shuffle"] = int(len(p.significant(p.null(500, kind="row", seed=1))))
log(f"  sensitivity {'row shuffle':16s} {sens['row shuffle']:7,}")
OUT["sensitivity"] = sens

json.dump(OUT, open("all_final.json","w"), indent=1, default=float)
log("wrote all_final.json  DONE")
