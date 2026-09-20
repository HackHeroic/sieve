"""
Final cross-grid replication, each grid at ITS OWN calibrated block length.

Block length must cover the series' dependence horizon.  Measured outcome ACF:
    PJM   L168=0.007  L504=0.001      -> short horizon
    MISO  L168=0.084  L504=0.020      -> short-ish
    CISO  L168=0.211  L504=0.152  L1008=0.116  -> LONG
Using one length everywhere gave FPR 0.850 on CISO.  Each grid is calibrated
here and then the replication is run at the length that calibrates it.
"""
import json
import numpy as np, pandas as pd
from core import BlockPermuter, pack, popcount
from fastp import logfac_table, logp_all, minp_scan
from features import build, lattice
from engine import Problem

ALPHA, JC, NREP, JR = 0.05, 200, 60, 400
GRIDS = (("PJM",1), ("CISO",0), ("ERCO",0), ("MISO",0), ("NYIS",0))
CANDS = (504, 1008, 2016, 4368)
built, chosen = {}, {}

for ba, sh in GRIDS:
    X, y, _ = build(ba, years=range(2019,2027), q=.90, fc_shift=sh)
    r, lab = lattice(X, max_arity=3, min_support=1, verbose=False)
    months = np.asarray(pd.to_datetime(X.index).month)
    p = Problem(r, lab, y, months=months, device="auto")
    built[ba] = (X, y, r, lab, months, p)
    n, L, sup, bk = p.n, logfac_table(p.n), p.sup, p.backend
    best, bestd = None, 9
    fprs = {}
    for B in CANDS:
        bp = BlockPermuter(n, B, months); hb = 0
        for rep in range(NREP):
            yn = np.roll(y, int(n*(0.15+0.7*rep/NREP))); n1 = int(yn.sum())
            obs = logp_all(popcount(r & pack(yn)[None,:]).astype(np.int64), sup, n, n1, L).min()
            rng = np.random.default_rng(100+rep); o = np.empty(JC); d = 0
            while d < JC:
                Bc = min(bk.batch, JC-d)
                C = bk.counts(yn, [bp.draw(rng) for _ in range(Bc)])
                for k in range(Bc): o[d+k], _ = minp_scan(C[k], sup, n, n1, L, 0.0)
                d += Bc
            hb += (np.mean(o <= obs) <= ALPHA)
        fprs[B] = hb/NREP
        if abs(fprs[B]-ALPHA) < bestd: best, bestd = B, abs(fprs[B]-ALPHA)
        if fprs[B] <= 0.10: break          # good enough, stop early
    chosen[ba] = dict(block=best, fpr=fprs[best], sweep=fprs)
    print(f"  {ba:5s} -> {best}h ({best/24:.0f}d)  FPR {fprs[best]:.3f}   sweep {fprs}", flush=True)

# replication with each grid at its own calibrated length
import engine
X,y,r,lab,months,_ = built["PJM"]
engine.BLOCK = chosen["PJM"]["block"]
pP = Problem(r, lab, y, months=months, device="auto")
nP = pP.null(JR, kind="block", seed=3)
sP = pP.significant(nP); cP = sP[pP.sup[sP] > 1000]
topP = cP[np.argsort(-pP.lift[cP])][:15]
print(f"\nPJM {len(sP):,} significant at {chosen['PJM']['block']}h; testing top {len(topP)}", flush=True)
rep, okt, nt = {}, 0, 0
for ba, _ in GRIDS[1:]:
    Xb,yb,rb,lb,mb,_ = built[ba]
    engine.BLOCK = chosen[ba]["block"]
    q = Problem(rb, lb, yb, months=mb, device="auto")
    nq = q.null(JR, kind="block", seed=4); cq = q.cutoff(nq)
    iq = {l:i for i,l in enumerate(lb)}
    ok, cnt, lifts = 0, 0, []
    for i in topP:
        j = iq.get(lab[i])
        if j is None: continue
        cnt += 1; ok += int(q.lp[j] <= cq); lifts.append(float(q.lift[j]))
    rep[ba] = dict(replicated=ok, tested=cnt, block=chosen[ba]["block"],
                   fpr=chosen[ba]["fpr"], mean_lift=float(np.mean(lifts)) if lifts else None)
    okt += ok; nt += cnt
    print(f"  {ba}: {ok}/{cnt} at {chosen[ba]['block']}h (FPR {chosen[ba]['fpr']:.3f})  "
          f"lift {np.mean(lifts):.2f}", flush=True)
print(f"\nREPLICATION (each grid calibrated): {okt}/{nt}")
json.dump(dict(calibration=chosen, replication=rep, total=f"{okt}/{nt}"),
          open("final_replication.json","w"), indent=1, default=float)
print("wrote final_replication.json")
