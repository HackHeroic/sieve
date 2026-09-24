"""Full re-run on the repaired dataset: 64,690 hours, all predicates firing."""
import json, time
import numpy as np, pandas as pd, engine
from core import BlockPermuter, pack, popcount, row_shuffle
from fastp import logfac_table, logp_all, minp_scan
from features import build, lattice
from engine import Problem

OUT={}; t0=time.time()
X, y, meta = build("PJM", years=range(2019,2027), q=.90, fc_shift=1)
r, lab = lattice(X, max_arity=3, min_support=1, verbose=False)
months = np.asarray(pd.to_datetime(X.index).month)
print(f"{meta['n']:,} hours, {meta['n1']:,} positive, {r.shape[0]:,} rules  {time.time()-t0:.0f}s", flush=True)
OUT['meta']=dict(hours=meta['n'], positive=meta['n1'], rules=int(r.shape[0]),
                 predicates=int(X.shape[1]), end=meta['end'])

# calibrate the block length on the new sample
p = Problem(r, lab, y, months=months, device="auto")
n, L, sup, bk = p.n, logfac_table(p.n), p.sup, p.backend
print(f"backend {bk.label}\n", flush=True)
J, NREP, ALPHA = 250, 100, 0.05
best=None
for B in (504, 720, 1008, 1512):
    bp=BlockPermuter(n,B,months); hb=hr=0; t=time.time()
    for rep in range(NREP):
        yn=np.roll(y,int(n*(0.15+0.7*rep/NREP))); n1=int(yn.sum())
        obs=logp_all(popcount(r&pack(yn)[None,:]).astype(np.int64),sup,n,n1,L).min()
        for kind in ("block","row"):
            rng=np.random.default_rng(100+rep); o=np.empty(J); d=0
            while d<J:
                Bc=min(bk.batch,J-d)
                idxs=[bp.draw(rng) if kind=="block" else row_shuffle(n,rng) for _ in range(Bc)]
                C=bk.counts(yn,idxs)
                for k in range(Bc): o[d+k],_=minp_scan(C[k],sup,n,n1,L,0.0)
                d+=Bc
            if kind=="block": hb+=(np.mean(o<=obs)<=ALPHA)
            else:             hr+=(np.mean(o<=obs)<=ALPHA)
    OUT.setdefault('calibration',{})[B]=dict(block=hb/NREP,row=hr/NREP,blocks=n//B)
    ok = abs(hb/NREP-ALPHA)<=2*np.sqrt(ALPHA*.95/NREP)
    print(f"  block {B:5d}h ({n//B:4d} blk)  SIEVE {hb/NREP:.3f}  naive {hr/NREP:.3f}  {'OK' if ok else 'off'}  {time.time()-t:.0f}s", flush=True)
    if ok: best=B; break
best = best or 1512
OUT['chosen_block']=best

engine.BLOCK=best
p = Problem(r, lab, y, months=months, device="auto")
nb=p.null(1000,kind="block",seed=1); nr=p.null(1000,kind="row",seed=1)
sb,sr=len(p.significant(nb)),len(p.significant(nr))
est,sd=p.count_ci(nb)
ch,cum=p.cover(nb,top=8,min_support=800,min_new=40)
OUT['result']=dict(block=best, sig_block=sb, sig_row=sr, inflation=sr/max(sb,1),
                   boot=est, boot_sd=sd, findings=ch, covered=cum)
print(f"\n  block {best}h: significant {sb:,} [{est:,} +/- {sd:,}]  naive {sr:,}  inflation {sr/max(sb,1):.2f}x")
print(f"\n  FINDINGS")
for c in ch: print(f"    lift {c['lift']:4.2f}  n={c['support']:5,}  {c['rule']}")
hourrules=[c for c in ch if any(k in c['rule'] for k in ('is_morning','is_evening','is_overnight'))]
print(f"\n  findings using a time-of-day predicate: {len(hourrules)} (previously impossible)")
json.dump(OUT, open("repaired_results.json","w"), indent=1, default=float)
print(f"\n  wrote repaired_results.json  ({time.time()-t0:.0f}s)")
