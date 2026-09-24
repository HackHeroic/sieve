"""Canonical run on the repaired dataset, then calibration."""
import json, time
import numpy as np, pandas as pd, engine
from core import BlockPermuter, pack, popcount, row_shuffle
from fastp import logfac_table, logp_all, minp_scan
from features import build, lattice
from engine import Problem

OUT={}; t0=time.time()
X,y,meta = build("PJM", years=range(2019,2027), q=.90, fc_shift=1)
r,lab = lattice(X, max_arity=3, min_support=1, verbose=False)
months = np.asarray(pd.to_datetime(X.index).month)
OUT['meta']=dict(hours=meta['n'], positive=meta['n1'], rules=int(r.shape[0]),
                 predicates=int(X.shape[1]), start=meta['start'][:10], end=meta['end'][:10])
print(f"{meta['n']:,} hours ({meta['start'][:10]} to {meta['end'][:10]}), "
      f"{meta['n1']:,} positive, {r.shape[0]:,} rules", flush=True)

engine.BLOCK = 504
p = Problem(r, lab, y, months=months)
nb = p.null(1000, kind="block", seed=1); print(f"  block null {time.time()-t0:.0f}s", flush=True)
nr = p.null(1000, kind="row",   seed=1); print(f"  row null   {time.time()-t0:.0f}s", flush=True)
sb,sr = len(p.significant(nb)), len(p.significant(nr))
est,sd = p.count_ci(nb)
ch,cum = p.cover(nb, top=8, min_support=800, min_new=40)
OUT['result']=dict(block=504, sig_block=sb, sig_row=sr, inflation=sr/max(sb,1),
                   boot=est, boot_sd=sd, findings=ch, covered=cum)
print(f"\n  significant {sb:,} [{est:,} +/- {sd:,}]   naive {sr:,}   inflation {sr/max(sb,1):.2f}x\n")
for c in ch: print(f"    lift {c['lift']:4.2f}  n={c['support']:5,}  {c['rule']}")
hr=[c for c in ch if any(k in c['rule'] for k in ('is_morning','is_evening','is_overnight'))]
print(f"\n  findings using a time-of-day predicate: {len(hr)}  (previously structurally impossible)")
json.dump(OUT, open("repaired_results.json","w"), indent=1, default=float)
print(f"  wrote repaired_results.json  ({time.time()-t0:.0f}s)")

# calibration
print(f"\ncalibrating block length ...", flush=True)
n,L,sup = p.n, logfac_table(p.n), p.sup
J,NREP,ALPHA = 150, 50, 0.05
for B in (504, 1008, 2016):
    bp=BlockPermuter(n,B,months); hb=hrw=0; t=time.time()
    for rep in range(NREP):
        yn=np.roll(y,int(n*(0.15+0.7*rep/NREP))); n1=int(yn.sum())
        obs=logp_all(popcount(r&pack(yn)[None,:]).astype(np.int64),sup,n,n1,L).min()
        for kind in ("block","row"):
            rng=np.random.default_rng(100+rep); o=np.empty(J)
            for j in range(J):
                idx = bp.draw(rng) if kind=="block" else row_shuffle(n,rng)
                c=popcount(r & pack(y[idx])[None,:]).astype(np.int64)
                o[j],_=minp_scan(c,sup,n,n1,L,0.0)
            if kind=="block": hb+=(np.mean(o<=obs)<=ALPHA)
            else:             hrw+=(np.mean(o<=obs)<=ALPHA)
    OUT.setdefault('calibration',{})[B]=dict(block=hb/NREP,row=hrw/NREP,blocks=n//B)
    ok=abs(hb/NREP-ALPHA)<=2*np.sqrt(ALPHA*.95/NREP)
    print(f"  block {B:5d}h ({n//B:4d} blk)  SIEVE {hb/NREP:.3f}  naive {hrw/NREP:.3f}  "
          f"{'OK' if ok else 'off'}   {time.time()-t:.0f}s", flush=True)
    json.dump(OUT, open("repaired_results.json","w"), indent=1, default=float)
    if ok: break
print("DONE")
