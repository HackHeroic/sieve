"""Calibrate SIEVE on market data. Which block length controls error on VIX?"""
import json, time, numpy as np, pandas as pd
from core import BlockPermuter, pack, popcount, row_shuffle
from fastp import logfac_table, logp_all, minp_scan
from adapter import build_generic
from features import lattice
from engine import Problem

df = pd.read_csv("var_data.csv", index_col=0, parse_dates=True)
X, y, meta = build_generic(df, target="VIXCLS", horizon=1, q=.90, lags=(1,2,5), stratify="month")
r, lab = lattice(X, max_arity=3, min_support=1, verbose=False)
months = meta["months"]
p = Problem(r, lab, y, months=months, device="auto")
n, L, sup, bk = p.n, logfac_table(p.n), p.sup, p.backend
print(f"{n:,} days, {r.shape[0]:,} rules, {bk.label}")

def acf(x, lags):
    x=np.asarray(x,float)-np.asarray(x,float).mean(); m=len(x)
    f=np.fft.rfft(x,2*m); a=np.fft.irfft(f*np.conj(f))[:max(lags)+1].real; a/=a[0]
    return {k:a[k] for k in lags}
a = acf(y.astype(float), [1,5,21,63,126,252])
print("  outcome ACF: " + "  ".join(f"L{k}={v:+.3f}" for k,v in a.items()) + "\n")

J, NREP, ALPHA = 200, 80, 0.05
OUT={}
for B in (21, 63, 126, 252, 504):
    bp = BlockPermuter(n, B, months); hb = hr = 0
    t0=time.time()
    for rep in range(NREP):
        yn = np.roll(y, int(n*(0.15+0.7*rep/NREP))); n1=int(yn.sum())
        obs = logp_all(popcount(r & pack(yn)[None,:]).astype(np.int64), sup, n, n1, L).min()
        for kind in ("block","row"):
            rng=np.random.default_rng(100+rep); o=np.empty(J); d=0
            while d<J:
                Bc=min(bk.batch,J-d)
                idxs=[bp.draw(rng) if kind=="block" else row_shuffle(n,rng) for _ in range(Bc)]
                C=bk.counts(yn,idxs)
                for k in range(Bc): o[d+k],_=minp_scan(C[k],sup,n,n1,L,0.0)
                d+=Bc
            if kind=="block": hb += (np.mean(o<=obs)<=ALPHA)
            else:             hr += (np.mean(o<=obs)<=ALPHA)
    OUT[B]=dict(block=hb/NREP, row=hr/NREP, blocks=n//B)
    flag = "OK" if abs(hb/NREP-ALPHA)<=2*np.sqrt(ALPHA*.95/NREP) else "off"
    print(f"  block {B:4d}d ({n//B:4d} blocks)  SIEVE {hb/NREP:.3f}  naive {hr/NREP:.3f}  {flag}"
          f"   {time.time()-t0:.0f}s", flush=True)
best=min(OUT,key=lambda b:abs(OUT[b]['block']-ALPHA))
print(f"\n  calibrated at {best}d blocks: FPR {OUT[best]['block']:.3f} vs naive {OUT[best]['row']:.3f}")
json.dump(dict(sweep=OUT, chosen=best, acf={str(k):v for k,v in a.items()},
               days=int(n), rules=int(r.shape[0])), open("calib_var.json","w"), indent=1, default=float)
print("  wrote calib_var.json")
