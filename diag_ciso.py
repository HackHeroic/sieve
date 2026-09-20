"""Can CISO calibrate at ANY block length, or is the null construction wrong?"""
import numpy as np, pandas as pd
from core import BlockPermuter, pack, popcount
from fastp import logfac_table, logp_all, minp_scan
from features import build, lattice
from engine import Problem

def acf(x, lags):
    x = np.asarray(x,float)-np.asarray(x,float).mean(); n=len(x)
    f=np.fft.rfft(x,2*n); a=np.fft.irfft(f*np.conj(f))[:max(lags)+1].real; a/=a[0]
    return {k:a[k] for k in lags}

lags=[1,24,168,504,1008,2016,4368,8760]
print("outcome autocorrelation at long lags (8760h = 1 year):")
for ba,sh in (("PJM",1),("CISO",0),("MISO",0)):
    X,y,m = build(ba, years=range(2019,2027), q=.90, fc_shift=sh)
    a=acf(y.astype(float), lags)
    print(f"  {ba:5s} " + " ".join(f"L{k}={a[k]:+.3f}" for k in lags))

print("\nCISO: does a much longer block help?")
X,y,meta = build("CISO", years=range(2019,2027), q=.90, fc_shift=0)
r,lab = lattice(X, max_arity=3, min_support=1, verbose=False)
months=np.asarray(pd.to_datetime(X.index).month)
p = Problem(r,lab,y,months=months,device="auto")
n,L,sup,bk = p.n, logfac_table(p.n), p.sup, p.backend
J,NREP=200,60
for B in (504, 1008, 2016, 4368):
    bp=BlockPermuter(n,B,months); hb=0
    nblk = n//B
    for rep in range(NREP):
        yn=np.roll(y,int(n*(0.15+0.7*rep/NREP))); n1=int(yn.sum())
        obs=logp_all(popcount(r&pack(yn)[None,:]).astype(np.int64),sup,n,n1,L).min()
        rng=np.random.default_rng(100+rep); o=np.empty(J); d=0
        while d<J:
            Bc=min(bk.batch,J-d)
            C=bk.counts(yn,[bp.draw(rng) for _ in range(Bc)])
            for k in range(Bc): o[d+k],_=minp_scan(C[k],sup,n,n1,L,0.0)
            d+=Bc
        hb+=(np.mean(o<=obs)<=0.05)
    print(f"  block {B:5d}h ({B/24:5.1f}d, {nblk:3d} blocks, {nblk/12:4.1f}/stratum)  FPR {hb/NREP:.3f}", flush=True)
