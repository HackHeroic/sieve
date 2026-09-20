"""Re-run EVERY headline result on the corrected implementation."""
import json, time, numpy as np, cupy as cp, pandas as pd
from core import pack, popcount
from fastp import logfac_table, logp_all, minp_scan
from gpu_kernel import counts_batch
from gpu_full import gather_pack, block_idx_gpu
from features import build, lattice
from weather import strata, stratified_lift
R_ = {}

def engine(X, y, J=600, seed=0, block=168):
    r, lab = lattice(X, max_arity=3, min_support=1, verbose=False)
    rg = cp.asarray(r); sup = popcount(r).astype(np.int64)
    n = len(y); n1 = int(y.sum()); L = logfac_table(n)
    yg = cp.asarray(y.astype(np.uint8))
    rng = np.random.default_rng(seed); nul = np.empty(J); d = 0
    while d < J:
        b = min(32, J-d)
        ig = block_idx_gpu(n, block, b, rng)
        c = cp.asnumpy(counts_batch(rg, gather_pack(yg, ig, n, r.shape[1], b),
                                    b, r.shape[0], r.shape[1])).astype(np.int64)
        for k in range(b): nul[d+k], _ = minp_scan(c[k], sup, n, n1, L, 0.0)
        d += b
    a = popcount(r & pack(y)[None,:]).astype(np.int64)
    lp = logp_all(a, sup, n, n1, L)
    del rg; cp.get_default_memory_pool().free_all_blocks()
    return r, lab, sup, a, lp, nul

def shuffle_null(X, y, J=600, seed=0):
    r, lab = lattice(X, max_arity=3, min_support=1, verbose=False)
    rg = cp.asarray(r); sup = popcount(r).astype(np.int64)
    n=len(y); n1=int(y.sum()); L=logfac_table(n); yg=cp.asarray(y.astype(np.uint8))
    rng=np.random.default_rng(seed); nul=np.empty(J); d=0
    while d<J:
        b=min(32,J-d)
        ig=cp.asarray(np.stack([rng.permutation(n) for _ in range(b)]).astype(np.int32))
        c=cp.asnumpy(counts_batch(rg, gather_pack(yg,ig,n,r.shape[1],b), b,
                                  r.shape[0], r.shape[1])).astype(np.int64)
        for k in range(b): nul[d+k],_=minp_scan(c[k],sup,n,n1,L,0.0)
        d+=b
    del rg; cp.get_default_memory_pool().free_all_blocks()
    return nul

# ---- 1  calibration on true nulls -------------------------------------
print("1. FWER calibration, 120 true nulls (fixed code)", flush=True)
X, y, _ = build('PJM', years=range(2019,2027), q=.90)
r, lab, sup, a, lp, _ = engine(X, y, J=1, seed=0)
n = len(y); L = logfac_table(n); rg = cp.asarray(r); yg0 = None
hb = hr = 0; NREP = 120
t0=time.time()
for rep in range(NREP):
    yn = np.roll(y, int(n*(0.15+0.7*rep/NREP)))
    nb = engine(X, yn, J=300, seed=100+rep)[5]
    nr = shuffle_null(X, yn, J=300, seed=100+rep)
    o = logp_all(popcount(r & pack(yn)[None,:]).astype(np.int64), sup, n, int(yn.sum()), L).min()
    hb += (np.mean(nb<=o) <= .05); hr += (np.mean(nr<=o) <= .05)
print(f"   block permutation FWER = {hb/NREP:.3f}   (target 0.05)")
print(f"   row shuffle       FWER = {hr/NREP:.3f}")
print(f"   {time.time()-t0:.0f}s", flush=True)
R_['calibration'] = dict(block=hb/NREP, row=hr/NREP, n=NREP)

# ---- 2  multi-BA + weather, leak-free ---------------------------------
print("\n2. Replication + weather controls, all 5 grids (fixed code)", flush=True)
ba_res = {}
for ba in ("PJM","CISO","ERCO","MISO","NYIS"):
    Xb, yb, _ = build(ba, years=range(2019,2027), q=.90)
    rb, labb, supb, ab, lpb, nulb = engine(Xb, yb, J=500, seed=11)
    nrb = shuffle_null(Xb, yb, J=500, seed=11)
    cb = np.quantile(nulb,.05); cr = np.quantile(nrb,.05)
    sb = int((lpb<=cb).sum()); sr = int((lpb<=cr).sum())
    lift = (ab/np.maximum(supb,1))/yb.mean()
    cand = [i for i in np.flatnonzero(lpb<=cb) if supb[i]>800]
    top = sorted(cand, key=lambda i:-lift[i])[:10]
    s,_w = strata(Xb.index, ba)
    mg=[];po=[]
    for i in top:
        mk=np.unpackbits(rb[i].view(np.uint8),bitorder='little')[:len(yb)].astype(bool)
        m,p,_,_=stratified_lift(mk,yb,s); mg.append(m); po.append(p)
    ba_res[ba]=dict(sig_block=sb, sig_row=sr, inflation=sr/max(sb,1),
                    lift=float(np.mean(mg)), lift_wx=float(np.mean(po)),
                    retained=float(np.mean(po)/np.mean(mg)),
                    top=" AND ".join(labb[top[0]]))
    print(f"   {ba}: sig {sb:,} vs {sr:,} (infl {sr/max(sb,1):.2f}x)  "
          f"lift {np.mean(mg):.2f} -> {np.mean(po):.2f} within weather "
          f"({100*np.mean(po)/np.mean(mg):.0f}% kept)", flush=True)
R_['grids'] = ba_res
json.dump(R_, open('reverify.json','w'), indent=1, default=float)
print("\nDONE")
