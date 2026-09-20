"""Temporal holdout, redone with ZERO look-ahead in the thresholds."""
import numpy as np, cupy as cp, pandas as pd, json
from core import pack, popcount
from fastp import logfac_table, logp_all, minp_scan
from gpu_kernel import counts_batch
from gpu_full import gather_pack, block_idx_gpu
from features import lattice

X = pd.read_parquet('lf_X.parquet'); y = np.load('lf_y.npy'); early = np.load('lf_early.npy')
print(f"train {early.sum():,}  test {(~early).sum():,}   (thresholds fitted on train only)")

def run(Xs, ys, J=600, seed=0):
    r, lab = lattice(Xs, max_arity=3, min_support=1, verbose=False)
    rg = cp.asarray(r); sup = popcount(r).astype(np.int64)
    n = len(ys); n1 = int(ys.sum()); L = logfac_table(n)
    yg = cp.asarray(ys.astype(np.uint8))
    rng = np.random.default_rng(seed); nul = np.empty(J); d = 0
    while d < J:
        b = min(32, J - d)
        ig = block_idx_gpu(n, 168, b, rng)
        c = cp.asnumpy(counts_batch(rg, gather_pack(yg, ig, n, r.shape[1], b), b,
                                    r.shape[0], r.shape[1])).astype(np.int64)
        for k in range(b): nul[d+k], _ = minp_scan(c[k], sup, n, n1, L, 0.0)
        d += b
    a = popcount(r & pack(ys)[None,:]).astype(np.int64)
    lp = logp_all(a, sup, n, n1, L)
    del rg; cp.get_default_memory_pool().free_all_blocks()
    return r, lab, sup, a, lp, nul, ys.mean()

rA, labA, supA, aA, lpA, nulA, bA = run(X[early], y[early], seed=1)
cutA = np.quantile(nulA, .05); sigA = np.flatnonzero(lpA <= cutA)
liftA = (aA/np.maximum(supA,1))/bA
print(f"discovered {len(sigA):,} significant rules on TRAIN only")
top = [i for i in sigA[np.argsort(-liftA[sigA])] if supA[i] > 500][:20]

rB, labB, supB, aB, lpB, nulB, bB = run(X[~early], y[~early], seed=2)
cutB = np.quantile(nulB, .05); idxB = {l:i for i,l in enumerate(labB)}
held, tr, te = 0, [], []
rows=[]
for i in top:
    j = idxB.get(labA[i])
    if j is None: continue
    lt = (aB[j]/max(supB[j],1))/bB
    ok = bool(lpB[j] <= cutB)
    held += ok; tr.append(liftA[i]); te.append(lt)
    rows.append((liftA[i], lt, ok, " AND ".join(labA[i])))
print(f"\nLEAK-FREE HOLDOUT: {held}/{len(tr)} top rules still significant out of sample")
print(f"  mean lift  train {np.mean(tr):.2f}  ->  test {np.mean(te):.2f}"
      f"   ({'+' if np.mean(te)>=np.mean(tr) else ''}{100*(np.mean(te)/np.mean(tr)-1):.0f}%)")
for t_, e_, ok, rl in rows[:6]:
    print(f"   {t_:5.2f} -> {e_:5.2f}  {'HELD' if ok else 'LOST'}  {rl[:62]}")
json.dump(dict(held=held, tested=len(tr), lift_train=float(np.mean(tr)),
               lift_test=float(np.mean(te))), open('holdout_clean.json','w'))
