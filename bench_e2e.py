"""End-to-end ms/permutation on the REAL PJM lattice, both kernels."""
import time, numpy as np, pandas as pd
import backends
from core import block_permutation, popcount
from fastp import logfac_table, minp_scan
from features import build, lattice

X, y, _ = build("PJM", years=range(2019, 2027), q=.90)
r, lab = lattice(X, max_arity=3, min_support=1, verbose=False)
months = np.asarray(pd.to_datetime(X.index).month)
n, n1 = len(y), int(y.sum())
L = logfac_table(n); sup = popcount(r).astype(np.int64)
print(f"{r.shape[0]:,} rules x {n:,} rows, {r.nbytes/1e6:.0f} MB\n")

def run(bk, J=128):
    rng = np.random.default_rng(1); out = np.empty(J); d = 0
    while d < J:
        B = min(bk.batch, J - d)
        idxs = [block_permutation(n, 168, rng, months) for _ in range(B)]
        C = bk.counts(y, idxs)
        for k in range(B):
            out[d+k], _ = minp_scan(C[k], sup, n, n1, L, 0.0)
        d += B
    return out

res = {}
for kern in ("thread", "warp"):
    bk = backends.make("gpu", r, kernel=kern)
    run(bk, 32)
    t = time.time(); v = run(bk, 128); dt = (time.time()-t)/128*1000
    res[kern] = (dt, v)
    print(f"  {kern:6s}-per-rule  {dt:6.3f} ms/perm end to end   {bk.label}")
    bk.close()

print(f"\n  identical nulls: {np.array_equal(res['thread'][1], res['warp'][1])}")
print(f"  end-to-end speedup from the kernel change: {res['thread'][0]/res['warp'][0]:.2f}x")
print(f"\n  vs CPU baseline 58.4 ms/perm:")
for k,(dt,_) in res.items():
    print(f"    {k:6s}: {58.4/dt:.0f}x end to end")
