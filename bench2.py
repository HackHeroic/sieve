"""End-to-end after the cached permuter."""
import time, numpy as np, pandas as pd
from engine import Problem
from features import build, lattice

X, y, _ = build("PJM", years=range(2019, 2027), q=.90)
r, lab = lattice(X, max_arity=3, min_support=1, verbose=False)
months = np.asarray(pd.to_datetime(X.index).month)
p = Problem(r, lab, y, months=months, device="gpu")
print(f"{r.shape[0]:,} rules, {p.backend.label}\n")
p.null(32, kind="block", seed=0)            # warm
for J in (128,):
    t = time.time(); v = p.null(J, kind="block", seed=1); dt = (time.time()-t)/J*1000
    print(f"  end to end: {dt:.3f} ms/perm   (was 0.757 before the permuter fix)")
    print(f"  vs CPU baseline 58.4 ms/perm: {58.4/dt:.0f}x")
# identical to the serial permuter?
from core import block_permutation
from fastp import logfac_table, minp_scan
from core import popcount, pack
rng = np.random.default_rng(1)
L = logfac_table(p.n); sup = p.sup
ref = np.empty(16)
for j in range(16):
    idx = block_permutation(p.n, 168, rng, months)
    c = p.backend.counts(y, [idx])[0]
    ref[j], _ = minp_scan(c, sup, p.n, p.n1, L, 0.0)
print(f"\n  identical to the old permuter: {np.array_equal(ref, v[:16])}")
