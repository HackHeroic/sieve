"""End-to-end SIEVE run on PJM, both nulls, with the finding."""
import time, pickle, numpy as np, pandas as pd
from features import lattice
from core import Sieve, block_permutation, row_shuffle
from fastp import logfac_table, logp_all, minp_scan

X = pd.read_parquet('cache_X.parquet'); y = np.load('cache_y.npy')
rules, labels = lattice(X, max_arity=3, min_support=1, verbose=False)
s = Sieve(rules, y, alpha=0.05)
n, n1, L = s.n, s.n1, logfac_table(s.n)
sup = s.support.astype(np.int64)
J = 1000

def null(permuter, block, strata, seed):
    rng = np.random.default_rng(seed)
    out = np.empty(J)
    for j in range(J):
        idx = permuter(n, block=block, rng=rng, strata=strata)
        a = s.counts(s.y, idx).astype(np.int64)
        out[j], _ = minp_scan(a, sup, n, n1, L, 0.0)
    return out

months = pd.to_datetime(X.index).month.values
t0 = time.time(); nb = null(block_permutation, 168, months, 1); tb = time.time()-t0
t0 = time.time(); nr = null(row_shuffle, 168, None, 1);        tr = time.time()-t0
print(f"block permutation null: {tb:.1f}s  ({tb/J*1000:.0f} ms/perm)")
print(f"row shuffle null      : {tr:.1f}s")

a_obs = s.counts(s.y).astype(np.int64)
lp_obs = logp_all(a_obs, sup, n, n1, L)
best = int(np.argmin(lp_obs))

def adj(obs_lp, nulldist):
    return (np.sum(nulldist <= obs_lp) + 1) / (len(nulldist) + 1)

print(f"\nnull minima (log p): block median {np.median(nb):.1f} | row median {np.median(nr):.1f}")
print(f"  block permutation null is {np.median(nb)-np.median(nr):+.1f} log-units vs row shuffle")

order = np.argsort(lp_obs)[:10]
print(f"\nTOP RULES (n={n:,}, base rate {y.mean():.3f}, {len(rules):,} rules)")
print(f"{'raw log p':>11} {'adj p':>8} {'sup':>7} {'lift':>6}  rule")
for i in order:
    lift = (a_obs[i]/sup[i]) / y.mean()
    print(f"{lp_obs[i]:11.1f} {adj(lp_obs[i], nb):8.4f} {sup[i]:7d} {lift:6.2f}  {' AND '.join(labels[i])}")

sig = int(np.sum([adj(lp_obs[i], nb) <= 0.05 for i in range(len(lp_obs))]))
sig_row = int(np.sum([adj(lp_obs[i], nr) <= 0.05 for i in range(len(lp_obs))]))
print(f"\nSIGNIFICANT at 5% FWER:  block permutation {sig:,}   row shuffle {sig_row:,}")
np.savez('results_pjm.npz', nb=nb, nr=nr, lp=lp_obs, sup=sup, a=a_obs)
pickle.dump(labels, open('cache_labels.pkl','wb'))
