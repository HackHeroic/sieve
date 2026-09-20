"""SIEVE on PJM, post-fix: deduped lattice, measured block length, greedy cover."""
import time, pickle, numpy as np, pandas as pd
from core import Sieve, block_permutation, row_shuffle, popcount
from fastp import logfac_table, logp_all, minp_scan

X = pd.read_parquet('cache_X.parquet'); y = np.load('cache_y.npy')
rules = np.load('cache_rules2.npy'); labels = pickle.load(open('cache_labels2.pkl','rb'))
s = Sieve(rules, y, alpha=0.05)
n, n1, L = s.n, s.n1, logfac_table(s.n)
sup = s.support.astype(np.int64)
months = pd.to_datetime(X.index).month.values
J = 1000

def null(permuter, strata, seed):
    rng = np.random.default_rng(seed); out = np.empty(J)
    for j in range(J):
        idx = permuter(n, block=168, rng=rng, strata=strata)
        out[j], _ = minp_scan(s.counts(s.y, idx).astype(np.int64), sup, n, n1, L, 0.0)
    return out

t0=time.time(); nb = null(block_permutation, months, 1); print(f"block null {time.time()-t0:.0f}s")
t0=time.time(); nr = null(row_shuffle, None, 1);        print(f"row null   {time.time()-t0:.0f}s")

a = s.counts(s.y).astype(np.int64)
lp = logp_all(a, sup, n, n1, L)
cut_b, cut_r = np.quantile(nb, .05), np.quantile(nr, .05)
sig_b = np.flatnonzero(lp <= cut_b); sig_r = np.flatnonzero(lp <= cut_r)
print(f"\nsignificant at 5% FWER: block {len(sig_b):,}   row shuffle {len(sig_r):,}"
      f"   inflation {len(sig_r)/max(len(sig_b),1):.2f}x")

# greedy cover over the significant set
def unpack(bits, n):
    b = np.unpackbits(bits.view(np.uint8), bitorder='little')[:n].astype(bool)
    return b
order = sig_b[np.argsort(lp[sig_b])]
remaining = y.copy(); chosen = []
pool = list(order[:4000])
for _ in range(8):
    best, gain = None, 0
    for i in pool:
        g = int((unpack(rules[i], n) & remaining).sum())
        if g > gain: best, gain = i, g
    if best is None or gain == 0: break
    chosen.append((best, gain)); remaining &= ~unpack(rules[best], n); pool.remove(best)

print(f"\nNON-REDUNDANT FINDINGS (greedy cover of {int(y.sum()):,} high-error hours)")
print(f"{'lift':>5} {'sup':>7} {'new hrs':>8} {'log p':>9}  rule")
cum = 0
for i, g in chosen:
    cum += g
    lift = (a[i]/sup[i]) / y.mean()
    print(f"{lift:5.2f} {sup[i]:7d} {g:8d} {lp[i]:9.1f}  {' AND '.join(labels[i])}")
print(f"  -> {cum:,} of {int(y.sum()):,} high-error hours covered ({100*cum/y.sum():.0f}%)")
np.savez('results2.npz', nb=nb, nr=nr, lp=lp, sup=sup, a=a)
