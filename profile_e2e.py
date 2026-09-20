"""Where does the wall time actually go?"""
import time, numpy as np, pandas as pd
import backends
from core import block_permutation, popcount
from fastp import logfac_table, minp_scan, minp_null_fast
from features import build, lattice

X, y, _ = build("PJM", years=range(2019, 2027), q=.90)
r, lab = lattice(X, max_arity=3, min_support=1, verbose=False)
months = np.asarray(pd.to_datetime(X.index).month)
n, n1 = len(y), int(y.sum())
L = logfac_table(n); sup = popcount(r).astype(np.int64)
bk = backends.make("gpu", r, kernel="warp")
B, REP = 32, 6
print(f"{r.shape[0]:,} rules, batch {B}\n")

rng = np.random.default_rng(1)
# warm numba
_c = bk.counts(y, [block_permutation(n,168,rng,months) for _ in range(B)])
minp_scan(_c[0], sup, n, n1, L, 0.0); minp_null_fast(_c[:2], sup, n, n1, L)

t_idx = t_cnt = t_scan_serial = t_scan_par = 0.0
for _ in range(REP):
    t = time.time()
    idxs = [block_permutation(n, 168, rng, months) for _ in range(B)]
    t_idx += time.time() - t
    t = time.time(); C = bk.counts(y, idxs); t_cnt += time.time() - t
    t = time.time()
    for k in range(B): minp_scan(C[k], sup, n, n1, L, 0.0)
    t_scan_serial += time.time() - t
    t = time.time(); minp_null_fast(C, sup, n, n1, L); t_scan_par += time.time() - t

def ms(x): return x / REP / B * 1000
tot_old = ms(t_idx) + ms(t_cnt) + ms(t_scan_serial)
tot_new = ms(t_idx) + ms(t_cnt) + ms(t_scan_par)
print(f"{'stage':32s} {'ms/perm':>9} {'share':>7}")
print(f"{'permutation indices (host)':32s} {ms(t_idx):9.3f} {100*ms(t_idx)/tot_old:6.0f}%")
print(f"{'GPU counts (warp kernel)':32s} {ms(t_cnt):9.3f} {100*ms(t_cnt)/tot_old:6.0f}%")
print(f"{'min-p scan, SERIAL (current)':32s} {ms(t_scan_serial):9.3f} {100*ms(t_scan_serial)/tot_old:6.0f}%")
print(f"{'':32s} {'':>9}")
print(f"{'min-p scan, PARALLEL (unused!)':32s} {ms(t_scan_par):9.3f}")
print(f"\n  total now          {tot_old:.3f} ms/perm")
print(f"  total with prange  {tot_new:.3f} ms/perm   -> {tot_old/tot_new:.2f}x")
print(f"  scan speedup       {ms(t_scan_serial)/max(ms(t_scan_par),1e-9):.1f}x")
