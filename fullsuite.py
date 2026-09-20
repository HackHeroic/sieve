"""
SIEVE full validation suite on GPU.

Runs the experiments that were previously too expensive on CPU:
  E1  full J=10,000 run, both nulls
  E2  FWER calibration -- many null datasets, measured false-positive rate
  E3  power curve -- planted signal across effect sizes
  E4  the ladder, all rungs, one machine
"""
import json, time, sys
import numpy as np, cupy as cp, pandas as pd
from core import pack, popcount, block_permutation, row_shuffle
from fastp import logfac_table, logp_all, minp_scan
from gpu_kernel import counts_batch, pack_gpu, block_perm_batch
from features import lattice

OUT = {}
rules = np.load('rules.npy'); y0 = np.load('y.npy')
R, W = rules.shape; n = len(y0)
rg = cp.asarray(rules)
print(f"rules {R:,} x {W} words, {rg.nbytes/1e6:.0f} MB resident", flush=True)
sup = popcount(rules).astype(np.int64)

def gpu_minp(y, J, block, B=32, seed=0, shuffle=False):
    """J draws of min-p, counts on GPU, p-values on CPU."""
    rng = np.random.default_rng(seed)
    n1 = int(y.sum()); L = logfac_table(n)
    out = np.empty(J)
    done = 0
    while done < J:
        b = min(B, J - done)
        if shuffle:
            idx = np.stack([rng.permutation(n) for _ in range(b)])
        else:
            idx = block_perm_batch(n, block, b, rng)
        ys = pack_gpu(y[idx])
        c = cp.asnumpy(counts_batch(rg, ys, b, R, W)).astype(np.int64)
        for k in range(b):
            out[done + k], _ = minp_scan(c[k], sup, n, n1, L, 0.0)
        done += b
    return out

# ---------------------------------------------------------------- E1
print("\n=== E1  full J=10,000, both nulls", flush=True)
t0 = time.time(); nb = gpu_minp(y0, 10000, 168, seed=1); t_b = time.time()-t0
t0 = time.time(); nr = gpu_minp(y0, 10000, 168, seed=1, shuffle=True); t_r = time.time()-t0
L = logfac_table(n)
lp = logp_all(popcount(rules & pack(y0)[None,:]).astype(np.int64), sup, n, int(y0.sum()), L)
cb, cr = np.quantile(nb, .05), np.quantile(nr, .05)
sb, sr = int((lp <= cb).sum()), int((lp <= cr).sum())
print(f"  block null {t_b:.0f}s  row null {t_r:.0f}s")
print(f"  5% cutoff: block {cb:.1f}  row {cr:.1f}")
print(f"  significant: block {sb:,}  row {sr:,}  inflation {sr/max(sb,1):.2f}x")
OUT['E1'] = dict(t_block=t_b, cut_block=float(cb), cut_row=float(cr), sig_block=sb, sig_row=sr)

# ---------------------------------------------------------------- E2
print("\n=== E2  FWER calibration on TRUE nulls", flush=True)
# null datasets: circular-shift the real outcome far from its true alignment
NREP = 120
hits_b = hits_r = 0
t0 = time.time()
for rep in range(NREP):
    shift = int(n * (0.15 + 0.7 * rep / NREP))
    yn = np.roll(y0, shift)
    nbn = gpu_minp(yn, 300, 168, seed=100+rep)
    nrn = gpu_minp(yn, 300, 168, seed=100+rep, shuffle=True)
    obs = logp_all(popcount(rules & pack(yn)[None,:]).astype(np.int64),
                   sup, n, int(yn.sum()), L).min()
    hits_b += (np.mean(nbn <= obs) <= 0.05)
    hits_r += (np.mean(nrn <= obs) <= 0.05)
    if rep % 30 == 0: print(f"    rep {rep}/{NREP}", flush=True)
fb, fr = hits_b/NREP, hits_r/NREP
print(f"  block permutation FWER = {fb:.3f}   (target 0.05)")
print(f"  row shuffle       FWER = {fr:.3f}")
print(f"  took {time.time()-t0:.0f}s for {NREP} replicates")
OUT['E2'] = dict(n_rep=NREP, fwer_block=fb, fwer_row=fr)

# ---------------------------------------------------------------- E3
print("\n=== E3  power curve", flush=True)
rng = np.random.default_rng(7)
target = int(np.argmax(sup * (sup < n*0.35)))       # a mid-support rule to plant on
tmask = np.unpackbits(rules[target].view(np.uint8), bitorder='little')[:n].astype(bool)
base = y0.mean()
curve = []
for delta in (0.02, 0.05, 0.10, 0.20, 0.35, 0.50):
    rec = 0; NS = 12
    for s in range(NS):
        g = np.random.default_rng(900+s)
        yp = np.roll(y0, int(n*0.4))                 # clean background
        flip = tmask & (g.random(n) < delta)
        yp = yp.copy(); yp[flip] = True
        nn = gpu_minp(yp, 300, 168, seed=s)
        o = logp_all(popcount(rules & pack(yp)[None,:]).astype(np.int64),
                     sup, n, int(yp.sum()), L)
        cut = np.quantile(nn, .05)
        sig = np.flatnonzero(o <= cut)
        rec += int(target in sig)
    curve.append((delta, rec/NS))
    print(f"    delta={delta:.2f}  recovery={rec/NS:.2f}", flush=True)
OUT['E3'] = curve

json.dump(OUT, open('suite_results.json','w'), indent=1, default=float)
print("\nWROTE suite_results.json")
