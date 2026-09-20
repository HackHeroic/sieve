"""Finer power curve + multi-BA generalisation + the full ladder."""
import time, json, numpy as np, cupy as cp, pandas as pd
from core import pack, popcount
from fastp import logfac_table, logp_all, minp_scan
from gpu_kernel import counts_batch, pack_gpu, block_perm_batch

rules = np.load('rules.npy'); y0 = np.load('y.npy')
R, W = rules.shape; n = len(y0); rg = cp.asarray(rules)
sup = popcount(rules).astype(np.int64); L = logfac_table(n)
OUT = {}

def gpu_minp(y, J, B=32, seed=0, shuffle=False):
    rng = np.random.default_rng(seed); n1 = int(y.sum()); out = np.empty(J); done = 0
    while done < J:
        b = min(B, J-done)
        idx = (np.stack([rng.permutation(n) for _ in range(b)]) if shuffle
               else block_perm_batch(n, 168, b, rng))
        c = cp.asnumpy(counts_batch(rg, pack_gpu(y[idx]), b, R, W)).astype(np.int64)
        for k in range(b): out[done+k], _ = minp_scan(c[k], sup, n, n1, L, 0.0)
        done += b
    return out

def obs_lp(y):
    return logp_all(popcount(rules & pack(y)[None,:]).astype(np.int64),
                    sup, n, int(y.sum()), L)

# ---- finer power sweep -------------------------------------------------
print("=== power curve, fine grid", flush=True)
target = int(np.argmax(sup * (sup < n*0.35)))
tmask = np.unpackbits(rules[target].view(np.uint8), bitorder='little')[:n].astype(bool)
print(f"  planted rule support = {sup[target]:,} ({100*sup[target]/n:.1f}% of rows)")
curve = []
for delta in (0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.060):
    rec = 0; NS = 20
    for s in range(NS):
        g = np.random.default_rng(900+s)
        yp = np.roll(y0, int(n*0.4)).copy()
        yp[tmask & (g.random(n) < delta)] = True
        cut = np.quantile(gpu_minp(yp, 300, seed=s), .05)
        rec += int(obs_lp(yp)[target] <= cut)
    curve.append((delta, rec/NS)); print(f"    d={delta:.3f} recovery={rec/NS:.2f}", flush=True)
OUT['power'] = curve

# ---- multi-BA ----------------------------------------------------------
print("\n=== generalisation across balancing authorities", flush=True)
from features import build, lattice
res = {}
for ba in ("CISO", "ERCO", "MISO", "NYIS"):
    try:
        X, y, m = build(ba, years=range(2019, 2027), q=0.90)
        r2, lab2 = lattice(X, max_arity=3, min_support=1, verbose=False)
        rg2 = cp.asarray(r2); sup2 = popcount(r2).astype(np.int64)
        n2 = len(y); L2 = logfac_table(n2)
        def gm(J, seed=0, shuffle=False):
            rng = np.random.default_rng(seed); out = np.empty(J); done = 0
            while done < J:
                b = min(32, J-done)
                idx = (np.stack([rng.permutation(n2) for _ in range(b)]) if shuffle
                       else block_perm_batch(n2, 168, b, rng))
                c = cp.asnumpy(counts_batch(rg2, pack_gpu(y[idx]), b, r2.shape[0], r2.shape[1])).astype(np.int64)
                for k in range(b): out[done+k], _ = minp_scan(c[k], sup2, n2, int(y.sum()), L2, 0.0)
                done += b
            return out
        nb = gm(500, 1); nr = gm(500, 1, True)
        o = logp_all(popcount(r2 & pack(y)[None,:]).astype(np.int64), sup2, n2, int(y.sum()), L2)
        sb = int((o <= np.quantile(nb,.05)).sum()); sr = int((o <= np.quantile(nr,.05)).sum())
        best = int(np.argmin(o)); a_best = int(popcount(r2[best:best+1] & pack(y)[None,:])[0])
        lift = (a_best/sup2[best])/y.mean()
        res[ba] = dict(n=n2, rules=int(r2.shape[0]), sig_block=sb, sig_row=sr,
                       inflation=sr/max(sb,1), top_lift=float(lift),
                       top_rule=" AND ".join(lab2[best]))
        print(f"  {ba}: n={n2:,} rules={r2.shape[0]:,} sig {sb:,} vs {sr:,} "
              f"(infl {sr/max(sb,1):.2f}x) top lift {lift:.2f}", flush=True)
        print(f"       {res[ba]['top_rule'][:90]}", flush=True)
        del rg2
        cp.get_default_memory_pool().free_all_blocks()
    except Exception as e:
        print(f"  {ba}: FAILED {type(e).__name__}: {e}", flush=True)
OUT['multi_ba'] = res
json.dump(OUT, open('suite2_results.json','w'), indent=1, default=float)
print("\nDONE")
