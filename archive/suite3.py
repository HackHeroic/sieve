"""The tests that decide whether the finding is REAL, not just significant.

E5  temporal holdout   -- discover on 2019-2022, test the survivors on 2023-2026
E6  cross-BA replication -- rules found in PJM, do they hold in CISO/ERCO/MISO/NYIS?
E7  block-length sensitivity -- does the answer depend on an arbitrary choice?
"""
import json, time, numpy as np, cupy as cp, pandas as pd
from core import pack, popcount
from fastp import logfac_table, logp_all, minp_scan
from gpu_kernel import counts_batch, pack_gpu, block_perm_batch
from features import build, lattice
OUT = {}

def engine(X, y, block=168, J=500, seed=0, shuffle=False):
    r, lab = lattice(X, max_arity=3, min_support=1, verbose=False)
    rg = cp.asarray(r); sup = popcount(r).astype(np.int64)
    n = len(y); n1 = int(y.sum()); L = logfac_table(n)
    rng = np.random.default_rng(seed); out = np.empty(J); done = 0
    while done < J:
        b = min(32, J-done)
        idx = (np.stack([rng.permutation(n) for _ in range(b)]) if shuffle
               else block_perm_batch(n, block, b, rng))
        c = cp.asnumpy(counts_batch(rg, pack_gpu(y[idx]), b, r.shape[0], r.shape[1])).astype(np.int64)
        for k in range(b): out[done+k], _ = minp_scan(c[k], sup, n, n1, L, 0.0)
        done += b
    a = popcount(r & pack(y)[None,:]).astype(np.int64)
    lp = logp_all(a, sup, n, n1, L)
    del rg; cp.get_default_memory_pool().free_all_blocks()
    return dict(lp=lp, null=out, lab=lab, sup=sup, a=a, base=y.mean(), r=r)

# ---------------- E5 temporal holdout -----------------------------------
print("=== E5  temporal holdout (discover early, test late)", flush=True)
X, y, m = build('PJM', years=range(2019,2027), q=0.90)
t = pd.to_datetime(X.index)
tr = np.asarray(t.year <= 2022); te = ~tr
print(f"  train {tr.sum():,} hours  test {te.sum():,} hours", flush=True)
A = engine(X[tr], y[tr], J=500, seed=1)
cut = np.quantile(A['null'], .05)
sig = np.flatnonzero(A['lp'] <= cut)
print(f"  discovered {len(sig):,} significant rules in-sample", flush=True)
# rank by lift, take top 20 non-trivial
lift_tr = (A['a']/np.maximum(A['sup'],1))/A['base']
top = [i for i in sig[np.argsort(-lift_tr[sig])] if A['sup'][i] > 500][:20]
# evaluate those SAME rules out of sample
B = engine(X[te], y[te], J=500, seed=2)
lab2idx = {lab: i for i, lab in enumerate(B['lab'])}
held = []
for i in top:
    j = lab2idx.get(A['lab'][i])
    if j is None: continue
    lift_te = (B['a'][j]/max(B['sup'][j],1))/B['base']
    held.append(dict(rule=" AND ".join(A['lab'][i]), lift_train=float(lift_tr[i]),
                     lift_test=float(lift_te), sig_test=bool(B['lp'][j] <= np.quantile(B['null'],.05))))
kept = sum(h['sig_test'] for h in held)
mean_tr = np.mean([h['lift_train'] for h in held]); mean_te = np.mean([h['lift_test'] for h in held])
print(f"  of top {len(held)} rules: {kept} still significant out of sample")
print(f"  mean lift  in-sample {mean_tr:.2f}  ->  out-of-sample {mean_te:.2f}"
      f"   (shrinkage {100*(1-mean_te/mean_tr):.0f}%)", flush=True)
OUT['E5'] = dict(n_tested=len(held), n_held=kept, lift_train=mean_tr, lift_test=mean_te,
                 detail=held[:10])

# ---------------- E6 cross-BA replication -------------------------------
print("\n=== E6  cross-BA replication of PJM rules", flush=True)
P = engine(X, y, J=500, seed=3)
cutP = np.quantile(P['null'], .05)
sigP = np.flatnonzero(P['lp'] <= cutP)
liftP = (P['a']/np.maximum(P['sup'],1))/P['base']
topP = [i for i in sigP[np.argsort(-liftP[sigP])] if P['sup'][i] > 1000][:15]
rep = {}
for ba in ("CISO","ERCO","MISO","NYIS"):
    Xb, yb, mb = build(ba, years=range(2019,2027), q=0.90)
    Q = engine(Xb, yb, J=500, seed=4)
    cutQ = np.quantile(Q['null'], .05); idxQ = {l:i for i,l in enumerate(Q['lab'])}
    ok = 0; lifts = []
    for i in topP:
        j = idxQ.get(P['lab'][i])
        if j is None: continue
        lifts.append((Q['a'][j]/max(Q['sup'][j],1))/Q['base'])
        ok += int(Q['lp'][j] <= cutQ)
    rep[ba] = dict(tested=len(lifts), replicated=ok, mean_lift=float(np.mean(lifts)))
    print(f"  {ba}: {ok}/{len(lifts)} PJM rules replicate, mean lift {np.mean(lifts):.2f}", flush=True)
OUT['E6'] = rep

# ---------------- E7 block length sensitivity ---------------------------
print("\n=== E7  block-length sensitivity", flush=True)
sens = {}
for blk in (24, 72, 168, 336, 720):
    C = engine(X, y, block=blk, J=400, seed=5)
    s = int((C['lp'] <= np.quantile(C['null'], .05)).sum())
    sens[blk] = s
    print(f"  block={blk:4d}h -> {s:,} significant", flush=True)
OUT['E7'] = sens
json.dump(OUT, open('suite3_results.json','w'), indent=1, default=float)
print("\nDONE")
