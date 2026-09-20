"""Address all four weaknesses, with measurements."""
import json, time, numpy as np, cupy as cp, pandas as pd
from core import pack, popcount
from fastp import logfac_table, logp_all, minp_scan
from gpu_kernel import counts_batch, pack_gpu, block_perm_batch
from features import build, lattice
from weather import strata, stratified_lift
OUT = {}

def unpack(bits, n):
    return np.unpackbits(bits.view(np.uint8), bitorder='little')[:n].astype(bool)

def run(X, y, J=500, seed=0, block=168):
    r, lab = lattice(X, max_arity=3, min_support=1, verbose=False)
    rg = cp.asarray(r); sup = popcount(r).astype(np.int64)
    n = len(y); n1 = int(y.sum()); L = logfac_table(n)
    rng = np.random.default_rng(seed); nul = np.empty(J); done = 0
    while done < J:
        b = min(32, J-done)
        idx = block_perm_batch(n, block, b, rng)
        c = cp.asnumpy(counts_batch(rg, pack_gpu(y[idx]), b, r.shape[0], r.shape[1])).astype(np.int64)
        for k in range(b): nul[done+k], _ = minp_scan(c[k], sup, n, n1, L, 0.0)
        done += b
    a = popcount(r & pack(y)[None,:]).astype(np.int64)
    lp = logp_all(a, sup, n, n1, L)
    del rg; cp.get_default_memory_pool().free_all_blocks()
    return r, lab, sup, a, lp, nul

# ============ FIX 1: is it weather? ====================================
print("=== FIX 1  weather as a common cause", flush=True)
res1 = {}
for ba in ("PJM","CISO","ERCO","MISO","NYIS"):
    X, y, m = build(ba, years=range(2019,2027), q=0.90)
    r, lab, sup, a, lp, nul = run(X, y, J=400, seed=1)
    cut = np.quantile(nul, .05)
    lift = (a/np.maximum(sup,1))/y.mean()
    cand = [i for i in np.flatnonzero(lp <= cut) if sup[i] > 800]
    top = sorted(cand, key=lambda i: -lift[i])[:10]
    s, w = strata(X.index, ba)
    rows = []
    for i in top:
        mk = unpack(r[i], len(y))
        mg, pooled, obs, exp = stratified_lift(mk, y, s)
        rows.append((mg, pooled, " AND ".join(lab[i])))
    mg_m = np.mean([x[0] for x in rows]); po_m = np.mean([x[1] for x in rows])
    res1[ba] = dict(marginal=float(mg_m), pooled=float(po_m),
                    retained=float(po_m/mg_m), n_strata=int(len(np.unique(s))),
                    top=[(float(x[0]), float(x[1]), x[2]) for x in rows[:3]])
    print(f"  {ba}: marginal lift {mg_m:.2f} -> within-weather {po_m:.2f}"
          f"  ({100*po_m/mg_m:.0f}% retained, {len(np.unique(s))} strata)", flush=True)
    for mgv, pov, rl in rows[:2]:
        print(f"       {mgv:.2f} -> {pov:.2f}   {rl[:66]}", flush=True)
OUT['weather'] = res1

# ============ FIX 2: principled reduction ==============================
print("\n=== FIX 2  principled reduction (sequential conditional selection)", flush=True)
X, y, m = build('PJM', years=range(2019,2027), q=0.90)
r, lab, sup, a, lp, nul = run(X, y, J=500, seed=2)
cut = np.quantile(nul, .05); sig = np.flatnonzero(lp <= cut)
print(f"  raw significant: {len(sig):,}", flush=True)
# rank by EXCESS positives, not lift and not coverage
excess = a - sup * y.mean()
order = sig[np.argsort(-excess[sig])]
sel, remaining = [], y.copy()
from scipy.stats import hypergeom
for i in order[:5000]:
    mk = unpack(r[i], len(y))
    m_rem = mk & remaining
    if m_rem.sum() < 200: continue
    # is it still significant on the rows nobody has explained yet?
    nn = int(remaining.sum()); kk = int(m_rem.sum())
    # residual population = rows not yet covered by a selected rule
    keep_rows = ~np.zeros(len(y), bool)
    for j in sel: keep_rows &= ~unpack(r[j], len(y))
    nR = int(keep_rows.sum()); n1R = int(y[keep_rows].sum())
    xR = int((mk & keep_rows).sum()); aR = int((mk & keep_rows & y).sum())
    if xR < 200: continue
    p = hypergeom.sf(aR-1, nR, n1R, xR)
    if p < 0.05/len(sig):                     # Bonferroni over the surviving set
        sel.append(i); remaining &= ~mk
    if len(sel) >= 8: break
cov = int(y.sum() - remaining.sum())
print(f"  selected {len(sel)} rules, conditionally significant at each step")
print(f"  covering {cov:,}/{int(y.sum()):,} = {100*cov/y.sum():.0f}% of high-error hours", flush=True)
sel_rows = []
for i in sel:
    sel_rows.append(dict(rule=" AND ".join(lab[i]), lift=float((a[i]/sup[i])/y.mean()),
                         support=int(sup[i]), excess=float(excess[i])))
    print(f"    lift {sel_rows[-1]['lift']:.2f}  sup {sup[i]:6,}  {sel_rows[-1]['rule'][:64]}", flush=True)
OUT['reduction'] = dict(raw=len(sig), selected=len(sel), coverage=cov/float(y.sum()), rules=sel_rows)
json.dump(OUT, open('fix_results.json','w'), indent=1, default=float)
print("\nDONE PART A")
