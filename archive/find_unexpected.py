import numpy as np, cupy as cp, pandas as pd, json
from core import pack, popcount
from fastp import logfac_table, logp_all, minp_scan
from gpu_kernel import counts_batch
from gpu_full import gather_pack, block_idx_gpu
from features import build, lattice
from surprise import singleton_rates, surprise_scores, reversals

X, y, _ = build('PJM', years=range(2019,2027), q=.90)
r, lab = lattice(X, max_arity=3, min_support=1, verbose=False)
rg = cp.asarray(r); sup = popcount(r).astype(np.int64)
n=len(y); n1=int(y.sum()); L=logfac_table(n); yg=cp.asarray(y.astype(np.uint8))
rng=np.random.default_rng(1); J=600; nul=np.empty(J); d=0
while d<J:
    b=min(32,J-d); ig=block_idx_gpu(n,168,b,rng)
    c=cp.asnumpy(counts_batch(rg,gather_pack(yg,ig,n,r.shape[1],b),b,r.shape[0],r.shape[1])).astype(np.int64)
    for k in range(b): nul[d+k],_=minp_scan(c[k],sup,n,n1,L,0.0)
    d+=b
a = popcount(r & pack(y)[None,:]).astype(np.int64)
lp = logp_all(a, sup, n, n1, L)
cut = np.quantile(nul,.05); sig = np.flatnonzero(lp<=cut)
print(f"significant rules: {len(sig):,}", flush=True)

single = singleton_rates(X, y)
obs, pred, surp = surprise_scores(lab, a, sup, y, single)
lift = obs/y.mean()

print(f"\nA. RANKED BY LIFT (what we were doing) -- deviation from base rate")
for i in sorted([i for i in sig if sup[i]>800], key=lambda i:-lift[i])[:5]:
    print(f"   lift {lift[i]:5.2f}  surprise {surp[i]:+5.2f}   {' AND '.join(lab[i])[:62]}")

print(f"\nB. RANKED BY SURPRISE -- departure from what the PARTS predict")
cand=[i for i in sig if sup[i]>800]
for i in sorted(cand, key=lambda i:-surp[i])[:8]:
    print(f"   surprise {surp[i]:+5.2f}  obs {obs[i]:.3f} vs parts predict {pred[i]:.3f}  (lift {lift[i]:.2f})")
    print(f"       {' AND '.join(lab[i])[:74]}")

print(f"\nC. REVERSALS -- a component whose effect FLIPS in context")
rv = reversals(lab, a, sup, y, single)
rv = [x for x in rv if x['drop']>0.25]
print(f"   found {len(rv)}")
seen=set()
for x in sorted(rv, key=lambda z:-z['drop'])[:8]:
    key=x['component']
    if key in seen: continue
    seen.add(key)
    print(f"   '{x['component']}' alone: {100*x['rate_alone']:.1f}% error (base {100*x['base']:.1f}%)")
    print(f"      but inside [{' AND '.join(x['rule'])[:56]}]")
    print(f"      rate WITH it {100*x['rate_with']:.1f}%  vs WITHOUT it {100*x['rate_without']:.1f}%   ({100*x['drop']:.0f}% lower)")
json.dump(dict(n_sig=len(sig), n_rev=len(rv)), open('unexpected.json','w'))
