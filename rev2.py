"""Reversals, tested with a TWO-SIDED null -- depletion needs the left tail."""
import numpy as np, cupy as cp, json
from core import pack, popcount
from fastp import logfac_table, logp_all, logp_all_left, minp_scan
from gpu_kernel import counts_batch
from gpu_full import gather_pack, block_idx_gpu
from features import build, lattice
from surprise import singleton_rates, surprise_scores, reversals

X, y, _ = build('PJM', years=range(2019,2027), q=.90)
r, lab = lattice(X, max_arity=3, min_support=1, verbose=False)
rg=cp.asarray(r); sup=popcount(r).astype(np.int64)
n=len(y); n1=int(y.sum()); L=logfac_table(n); yg=cp.asarray(y.astype(np.uint8))

# TWO-SIDED null: per permutation take the most extreme tail in EITHER direction
J=800; rng=np.random.default_rng(3); nul=np.empty(J); d=0
while d<J:
    b=min(32,J-d); ig=block_idx_gpu(n,168,b,rng)
    c=cp.asnumpy(counts_batch(rg,gather_pack(yg,ig,n,r.shape[1],b),b,r.shape[0],r.shape[1])).astype(np.int64)
    for k in range(b):
        right,_=minp_scan(c[k],sup,n,n1,L,0.0)
        left=logp_all_left(c[k],sup,n,n1,L).min()
        nul[d+k]=min(right,left)
    d+=b
cut=np.quantile(nul,.05)
a=popcount(r & pack(y)[None,:]).astype(np.int64)
lpR=logp_all(a,sup,n,n1,L); lpL=logp_all_left(a,sup,n,n1,L)
two=np.minimum(lpR,lpL)
print(f"two-sided 5% cutoff (log p): {cut:.1f}")
print(f"significant, enrichment : {int((lpR<=cut).sum()):,}")
print(f"significant, DEPLETION  : {int((lpL<=cut).sum()):,}   <- rules with LESS error than chance")
single=singleton_rates(X,y)
rv=reversals(lab,a,sup,y,single)
idx={l:i for i,l in enumerate(lab)}
rv_sig=[x for x in rv if lpL[idx[x['rule']]]<=cut and x['drop']>0.25]
print(f"\nREVERSALS: {len(rv)} candidates, {len(rv_sig)} significant under the LEFT tail")
seen=set(); shown=0
for x in sorted(rv_sig,key=lambda z:-z['drop']):
    if x['component'] in seen: continue
    seen.add(x['component']); shown+=1
    i=idx[x['rule']]
    print(f"  '{x['component']}' raises error to {100*x['rate_alone']:.1f}% on its own (base {100*x['base']:.1f}%)")
    print(f"     but in [{' AND '.join(x['rule'])[:62]}]")
    print(f"     error is {100*x['rate_with']:.1f}% WITH it vs {100*x['rate_without']:.1f}% without  ({100*x['drop']:.0f}% lower)")
    print(f"     support {sup[i]:,}  log p(depletion) {lpL[i]:.1f}")
    if shown>=5: break
json.dump(dict(enrich=int((lpR<=cut).sum()), deplete=int((lpL<=cut).sum()),
               rev=len(rv), rev_sig=len(rv_sig)), open('rev2.json','w'))
