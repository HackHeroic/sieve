"""Are the SURPRISING rules also statistically real, or surprising by luck?

This is the exact gap our research found: Asta AutoDiscovery ranks findings by
"Bayesian surprise" with NO multiplicity control -- and surprise-ranking
actively REWARDS spurious results, because a fluke contradicts your priors by
definition.  So surprise without FWER is worse than useless.
"""
import numpy as np, cupy as cp, pandas as pd, json
from core import pack, popcount
from fastp import logfac_table, logp_all, minp_scan
from gpu_kernel import counts_batch
from gpu_full import gather_pack, block_idx_gpu
from features import build, lattice
from surprise import singleton_rates, surprise_scores, reversals

X, y, _ = build('PJM', years=range(2019,2027), q=.90)
r, lab = lattice(X, max_arity=3, min_support=1, verbose=False)
rg=cp.asarray(r); sup=popcount(r).astype(np.int64)
n=len(y); n1=int(y.sum()); L=logfac_table(n); yg=cp.asarray(y.astype(np.uint8))
def null(J, seed, shuffle=False):
    rng=np.random.default_rng(seed); out=np.empty(J); d=0
    while d<J:
        b=min(32,J-d)
        ig=(cp.asarray(np.stack([rng.permutation(n) for _ in range(b)]).astype(np.int32))
            if shuffle else block_idx_gpu(n,168,b,rng))
        c=cp.asnumpy(counts_batch(rg,gather_pack(yg,ig,n,r.shape[1],b),b,r.shape[0],r.shape[1])).astype(np.int64)
        for k in range(b): out[d+k],_=minp_scan(c[k],sup,n,n1,L,0.0)
        d+=b
    return out
nul=null(800,1); nrow=null(800,1,True)
a=popcount(r & pack(y)[None,:]).astype(np.int64)
lp=logp_all(a,sup,n,n1,L)
cutB=np.quantile(nul,.05); cutR=np.quantile(nrow,.05)
single=singleton_rates(X,y); obs,pred,surp=surprise_scores(lab,a,sup,y,single)
lift=obs/y.mean()

cand=[i for i in range(len(lab)) if sup[i]>800]
top_surp=sorted(cand,key=lambda i:-surp[i])[:50]
sigB=sum(lp[i]<=cutB for i in top_surp)
sigR=sum(lp[i]<=cutR for i in top_surp)
print(f"TOP 50 MOST SURPRISING RULES")
print(f"  survive FWER (block permutation): {sigB}/50")
print(f"  would 'survive' the naive row shuffle: {sigR}/50")
print()
# how does surprise relate to lift?
allsig=np.flatnonzero(lp<=cutB)
big=[i for i in allsig if sup[i]>800]
print(f"among {len(big):,} significant rules with support>800:")
print(f"  correlation(lift, surprise) = {np.corrcoef(lift[big],surp[big])[0,1]:+.3f}")
hi_lift=sorted(big,key=lambda i:-lift[i])[:200]
hi_surp=sorted(big,key=lambda i:-surp[i])[:200]
print(f"  mean surprise of the top-200 by LIFT     : {np.mean([surp[i] for i in hi_lift]):+.3f}")
print(f"  mean surprise of the top-200 by SURPRISE : {np.mean([surp[i] for i in hi_surp]):+.3f}")
print(f"  overlap between those two sets           : {len(set(hi_lift)&set(hi_surp))}/200")
print()
# reversals, tested
rv=reversals(lab,a,sup,y,single)
idx={l:i for i,l in enumerate(lab)}
rv_sig=[x for x in rv if lp[idx[x['rule']]]<=cutB and x['drop']>0.25]
print(f"REVERSALS: {len(rv)} found, {len(rv_sig)} also significant at 5% FWER")
seen=set(); shown=0
for x in sorted(rv_sig,key=lambda z:-z['drop']):
    if x['component'] in seen: continue
    seen.add(x['component']); shown+=1
    print(f"  '{x['component']}': {100*x['rate_alone']:.1f}% alone vs {100*x['base']:.1f}% base")
    print(f"     with    {100*x['rate_with']:.1f}%   without {100*x['rate_without']:.1f}%   -> {100*x['drop']:.0f}% LOWER in context")
    print(f"     {' AND '.join(x['rule'])[:70]}")
    if shown>=5: break
json.dump(dict(top50_sig=int(sigB), reversals=len(rv), reversals_sig=len(rv_sig)),
          open('unexpected_verified.json','w'))
