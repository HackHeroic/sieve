"""Correctness audit.  Known-answer tests, not code reading."""
import numpy as np, itertools
from scipy.stats import hypergeom, kstest
from core import pack, popcount, block_permutation, row_shuffle, Sieve, tarone_threshold, FisherTable
from fastp import logfac_table, logp_all, minp_scan
FAIL=[]
def ck(name, ok, detail=""):
    print(f"  {'PASS' if ok else '*** FAIL':9} {name} {detail}"); FAIL.append(name) if not ok else None

print("A. Adjusted p-values must be UNIFORM under a true null")
print("   (the single most decisive test of the whole procedure)")
def ar1(n,rho,g):
    e=g.standard_normal(n); x=np.empty(n); x[0]=e[0]
    for t in range(1,n): x[t]=rho*x[t-1]+np.sqrt(1-rho**2)*e[t]
    return x
pvals=[]
for trial in range(200):
    g=np.random.default_rng(5000+trial)
    n,d=1500,8
    feats=np.stack([ar1(n,.9,g)>0 for _ in range(d)])
    masks=[pack(f) for f in feats]
    rules=[m for m in masks]+[masks[i]&masks[j] for i in range(d) for j in range(i+1,d)]
    rules=np.stack(rules)
    y=ar1(n,.9,g)>0.3
    s=Sieve(rules,y,alpha=0.05)
    if s.testable.sum()==0: continue
    nul=s.minp_null(J=200,block=100,seed=trial)
    obs=s.observed()[s.testable].min()
    pvals.append((np.sum(nul<=obs)+1)/(len(nul)+1))
pvals=np.array(pvals)
ksD,ksp=kstest(pvals,'uniform')
print(f"   n={len(pvals)} trials, KS D={ksD:.3f} p={ksp:.3f}")
print(f"   fraction <= 0.05 : {np.mean(pvals<=0.05):.3f}  (want ~0.05)")
print(f"   fraction <= 0.10 : {np.mean(pvals<=0.10):.3f}  (want ~0.10)")
print(f"   fraction <= 0.50 : {np.mean(pvals<=0.50):.3f}  (want ~0.50)")
ck("adjusted p is uniform under the null", ksp>0.01, f"(KS p={ksp:.3f})")

print("\nB. The early-exit bound must be ADMISSIBLE (never miss the true minimum)")
g=np.random.default_rng(1)
bad=0
for t in range(300):
    n=800; n1=int(n*g.uniform(.05,.5))
    R=400
    sup=g.integers(5,n,size=R).astype(np.int64)
    a=np.array([g.integers(0,min(s,n1)+1) for s in sup],dtype=np.int64)
    L=logfac_table(n)
    fast,_=minp_scan(a,sup,n,n1,L,0.0)
    full=logp_all(a,sup,n,n1,L).min()
    if abs(fast-full)>1e-9: bad+=1
ck("early exit equals exhaustive minimum", bad==0, f"({bad}/300 mismatches)")

print("\nC. Apriori lattice must be COMPLETE vs brute force")
import pandas as pd
from features import lattice
g=np.random.default_rng(3)
X=pd.DataFrame({f"f{i}>q50": g.random(400)>0.5 for i in range(7)})
r_ap,lab_ap=lattice(X,max_arity=3,min_support=1,verbose=False,closed=False)
cols=list(X.columns); base={c:pack(X[c].values) for c in cols}
brute=set()
for k in (1,2,3):
    for combo in itertools.combinations(cols,k):
        if len({c.split('>')[0] for c in combo})<k: continue   # same-feature excluded by design
        m=base[combo[0]].copy()
        for c in combo[1:]: m&=base[c]
        if popcount(m)>=1: brute.add(combo)
ap=set(lab_ap)
ck("apriori finds every brute-force rule", brute<=ap, f"(missing {len(brute-ap)})")
ck("apriori invents nothing extra", ap<=brute, f"(extra {len(ap-brute)})")

print("\nD. LEAKAGE: are feature thresholds computed using the test period?")
from features import build
Xf,yf,mf=build('PJM',years=range(2019,2027),q=0.90)
Xe,ye,me=build('PJM',years=range(2019,2023),q=0.90)
tf=pd.to_datetime(Xf.index); early=np.asarray(tf.year<=2022)
common=Xf.index[early].intersection(Xe.index)
Af=Xf.loc[common].values; Ae=Xe.loc[common].values
diff=(Af!=Ae).mean()
yd=(yf[early][:len(common)]!=ye[:len(common)]).mean()
print(f"   predicates differing when fitted on all-data vs early-only: {100*diff:.2f}%")
print(f"   outcome labels differing                                  : {100*yd:.2f}%")
ck("no look-ahead in predicate thresholds", diff<1e-9, f"({100*diff:.2f}% differ)")
ck("no look-ahead in outcome definition", yd<1e-9, f"({100*yd:.2f}% differ)")

print("\nE. Round trips and edge cases")
for n in (1,63,64,65,4095,46324):
    b=np.random.default_rng(n).random(n)<0.4
    ck(f"pack/popcount n={n}", int(popcount(pack(b)))==int(b.sum()))
ck("all-zero rule", int(popcount(pack(np.zeros(500,bool))))==0)
ck("all-one rule", int(popcount(pack(np.ones(500,bool))))==500)
ft=FisherTable(1000,300)
ck("psi(x) == exhaustive min over a", abs(ft.psi(250)-ft.tail(250).min())<1e-15)

print("\n"+("ALL AUDIT TESTS PASSED" if not FAIL else f"FAILURES: {FAIL}"))
