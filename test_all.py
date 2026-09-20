"""Full regression suite.  Run this before trusting any result.
    python3 test_all.py
Covers every bug found in the audit so a regression cannot slip back in.
"""
import itertools, numpy as np, pandas as pd
from scipy.stats import hypergeom, kstest
from core import pack, popcount, Sieve, FisherTable, block_permutation
from fastp import logfac_table, logp_all, minp_scan
from features import lattice, build
FAIL = []
def ck(name, ok, d=""):
    print(f"  {'PASS' if ok else '*** FAIL':9} {name} {d}")
    if not ok: FAIL.append(name)

print("1. bitsets")
for n in (1, 63, 64, 65, 4095, 46324):
    b = np.random.default_rng(n).random(n) < .4
    ck(f"pack/popcount n={n}", int(popcount(pack(b))) == int(b.sum()))
ck("all-zero rule", int(popcount(pack(np.zeros(500, bool)))) == 0)
ck("all-one rule", int(popcount(pack(np.ones(500, bool)))) == 500)

print("\n2. Fisher tail vs scipy")
ft = FisherTable(1000, 300)
ck("tail == scipy.sf", all(abs(ft.p(x, a) - hypergeom.sf(a-1, 1000, 300, x)) < 1e-12
    for x in (5, 50, 300, 700) for a in (0, 1, min(x,300)//2, min(x,300))))
ck("psi == exhaustive min", abs(ft.psi(250) - ft.tail(250).min()) < 1e-15)

print("\n3. REGRESSION: hypergeometric support boundary (bug #1)")
g = np.random.default_rng(1); bad = 0
for t in range(300):
    n = 800; n1 = int(n*g.uniform(.05, .6)); R = 300
    sup = g.integers(5, n, size=R).astype(np.int64)
    a = np.array([g.integers(0, min(s, n1)+1) for s in sup], dtype=np.int64)
    L = logfac_table(n)
    if abs(minp_scan(a, sup, n, n1, L, 0.0)[0] - logp_all(a, sup, n, n1, L).min()) > 1e-9:
        bad += 1
ck("early exit == exhaustive min (no div-by-zero)", bad == 0, f"({bad}/300 bad)")
n, n1 = 1000, 300
errs = [abs(logp_all(np.array([a]), np.array([x]), n, n1, logfac_table(n))[0]
            - hypergeom.logsf(a-1, n, n1, x))
        for x in (700, 800, 900) for a in range(max(0, x+n1-n), min(x, n1)+1, 23)
        if np.isfinite(hypergeom.logsf(a-1, n, n1, x))]
ck("high-support regime vs scipy", max(errs) < 1e-7, f"(max {max(errs):.1e})")

print("\n4. lattice completeness vs brute force")
g = np.random.default_rng(3)
X = pd.DataFrame({f"f{i}>q50": g.random(400) > .5 for i in range(7)})
r, lab = lattice(X, max_arity=3, min_support=1, verbose=False, closed=False)
base = {c: pack(X[c].values) for c in X.columns}
brute = set()
for k in (1, 2, 3):
    for cb in itertools.combinations(X.columns, k):
        m = base[cb[0]].copy()
        for c in cb[1:]: m &= base[c]
        if popcount(m) >= 1: brute.add(cb)
ck("no rule missed", brute <= set(lab), f"({len(brute-set(lab))} missing)")
ck("no rule invented", set(lab) <= brute, f"({len(set(lab)-brute)} extra)")

print("\n5. REGRESSION: look-ahead leakage (bug #2)")
Xs, ys, _ = build('PJM', years=range(2019, 2027), q=.90, fit_until_year=2022)
Xe, ye, _ = build('PJM', years=range(2019, 2023), q=.90)
early = np.asarray(pd.to_datetime(Xs.index).year <= 2022)
common = Xs.index[early].intersection(Xe.index)
d = (Xs.loc[common].values != Xe.loc[common].values).mean()
pf = pd.Series(ys, index=Xs.index).loc[common].values
pe = pd.Series(ye, index=Xe.index).loc[common].values
ck("predicate thresholds do not see the future", d < 1e-4, f"({100*d:.4f}% differ)")
ck("outcome thresholds do not see the future", (pf != pe).mean() < 1e-4,
   f"({100*(pf!=pe).mean():.4f}% differ)")

print("\n6. adjusted p uniform under a true null")
def ar1(n, rho, g):
    e = g.standard_normal(n); x = np.empty(n); x[0] = e[0]
    for t in range(1, n): x[t] = rho*x[t-1] + np.sqrt(1-rho**2)*e[t]
    return x
pv = []
for tr in range(120):
    g = np.random.default_rng(5000+tr); n, dd = 1200, 8
    fts = np.stack([ar1(n, .9, g) > 0 for _ in range(dd)])
    ms = [pack(f) for f in fts]
    rules = np.stack(ms + [ms[i] & ms[j] for i in range(dd) for j in range(i+1, dd)])
    y = ar1(n, .9, g) > .3
    s = Sieve(rules, y, alpha=.05)
    if s.testable.sum() == 0: continue
    nul = s.minp_null(J=150, block=100, seed=tr)
    pv.append((np.sum(nul <= s.observed()[s.testable].min())+1)/(len(nul)+1))
pv = np.array(pv); ks = kstest(pv, 'uniform')
ck("uniform under null", ks.pvalue > .01, f"(KS p={ks.pvalue:.3f}, 5% rate={np.mean(pv<=.05):.3f})")

print("\n" + ("ALL REGRESSION TESTS PASSED" if not FAIL else f"FAILURES: {FAIL}"))
import sys; sys.exit(1 if FAIL else 0)
