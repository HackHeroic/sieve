"""Correctness tests for the SIEVE core.  Run: python3 test_core.py"""
import numpy as np
from scipy.stats import hypergeom
from core import pack, popcount, FisherTable, tarone_threshold, Sieve, \
    block_permutation, row_shuffle

rng = np.random.default_rng(0)
FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}")
    if not cond:
        FAIL.append(name)


print("\n[1] bitset pack / popcount")
for n in (1, 63, 64, 65, 1000, 8761):
    b = rng.random(n) < 0.37
    check(f"popcount n={n}", int(popcount(pack(b))) == int(b.sum()))

print("\n[2] AND of bitsets equals AND of bools")
n = 5000
u, v = rng.random(n) < .4, rng.random(n) < .3
check("AND popcount", int(popcount(pack(u) & pack(v))) == int((u & v).sum()))

print("\n[3] Fisher tail matches scipy")
ft = FisherTable(n=1000, n1=300)
ok = True
for x in (5, 50, 300, 700):
    for a in (0, 1, min(x, 300) // 2, min(x, 300)):
        if abs(ft.p(x, a) - hypergeom.sf(a - 1, 1000, 300, x)) > 1e-12:
            ok = False
check("cache == scipy.hypergeom.sf", ok)

print("\n[4] psi is the minimum attainable p")
ok = all(
    abs(ft.psi(x) - ft.tail(x).min()) < 1e-15 for x in (10, 100, 400)
)
check("psi == min over a", ok)

print("\n[5] Tarone sweep terminates, threshold is self-consistent")
sup = rng.integers(1, 800, size=20000)
sigma, delta, testable = tarone_threshold(sup, ft, alpha=0.05)
k = int(testable.sum())
check("delta == alpha/k", abs(delta - 0.05 / max(k, 1)) < 1e-15,
      f"(sigma={sigma}, k={k})")
check("every testable rule can reach delta",
      all(ft.psi(int(x)) <= delta for x in np.unique(sup[testable])))
check("pruning is non-trivial", k < sup.size, f"({k} of {sup.size} kept)")

print("\n[6] block permutation preserves multiset and moves blocks")
idx = block_permutation(1000, block=24, rng=rng)
check("is a permutation", np.array_equal(np.sort(idx), np.arange(1000)))
check("not identity", not np.array_equal(idx, np.arange(1000)))

print("\n[7] stratified permutation stays inside its stratum")
strata = np.repeat(np.arange(10), 100)
idx = block_permutation(1000, block=25, rng=rng, strata=strata)
check("stratum preserved", np.array_equal(strata[idx], strata))

print("\n[8] CALIBRATION under a true null  (the one that matters)")
# real-ish autocorrelated features, outcome with NO relationship to them
def ar1(n, rho, g):
    e = g.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = rho * x[t - 1] + np.sqrt(1 - rho ** 2) * e[t]
    return x

n, d, R_TRIAL = 4000, 12, 60
hits_block = hits_row = 0
for trial in range(R_TRIAL):
    g = np.random.default_rng(1000 + trial)
    feats = np.stack([ar1(n, 0.95, g) > 0 for _ in range(d)])
    # rules = singletons + pairs
    masks = [pack(f) for f in feats]
    rules = [m for m in masks]
    for i in range(d):
        for j in range(i + 1, d):
            rules.append(masks[i] & masks[j])
    rules = np.stack(rules)
    y = ar1(n, 0.95, g) > 0.4            # autocorrelated, independent of feats

    s = Sieve(rules, y, alpha=0.05)
    if s.testable.sum() == 0:
        continue
    nb = s.minp_null(J=200, permuter=block_permutation, block=168, seed=trial)
    nr = s.minp_null(J=200, permuter=row_shuffle, seed=trial)
    obs = s.observed().min()
    hits_block += (np.mean(nb <= obs) <= 0.05)
    hits_row += (np.mean(nr <= obs) <= 0.05)

fpr_block = hits_block / R_TRIAL
fpr_row = hits_row / R_TRIAL
print(f"    block permutation FPR = {fpr_block:.3f}   (target 0.05)")
print(f"    row shuffle       FPR = {fpr_row:.3f}   (expect inflated)")
check("block permutation is calibrated", fpr_block <= 0.15,
      f"({fpr_block:.3f})")
check("row shuffle over-rejects relative to block",
      fpr_row >= fpr_block, f"({fpr_row:.3f} vs {fpr_block:.3f})")

print("\n[9] POWER: a planted rule is recovered")
g = np.random.default_rng(7)
n, d = 4000, 12
feats = np.stack([ar1(n, 0.95, g) > 0 for _ in range(d)])
masks = [pack(f) for f in feats]
rules, labels = [], []
for i in range(d):
    rules.append(masks[i]); labels.append((i,))
for i in range(d):
    for j in range(i + 1, d):
        rules.append(masks[i] & masks[j]); labels.append((i, j))
rules = np.stack(rules)

planted = feats[0] & feats[1]
base = ar1(n, 0.95, g) > 0.4
y = base.copy()
flip = planted & (g.random(n) < 0.75)
y[flip] = True

s = Sieve(rules, y, alpha=0.05)
null = s.minp_null(J=300, block=168, seed=3)
adj = s.adjusted(null)
raw = s.observed()
idx_testable = np.flatnonzero(s.testable)
# adjusted p saturates at 1/(J+1); break ties on raw p
best = min(idx_testable, key=lambda i: (adj[i], raw[i]))
found = labels[best]
sig = {labels[i] for i in idx_testable if adj[i] <= 0.05}
check("planted rule (0,1) is top hit", found == (0, 1), f"got {found}")
check("planted rule is in the significant set", (0, 1) in sig)
check("it is significant", adj[best] <= 0.05, f"(adj p = {adj[best]:.4f})")

print("\n" + ("ALL TESTS PASSED" if not FAIL else f"FAILURES: {FAIL}"))
