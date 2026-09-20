"""Is test 9 a bug, or correct behaviour?  Plant (0,1); see what ranks top and why."""
import numpy as np
from core import pack, popcount, Sieve

def ar1(n, rho, g):
    e = g.standard_normal(n); x = np.empty(n); x[0] = e[0]
    for t in range(1, n): x[t] = rho*x[t-1] + np.sqrt(1-rho**2)*e[t]
    return x

g = np.random.default_rng(7)
n, d = 4000, 12
feats = np.stack([ar1(n, .95, g) > 0 for _ in range(d)])
masks = [pack(f) for f in feats]
rules, labels = [], []
for i in range(d): rules.append(masks[i]); labels.append((i,))
for i in range(d):
    for j in range(i+1, d): rules.append(masks[i] & masks[j]); labels.append((i,j))
rules = np.stack(rules)

planted = feats[0] & feats[1]
y = (ar1(n, .95, g) > 0.4)
y[planted & (g.random(n) < .75)] = True

s = Sieve(rules, y, alpha=.05)
null = s.minp_null(J=300, block=168, seed=3)
adj  = s.adjusted(null)
obs  = s.observed()
a    = s.counts(s.y)

rows = [(labels[i], s.support[i], a[i], a[i]/max(s.support[i],1), obs[i], adj[i])
        for i in np.flatnonzero(s.testable)]
rows.sort(key=lambda r: r[4])
print(f"n={n}  n1={int(y.sum())}  base rate={y.mean():.3f}")
print(f"planted rows={int(planted.sum())}  testable rules={int(s.testable.sum())}\n")
print(f"{'rule':10} {'support':>8} {'a':>7} {'enrich':>7} {'raw p':>11} {'adj p':>8}")
for r in rows[:6]:
    print(f"{str(r[0]):10} {r[1]:8d} {r[2]:7d} {r[3]:7.3f} {r[4]:11.3e} {r[5]:8.4f}")
print()
sig = [r[0] for r in rows if r[5] <= .05]
print("significant set:", sig)
print("planted (0,1) in significant set:", (0,1) in sig)
