"""
Does the GPU give the same answer as the CPU?  Run on the pod: python3 test_gpu_equal.py

Self-contained on purpose.  It builds a synthetic lattice rather than loading
the 387 MB PJM cache, because the question here is whether the two BACKENDS
agree on identical input, which does not depend on the input being real.
"""
import time
import numpy as np

import backends
from core import Sieve, block_permutation, pack, popcount
from fastp import logfac_table, minp_scan

BLOCK = 168


def ar1(n, rho, g):
    e = g.standard_normal(n)
    x = np.empty(n); x[0] = e[0]
    for t in range(1, n):
        x[t] = rho * x[t-1] + np.sqrt(1 - rho**2) * e[t]
    return x


def build(n=46324, d=40, seed=0):
    g = np.random.default_rng(seed)
    feats = [ar1(n, .95, g) for _ in range(d)]
    masks, names = [], []
    for i, f in enumerate(feats):
        for q in (.25, .5, .75, .90):
            masks.append(pack(f > np.quantile(f, q))); names.append((i, q))
    rules = [m for m in masks]
    for i in range(len(masks)):
        for j in range(i+1, len(masks)):
            if names[i][0] != names[j][0]:
                rules.append(masks[i] & masks[j])
    rules = np.stack(rules)
    y = ar1(n, .95, g) > 1.28                      # ~10% positive, autocorrelated
    months = np.repeat(np.arange(12), n//12 + 1)[:n]
    return rules, y, months


FAIL = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}", flush=True)
    if not cond: FAIL.append(name)


print("building synthetic lattice ...", flush=True)
rules, y, months = build()
print(f"  {rules.shape[0]:,} rules x {y.size:,} rows "
      f"({rules.nbytes/1e6:.0f} MB)   positives {y.mean():.1%}", flush=True)

s = Sieve(rules, y, alpha=0.05)
sup = s.support.astype(np.int64); L = logfac_table(s.n)

def run(backend, J, seed):
    rng = np.random.default_rng(seed)
    out = np.empty(J); d = 0
    while d < J:
        B = min(backend.batch, J - d)
        idxs = [block_permutation(s.n, BLOCK, rng, months) for _ in range(B)]
        C = backend.counts(y, idxs)
        for k in range(B):
            out[d+k], _ = minp_scan(C[k], sup, s.n, s.n1, L, 0.0)
        d += B
    return out

print("\n[1] GPU present")
ok = backends.gpu_available()
check("cupy sees a device", ok)
if not ok:
    raise SystemExit("no GPU -- nothing further to test")

cpu = backends.CPUBackend(rules)
gpu = backends.GPUBackend(rules)
print(f"        cpu: {cpu.label}\n        gpu: {gpu.label}")

print("\n[2] counts agree exactly on the same permutations")
rng = np.random.default_rng(5)
idxs = [block_permutation(s.n, BLOCK, rng, months) for _ in range(32)]
cc, cg = cpu.counts(y, idxs), gpu.counts(y, idxs)
check("counts identical", np.array_equal(cc, cg),
      "" if np.array_equal(cc, cg) else f"max diff {np.abs(cc-cg).max()}")

print("\n[3] the full null agrees, bit for bit")
J = 64
a = run(cpu, J, 3); b = run(gpu, J, 3)
check(f"J={J} null identical", np.array_equal(a, b),
      "" if np.array_equal(a, b) else f"max diff {np.abs(a-b).max():.3e}")

print("\n[4] timing, same work both ways")
for name, bk in (("cpu", cpu), ("gpu", gpu)):
    run(bk, 8, 0)                                   # warm up
    t = time.time(); run(bk, 64, 0); dt = (time.time()-t)/64*1000
    print(f"        {name}: {dt:7.2f} ms/perm")

print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
raise SystemExit(1 if FAIL else 0)
