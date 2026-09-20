"""
The two devices must agree.  Run: python3 test_backends.py

`--device` is advertised as a speed knob.  That claim is only true if the same
seed produces the same permutations regardless of batch size, so this asserts
it directly rather than trusting the construction.
"""
import numpy as np, pandas as pd
import backends
from engine import Problem, load_cached

rules, labels, y, X = load_cached()
months = pd.to_datetime(X.index).month.values
p = Problem(rules, labels, y, months=months, device="cpu")

FAIL = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}")
    if not cond: FAIL.append(name)

print("\n[1] batch size does not change the answer (CPU)")
for kind in ("block", "row"):
    a = p.null(16, kind=kind, seed=7, batch=1)
    b = p.null(16, kind=kind, seed=7, batch=8)
    c = p.null(16, kind=kind, seed=7, batch=16)
    check(f"{kind}: batch 1 == 8", np.array_equal(a, b))
    check(f"{kind}: batch 1 == 16", np.array_equal(a, c))

print("\n[2] a different seed does change it")
check("seed 7 != seed 8",
      not np.array_equal(p.null(8, seed=7, batch=4), p.null(8, seed=8, batch=4)))

print("\n[3] stratification is actually applied")
from core import block_permutation
rng = np.random.default_rng(0)
idx = block_permutation(p.n, 168, rng, months)
starts = np.arange(0, p.n, 168)
# the guarantee: a block only ever swaps with another block of the same
# stratum, where a block's stratum is the month of its FIRST row
# the ragged tail is pinned, so output block boundaries match input ones
check("every block lands in its own stratum",
      all(np.array_equal(months[block_permutation(p.n,168,np.random.default_rng(s),months)[starts]],
                         months[starts]) for s in range(20)))
check("it is a genuine permutation", np.array_equal(np.sort(idx), np.arange(p.n)))
check("it is not the identity", not np.array_equal(idx, np.arange(p.n)))
# 168h blocks straddle month ends, so rows near a boundary do move month.
moved = float((months[idx] != months).mean())
print(f"        rows whose month changes at a block boundary: {100*moved:.1f}%")
check("row-level month leakage is the inherent block-straddle only",
      0.10 < moved < 0.30, f"({100*moved:.1f}%)")
idx2 = block_permutation(p.n, 168, np.random.default_rng(0), None)
check("unstratified permutation moves far more",
      float((months[idx2] != months).mean()) > moved * 2)

print("\n[4] GPU agrees with CPU, bit for bit")
if not backends.gpu_available():
    print("  SKIP  no GPU on this machine")
else:
    g = Problem(rules, labels, y, months=months, device="gpu")
    print(f"        backend: {g.backend.label}")
    for kind in ("block", "row"):
        a = p.null(32, kind=kind, seed=3, batch=1)
        b = g.null(32, kind=kind, seed=3)
        check(f"{kind}: CPU == GPU", np.array_equal(a, b),
              f"max diff {np.abs(a-b).max():.3e}" if not np.array_equal(a,b) else "")

print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
raise SystemExit(1 if FAIL else 0)
