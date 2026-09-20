"""
Temporal holdout on the SHIPPED permutation.

What was wrong with redo_holdout.py
-----------------------------------
It drew permutations from `gpu_full.block_idx_gpu`, which takes no `strata`
argument at all, so the null it used was NOT the month-stratified circular block
permutation every document describes.  It also took the cutoff with
`np.quantile`, which interpolates and divides by J rather than J+1.

This version routes through `engine.Problem`, which uses
`core.block_permutation(strata=months)` and the order-statistic cutoff, so the
number it produces is a result of the method we actually ship.

Thresholds were already fitted on train rows only (lf_* files, built with
features.build(fit_until_year=2022)); that part was correct and is unchanged.
"""
import json
import numpy as np
import pandas as pd

from engine import Problem
from features import lattice

J, SUPPORT_FLOOR, TOP = 500, 500, 20

X = pd.read_parquet("lf_X.parquet")
y = np.load("lf_y.npy")
early = np.load("lf_early.npy")
months = np.asarray(pd.to_datetime(X.index).month)

print(f"train {early.sum():,} rows ({X.index[early][0].date()} to {X.index[early][-1].date()})")
print(f"test  {(~early).sum():,} rows ({X.index[~early][0].date()} to {X.index[~early][-1].date()})")
print("thresholds fitted on train only; null is month-stratified block permutation\n")


def fit(mask, seed):
    r, lab = lattice(X[mask], max_arity=3, min_support=1, verbose=False)
    p = Problem(r, lab, y[mask], months=months[mask], alpha=0.05)
    nul = p.null(J, kind="block", seed=seed)
    return p, lab, nul


pA, labA, nulA = fit(early, 1)
sigA = pA.significant(nulA)
print(f"TRAIN: {len(sigA):,} significant of {len(labA):,} rules")

cand = sigA[pA.sup[sigA] > SUPPORT_FLOOR]
top = cand[np.argsort(-pA.lift[cand])][:TOP]
print(f"       taking top {len(top)} by lift with support > {SUPPORT_FLOOR}\n")

pB, labB, nulB = fit(~early, 2)
print(f"TEST:  {len(pB.significant(nulB)):,} significant of {len(labB):,} rules")
cutB = pB.cutoff(nulB)
idxB = {l: i for i, l in enumerate(labB)}

held, tr, te, rows = 0, [], [], []
for i in top:
    j = idxB.get(labA[i])
    if j is None:                      # rule not expressible in the test lattice
        continue
    ok = bool(pB.lp[j] <= cutB)
    held += ok
    tr.append(float(pA.lift[i])); te.append(float(pB.lift[j]))
    rows.append((pA.lift[i], pB.lift[j], ok, " AND ".join(labA[i])))

mt, me = float(np.mean(tr)), float(np.mean(te))
print(f"\nHOLDOUT ON THE SHIPPED NULL: {held}/{len(tr)} top rules still significant")
print(f"  mean lift  train {mt:.2f} -> test {me:.2f}   ({100*(me/mt-1):+.0f}%)\n")
for t_, e_, ok, rl in rows[:8]:
    print(f"   {t_:5.2f} -> {e_:5.2f}  {'HELD' if ok else 'LOST'}  {rl[:64]}")

json.dump(dict(held=held, tested=len(tr), lift_train=mt, lift_test=me, J=J,
               stratified=True, cutoff="order-statistic",
               train_rows=int(early.sum()), test_rows=int((~early).sum()),
               test_window=f"{X.index[~early][0].date()} to {X.index[~early][-1].date()}"),
          open("holdout_clean.json", "w"), indent=1)
print("\nwrote holdout_clean.json")
