"""
Surprise and reversals, re-run on the SHIPPED null.

verify_unexpected.py drew permutations from gpu_full.block_idx_gpu, which
ignored the `strata` argument, and cut with np.quantile.  This uses the
canonical J=1000 nulls in results2.npz (produced by recanon.py on the fixed,
month-stratified block_permutation) and engine.Problem's order-statistic cutoff.

The question: significance and interestingness are different things.  Does a
rule that is SURPRISING -- doing something its own components do not explain --
also survive family-wise error control?
"""
import json
import numpy as np
import pandas as pd

from engine import Problem, load_cached
from surprise import singleton_rates, surprise_scores, reversals

SUP_FLOOR = 800

rules, labels, y, X = load_cached()
p = Problem(rules, labels, y, months=np.asarray(pd.to_datetime(X.index).month))
d = np.load("results2.npz")
nb, nr = d["nb"], d["nr"]
cutB, cutR = p.cutoff(nb), p.cutoff(nr)
print(f"canonical nulls: J={nb.size}, month-stratified")
print(f"  block cutoff {cutB:.1f}   row-shuffle cutoff {cutR:.1f}\n")

single = singleton_rates(X, y)
obs, pred, surp = surprise_scores(labels, p.a, p.sup, y, single)
lift = obs / y.mean()

cand = np.flatnonzero(p.sup > SUP_FLOOR)
top_surp = cand[np.argsort(-surp[cand])][:50]
sigB = int((p.lp[top_surp] <= cutB).sum())
sigR = int((p.lp[top_surp] <= cutR).sum())
print(f"TOP 50 MOST SURPRISING RULES (support > {SUP_FLOOR})")
print(f"  survive FWER on the stratified null : {sigB}/50")
print(f"  would 'survive' the naive shuffle   : {sigR}/50")
print(f"  -> ranking by surprise alone would ship {sigR-sigB} rules that are not real\n")

big = cand[p.lp[cand] <= cutB]
hl = big[np.argsort(-lift[big])][:200]
hs = big[np.argsort(-surp[big])][:200]
ov = len(set(hl.tolist()) & set(hs.tolist()))
print(f"among {len(big):,} significant rules with support > {SUP_FLOOR}:")
print(f"  correlation(lift, surprise) = {np.corrcoef(lift[big], surp[big])[0,1]:+.3f}")
print(f"  overlap of top-200 by lift and top-200 by surprise = {ov}/200")
print(f"  -> significance and interestingness are near-orthogonal\n")

rv = reversals(labels, p.a, p.sup, y, single)
idx = {l: i for i, l in enumerate(labels)}
rv_sig = [x for x in rv if p.lp[idx[x["rule"]]] <= cutB and x["drop"] > 0.25]
print(f"REVERSALS: {len(rv):,} found, {len(rv_sig)} also significant at 5% FWER")
seen, shown = set(), 0
for x in sorted(rv_sig, key=lambda z: -z["drop"]):
    if x["component"] in seen:
        continue
    seen.add(x["component"]); shown += 1
    print(f"  '{x['component']}': {100*x['rate_alone']:.1f}% alone vs {100*x['base']:.1f}% base")
    print(f"     with {100*x['rate_with']:.1f}%  without {100*x['rate_without']:.1f}%"
          f"  -> {100*x['drop']:.0f}% LOWER in context")
    if shown >= 4:
        break

json.dump(dict(top50_surprising_significant=sigB, top50_naive=sigR,
               reversals=len(rv), reversals_significant=len(rv_sig),
               lift_surprise_corr=float(np.corrcoef(lift[big], surp[big])[0, 1]),
               top200_overlap=ov, J=int(nb.size), stratified=True,
               cutoff="order-statistic"),
          open("unexpected_clean.json", "w"), indent=1)
print("\nwrote unexpected_clean.json")
