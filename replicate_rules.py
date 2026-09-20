"""
Cross-BA rule-level replication, on the SHIPPED null.

The 57/58 figure came from suite3.py E6, which drew permutations from
gpu_kernel.block_perm_batch -- unstratified, and it also shuffled the ragged
tail.  It was also cut with np.quantile.  This re-runs the same experiment
through engine.Problem, which uses core.block_permutation(strata=months) and the
order-statistic cutoff.

Design (unchanged from E6): take PJM's top rules by lift among the significant
ones with support > 1000, then ask whether each is ALSO significant on each of
the other four balancing authorities, in that grid's own lattice and against
that grid's own null.
"""
import json
import numpy as np
import pandas as pd

from engine import Problem
from features import build, lattice

J, SUP_FLOOR, TOP = 500, 1000, 15


def grid(ba, seed):
    X, y, _ = build(ba, years=range(2019, 2027), q=.90)
    r, lab = lattice(X, max_arity=3, min_support=1, verbose=False)
    months = np.asarray(pd.to_datetime(X.index).month)
    p = Problem(r, lab, y, months=months, alpha=0.05)
    return p, lab, p.null(J, kind="block", seed=seed)


pP, labP, nulP = grid("PJM", 3)
sigP = pP.significant(nulP)
cand = sigP[pP.sup[sigP] > SUP_FLOOR]
topP = cand[np.argsort(-pP.lift[cand])][:TOP]
print(f"PJM: {len(sigP):,} significant; testing its top {len(topP)} "
      f"(support > {SUP_FLOOR}) elsewhere\n", flush=True)

rep, tot_ok, tot_n = {}, 0, 0
for ba in ("CISO", "ERCO", "MISO", "NYIS"):
    q, labQ, nulQ = grid(ba, 4)
    cutQ = q.cutoff(nulQ)
    idxQ = {l: i for i, l in enumerate(labQ)}
    ok, lifts, missing = 0, [], 0
    for i in topP:
        j = idxQ.get(labP[i])
        if j is None:
            missing += 1
            continue
        lifts.append(float(q.lift[j]))
        ok += int(q.lp[j] <= cutQ)
    rep[ba] = dict(tested=len(lifts), replicated=ok, not_expressible=missing,
                   mean_lift=float(np.mean(lifts)) if lifts else None)
    tot_ok += ok; tot_n += len(lifts)
    print(f"  {ba}: {ok}/{len(lifts)} replicate   mean lift {np.mean(lifts):.2f}"
          f"{'   (' + str(missing) + ' not in lattice)' if missing else ''}", flush=True)

print(f"\nCROSS-BA REPLICATION ON THE SHIPPED NULL: {tot_ok}/{tot_n}")
json.dump(dict(per_ba=rep, replicated=tot_ok, tested=tot_n, J=J,
               stratified=True, cutoff="order-statistic",
               source_grid="PJM", support_floor=SUP_FLOOR, top_n=TOP),
          open("replication_clean.json", "w"), indent=1)
print("wrote replication_clean.json")
