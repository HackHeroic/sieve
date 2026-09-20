"""
Unexpectedness, not just significance.

Our lift measures deviation from the MARGINAL base rate.  That makes broad
rules look impressive and, worse, it makes a rule look interesting when it is
merely a restatement of its own parts: if A raises the error rate and B raises
the error rate, nobody is surprised that A AND B raises it.

De Bie's FORSIED framework (subjective interestingness, 2010) says: score a
pattern by how much it departs from an explicit BACKGROUND MODEL of what the
analyst already expects -- not from a flat prior.

Background model here: the analyst already knows every singleton effect.  The
natural maximum-entropy completion consistent with those marginals is additive
in log-odds:

    logit(p_hat[A and B and C]) = logit(p0) + sum_i [ logit(p[X_i]) - logit(p0) ]

SURPRISE = observed logit - predicted logit.

A rule with high surprise is doing something its own components do not explain.
That is an interaction, and it is unexpected by construction.

REVERSAL is the sharpest special case: a component whose effect flips sign in
context.  Simpson's-paradox-shaped, and inherently counterintuitive.
"""
import numpy as np

EPS = 1e-9

def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def singleton_rates(X, y):
    """
    p[predicate] = P(Y=1 | predicate true), for every column.

    A predicate that is never true has no conditional rate.  Taking the mean of
    an empty slice returns NaN with a RuntimeWarning, and that NaN then
    propagates silently through every surprise score of every rule containing
    it.  Such predicates exist here: `hour_sin`/`hour_cos` are constant within
    each (hour x month) binarisation cell, so all eight of their threshold
    predicates are identically False.

    Fall back to the base rate, which is the correct MaxEnt answer when the
    predicate carries no information, and report how many were affected.
    """
    base = float(y.mean())
    out, empty = {}, []
    for c in X.columns:
        m = X[c].values.astype(bool)
        if not m.any():
            out[c] = base
            empty.append(c)
        else:
            out[c] = float(y[m].mean())
    if empty:
        print(f"  note: {len(empty)} predicate(s) never true, using base rate "
              f"({empty[0]}{', ...' if len(empty) > 1 else ''})")
    return out


def surprise_scores(labels, a, sup, y, single):
    """
    For each rule: observed rate, rate predicted from its parts, and surprise.

    Returns arrays aligned with `labels`.
    """
    p0 = float(y.mean())
    l0 = _logit(p0)
    obs = a / np.maximum(sup, 1)
    pred_logit = np.empty(len(labels))
    for i, lab in enumerate(labels):
        s = l0
        for c in lab:
            s += _logit(single[c]) - l0
        pred_logit[i] = s
    pred = 1.0 / (1.0 + np.exp(-pred_logit))
    return obs, pred, _logit(obs) - pred_logit


def reversals(labels, a, sup, y, single, min_flip=0.15):
    """
    Rules where a component's effect REVERSES in context.

    A component that raises the outcome rate on its own, but whose presence in
    this conjunction is associated with a LOWER rate than the conjunction
    without it, is a genuine contradiction of the marginal story.
    """
    p0 = float(y.mean())
    obs = a / np.maximum(sup, 1)
    idx = {lab: i for i, lab in enumerate(labels)}
    out = []
    for i, lab in enumerate(labels):
        if len(lab) < 2:
            continue
        for c in lab:
            rest = tuple(x for x in lab if x != c)
            j = idx.get(rest)
            if j is None or sup[j] < 200 or sup[i] < 200:
                continue
            solo_up = single[c] > p0          # component raises the rate alone
            ctx_dn = obs[i] < obs[j]          # but lowers it inside this rule
            if solo_up and ctx_dn and (obs[j] - obs[i]) / max(obs[j], EPS) > min_flip:
                out.append(dict(rule=lab, component=c,
                                rate_alone=single[c], base=p0,
                                rate_with=float(obs[i]), rate_without=float(obs[j]),
                                drop=float((obs[j]-obs[i])/max(obs[j], EPS))))
    return out
