"""
SIEVE orchestration: the single code path behind both the CLI and the demo page.

Nothing here re-implements statistics.  It calls core.py and fastp.py, which are
the audited modules, and adds only the parts that are not inference: loading a
problem, running the two nulls with a progress callback, and reporting a
non-redundant set of rules.

Why the significant COUNT is reported with an error bar
-------------------------------------------------------
The 5% cutoff is a sample quantile of a random permutation null, so the count of
rules below it is itself a random variable.  Measured on the J=1000 PJM null:

    J=100   sd  850      J=500   sd 238
    J=200   sd  655      J=1000  sd   0   (the whole null)

At J=200 the count moves by more than 1,000 between seeds.  Quoting it to five
figures from a short run would be false precision, so `count_ci` bootstraps the
null and returns the spread alongside the point estimate.
"""
from __future__ import annotations

import pickle

import numpy as np

import backends
from core import BlockPermuter, Sieve, block_permutation, row_shuffle
from fastp import logfac_table, logp_all, minp_scan

BLOCK = 168          # hours; chosen from where the ACF decays


def load_cached(prefix: str = "cache"):
    """The PJM problem as built by features.build, cached to disk."""
    import pandas as pd
    rules = np.load(f"{prefix}_rules2.npy")
    labels = pickle.load(open(f"{prefix}_labels2.pkl", "rb"))
    y = np.load(f"{prefix}_y.npy")
    X = pd.read_parquet(f"{prefix}_X.parquet")
    return rules, labels, y, X


def unpack(bits, n):
    return np.unpackbits(np.asarray(bits).view(np.uint8),
                         bitorder="little")[:n].astype(bool)


class Problem:
    """A lattice plus an outcome, ready to test."""

    def __init__(self, rules, labels, y, months=None, alpha=0.05, device="cpu"):
        self.rules, self.labels, self.y = rules, labels, y
        self.alpha = alpha
        self.months = months
        self.backend = backends.make(device, rules)
        self._permuters = {}
        self.s = Sieve(rules, y, alpha=alpha)
        self.n, self.n1 = self.s.n, self.s.n1
        self.L = logfac_table(self.n)
        self.sup = self.s.support.astype(np.int64)
        self.a = self.s.counts(self.y).astype(np.int64)
        self.lp = logp_all(self.a, self.sup, self.n, self.n1, self.L)
        self.lift = (self.a / np.maximum(self.sup, 1)) / self.y.mean()

    # ---- the permutation null -------------------------------------------
    def null(self, J, kind="block", seed=1, cb=None, batch=None):
        """
        J draws of the smallest log p over all rules.

        Permutations are drawn from `core.block_permutation` one at a time and
        then handed to the backend in batches.  The draw ORDER does not depend
        on the batch size, so a given seed yields the same J permutations
        whether the counting runs on one CPU core or on a GPU 32 at a time.
        `test_backends.py` asserts exactly that.

        kind="block" is circular block permutation stratified by calendar
        month; kind="row" is the naive i.i.d. shuffle, present so the demo can
        show it failing.
        """
        rng = np.random.default_rng(seed)
        bs = batch or self.backend.batch
        if kind == "block":
            # cached structure; draw() issues the same rng calls in the same
            # order as core.block_permutation, so results are unchanged
            key = ("block", self.n, BLOCK)
            bp = self._permuters.get(key)
            if bp is None:
                bp = self._permuters[key] = BlockPermuter(self.n, BLOCK, self.months)
            draw = lambda: bp.draw(rng)
        else:
            draw = lambda: row_shuffle(self.n, rng)
        out = np.empty(J)
        d = 0
        while d < J:
            B = min(bs, J - d)
            idxs = [draw() for _ in range(B)]
            C = self.backend.counts(self.y, idxs)
            for k in range(B):
                out[d + k], _ = minp_scan(C[k], self.sup, self.n, self.n1,
                                          self.L, 0.0)
                if cb is not None:
                    cb(d + k + 1, J, float(out[d + k]))
            d += B
        return out

    # ---- reading off the answer -----------------------------------------
    def cutoff(self, null):
        """
        The alpha-level cutoff as an ORDER STATISTIC, not an interpolated
        quantile.

        np.quantile interpolates between order statistics and implicitly divides
        by J rather than J+1.  Measured over 4,000 synthetic trials the achieved
        rate is 0.0565 at J=200 against a nominal 0.05; the order statistic below
        gives 0.0512.  The permutation literature's rule is the same one already
        used by Sieve.adjusted: the observed value competes against the J null
        draws, so the denominator is J+1.
        """
        srt = np.sort(null)
        k = int(np.floor(self.alpha * (null.size + 1))) - 1
        return float(srt[max(k, 0)])

    def significant(self, null):
        return np.flatnonzero(self.lp <= self.cutoff(null))

    def count_ci(self, null, B=300, seed=0):
        """Point estimate and sd of the significant count, by bootstrapping
        the null draws.  See the module docstring for why this is not optional."""
        rng = np.random.default_rng(seed)
        J = null.size
        cs = np.array([int((self.lp <= self.cutoff(rng.choice(null, J))).sum())
                       for _ in range(B)])
        return int(round(cs.mean())), int(round(cs.std()))

    def cover(self, null, top=8, min_support=800, min_new=40):
        """
        A non-redundant set of FINDINGS.

        Two things have to be true of a reported rule: it has to be strong, and
        it has to say something the rules above it did not already say.

        Ranking by coverage alone fails the first test.  The widest significant
        rule on PJM is `err_lag24>q25`, which covers 75% of all hours at a lift
        of 1.22 -- arithmetically significant, substantively nothing.  So the
        pool is ranked by LIFT, over rules whose support clears `min_support`
        (below that, lift is estimated off a handful of hours and is unstable).

        Redundancy is then handled the greedy way: accept a rule only if it
        covers at least `min_new` high-error hours that no accepted rule
        already covers.  That is what stops the table filling with eight
        near-copies of the same effect.
        """
        sig = self.significant(null)
        sig = sig[self.sup[sig] >= min_support]
        if sig.size == 0:
            return [], 0
        order = sig[np.argsort(-self.lift[sig])]
        remaining = self.y.copy()
        chosen, cum = [], 0
        for i in order:
            if len(chosen) >= top:
                break
            m = unpack(self.rules[i], self.n)
            gain = int((m & remaining).sum())
            if gain < min_new:
                continue
            cum += gain
            chosen.append(dict(idx=int(i),
                               rule=" AND ".join(self.labels[i]),
                               support=int(self.sup[i]),
                               lift=float(self.lift[i]),
                               logp=float(self.lp[i]),
                               new_hours=gain))
            remaining &= ~m
        return chosen, cum
