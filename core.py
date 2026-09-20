"""
SIEVE core: Tarone-pruned, Westfall-Young minP significant pattern mining.

Statistical contract
--------------------
  * Fisher's exact test (right tail) is used as a RANKING statistic only.
    All validity comes from the permutation distribution, never from the
    analytic p-value, because hourly rows are not independent.
  * Family-wise error rate is controlled by the Westfall-Young minP
    procedure: per permutation we keep only the SMALLEST p-value over all
    testable rules, and an observed rule's adjusted p is its rank against
    that null.  Weak control is exact under the complete null; strong
    control additionally assumes subset pivotality (standard in this
    literature).
  * Permutation is CIRCULAR BLOCK permutation, not a row shuffle, because
    hours are not exchangeable.  Validity is therefore asymptotic under
    stationarity rather than exact.

Everything here is CPU.  The GPU port replaces only `minp_null`.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import hypergeom

WORD = 64


# --------------------------------------------------------------------------
# bitsets
# --------------------------------------------------------------------------
def pack(bools: np.ndarray) -> np.ndarray:
    """(..., n) bool  ->  (..., ceil(n/64)) uint64 bitset."""
    bools = np.ascontiguousarray(bools, dtype=bool)
    n = bools.shape[-1]
    nw = (n + WORD - 1) // WORD
    pad = nw * WORD - n
    if pad:
        bools = np.concatenate(
            [bools, np.zeros(bools.shape[:-1] + (pad,), dtype=bool)], axis=-1
        )
    # view as uint64 little-endian bit order
    packed = np.packbits(bools, axis=-1, bitorder="little")
    return packed.view(np.uint64).reshape(*bools.shape[:-1], nw)


def popcount(bitset: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.bitwise_count(bitset).sum(axis=axis, dtype=np.int64)


# --------------------------------------------------------------------------
# Fisher exact, and Tarone's testability bound
# --------------------------------------------------------------------------
class FisherTable:
    """
    Right-tailed Fisher p for a 2x2 table with FIXED margins n, n1.

    Under permutation the margins never change, so p depends only on
    (support x, positives-in-rule a).  We cache the whole a-vector for each
    distinct support actually encountered, which turns the inner loop into
    an array index.
    """

    def __init__(self, n: int, n1: int):
        self.n = int(n)
        self.n1 = int(n1)
        self._cache: dict[int, np.ndarray] = {}

    def tail(self, x: int) -> np.ndarray:
        """p[a] for a = 0 .. min(x, n1), right tail."""
        x = int(x)
        hit = self._cache.get(x)
        if hit is None:
            amax = min(x, self.n1)
            a = np.arange(amax + 1)
            # P(A >= a) with A ~ Hypergeom(n, n1, x)
            hit = hypergeom.sf(a - 1, self.n, self.n1, x)
            self._cache[x] = hit
        return hit

    def p(self, x: int, a: int) -> float:
        return float(self.tail(x)[a])

    def psi(self, x: int) -> float:
        """
        Smallest p-value support x can ever attain, over any labelling.

        This is ONE scalar, so do not build the whole tail for it: the tail has
        min(x, n1) entries and the Tarone sweep asks for psi at thousands of
        distinct supports.
        """
        x = int(x)
        amax = min(x, self.n1)
        return float(hypergeom.sf(amax - 1, self.n, self.n1, x))

    def psi_many(self, xs: np.ndarray) -> np.ndarray:
        """Vectorised psi over an array of supports."""
        xs = np.asarray(xs, dtype=np.int64)
        amax = np.minimum(xs, self.n1)
        return hypergeom.sf(amax - 1, self.n, self.n1, xs)


def tarone_threshold(supports: np.ndarray, ft: FisherTable, alpha: float = 0.05):
    """
    LAMP's monotone sweep for Tarone's self-consistent threshold.

    delta = alpha / k(delta),  k(delta) = #{rules : psi(support) <= delta}

    Naive iteration from k = |H| can oscillate.  Instead sweep the support
    threshold sigma downward; k(sigma) only grows, psi(sigma) only falls,
    so the stopping test is monotone and always terminates.

    Returns (sigma_min, delta, testable_mask).
    """
    supports = np.asarray(supports, dtype=np.int64)
    uniq = np.unique(supports[supports > 0])
    if uniq.size == 0:
        return 0, alpha, np.zeros(supports.shape, dtype=bool)

    # counts by support, vectorised: k(sigma) = #rules with support >= sigma
    order = np.sort(uniq)[::-1]                       # descending support
    hist = np.bincount(supports, minlength=int(order[0]) + 1)
    cum = np.cumsum(hist[::-1])[::-1]                 # cum[x] = #(support >= x)

    psis = ft.psi_many(order)                         # one vectorised call
    ks = cum[order]
    deltas = alpha / np.maximum(ks, 1)
    bad = np.flatnonzero(psis > deltas)
    stop = int(bad[0]) if bad.size else len(order)
    if stop == 0:
        best_sigma, best_delta = int(order[0]), alpha
    else:
        best_sigma, best_delta = int(order[stop - 1]), float(deltas[stop - 1])

    testable = supports >= best_sigma
    return best_sigma, best_delta, testable


# --------------------------------------------------------------------------
# circular block permutation
# --------------------------------------------------------------------------
def block_permutation(n: int, block: int, rng: np.random.Generator,
                      strata: np.ndarray | None = None) -> np.ndarray:
    """
    Index array permuting rows in contiguous blocks.

    Hours are not exchangeable, so we shuffle whole blocks (24h, 168h, ...)
    rather than rows.  If `strata` is given (e.g. calendar month), blocks are
    only exchanged WITHIN a stratum, which stops seasonality leaking into the
    null.

    The ragged tail stays put
    -------------------------
    n is not generally a multiple of `block` (PJM: 46,324 = 275*168 + 124).  If
    that short final block joins the shuffle, then whenever it lands early
    every row after it shifts by `block - len(tail)` positions, and the output
    block boundaries no longer line up with the input ones.  The stratum
    assignment is computed on the INPUT boundaries, so the guarantee above
    silently fails: measured over 20 draws on PJM, only 3 kept every block
    inside its own stratum.  Pinning the tail restores it to 20 of 20, at the
    cost of freezing 124 of 46,324 rows (0.27%).

    Week-long blocks still straddle month ends, so about 19% of individual
    rows change calendar month under a stratified draw.  That is inherent to
    blocks longer than a day and is not what this guard is about.
    """
    idx = np.arange(n)
    nfull = (n // block) * block
    tail = idx[nfull:]                 # see below: the ragged tail stays put
    blocks = [idx[s:s + block] for s in range(0, nfull, block)]

    if strata is None:
        rng.shuffle(blocks)
        return np.concatenate(blocks + [tail])

    key = np.array([strata[b[0]] for b in blocks])
    out = [None] * len(blocks)
    for s in np.unique(key):
        pos = np.flatnonzero(key == s)
        perm = rng.permutation(pos)
        for dst, src in zip(pos, perm):
            out[dst] = blocks[src]
    return np.concatenate(out + [tail])


class BlockPermuter:
    """
    Precomputed circular block permutation.

    `block_permutation` rebuilds the same structure on every call: 275 array
    slices, a Python loop to read each block's stratum, and a per-stratum
    `zip` loop, then a 276-way concatenate.  Profiled end to end on the real
    PJM lattice that was **53% of wall time**, against 15% for the GPU kernel.

    None of that structure depends on the draw.  This caches it and reduces the
    per-call work to one `rng.permutation` per stratum plus a single vectorised
    gather.

    The RNG call sequence is IDENTICAL to `block_permutation` -- same strata in
    the same `np.unique` order, one `rng.permutation(pos)` each -- so a given
    seed yields exactly the same permutations and every cached result stays
    valid.  `test_core.py` asserts that directly.
    """

    def __init__(self, n, block, strata=None):
        self.n, self.block = int(n), int(block)
        self.nfull = (self.n // self.block) * self.block
        self.nblocks = self.nfull // self.block
        self.base = np.arange(self.nfull, dtype=np.int64).reshape(self.nblocks,
                                                                  self.block)
        self.tail = np.arange(self.nfull, self.n, dtype=np.int64)
        if strata is None:
            self.groups = None
        else:
            strata = np.asarray(strata)
            key = strata[self.base[:, 0]]          # stratum of each block
            self.groups = [np.flatnonzero(key == s) for s in np.unique(key)]
        self._map = np.empty(self.nblocks, dtype=np.int64)

    def draw(self, rng):
        if self.groups is None:
            order = rng.permutation(self.nblocks)
            out = self.base[order].ravel()
        else:
            m = self._map
            for pos in self.groups:
                m[pos] = rng.permutation(pos)
            out = self.base[m].ravel()
        return out if self.tail.size == 0 else np.concatenate([out, self.tail])


def row_shuffle(n: int, rng: np.random.Generator, **_) -> np.ndarray:
    """The naive i.i.d. shuffle.  Present so we can show it fails."""
    return rng.permutation(n)


# --------------------------------------------------------------------------
# the engine
# --------------------------------------------------------------------------
class Sieve:
    """
    rules : (R, nw) uint64  bitset per rule over n rows
    y     : (n,)   bool     binary outcome
    """

    def __init__(self, rules: np.ndarray, y: np.ndarray, alpha: float = 0.05):
        self.rules = np.ascontiguousarray(rules, dtype=np.uint64)
        self.y = np.ascontiguousarray(y, dtype=bool)
        self.n = int(y.size)
        self.n1 = int(y.sum())
        self.alpha = float(alpha)
        self.ft = FisherTable(self.n, self.n1)
        self.support = popcount(self.rules)
        self.sigma, self.delta, self.testable = tarone_threshold(
            self.support, self.ft, alpha
        )

    # ---- statistic -------------------------------------------------------
    def counts(self, y: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
        """a = popcount(rule AND y) for every rule."""
        yy = y if idx is None else y[idx]
        yb = pack(yy)
        return popcount(self.rules & yb[None, :])

    def pvals(self, a: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """
        Right-tailed Fisher p for every rule, in ONE vectorised call.

        The per-support tail cache is a trap at this scale: it issues one
        hypergeom call per distinct support (thousands of them), each building
        an array of length min(x, n1).  Asking scipy for exactly the (a, x)
        pairs we need is far cheaper.
        """
        sup = self.support
        sel = np.ones(sup.shape, bool) if mask is None else mask
        out = np.ones(sup.shape, float)
        out[sel] = hypergeom.sf(a[sel] - 1, self.n, self.n1, sup[sel])
        return out

    def observed(self) -> np.ndarray:
        return self.pvals(self.counts(self.y), self.testable)

    # ---- the permutation null -------------------------------------------
    def minp_null(self, J: int, permuter=block_permutation, block: int = 168,
                  strata: np.ndarray | None = None, seed: int = 0,
                  progress=None) -> np.ndarray:
        """
        J draws of  min over testable rules of p.
        This is the loop the GPU replaces; nothing else changes.
        """
        rng = np.random.default_rng(seed)
        out = np.empty(J, dtype=float)
        for j in range(J):
            idx = permuter(self.n, block=block, rng=rng, strata=strata)
            a = self.counts(self.y, idx)
            out[j] = self.pvals(a, self.testable).min()
            if progress and j % progress == 0:
                print(f"  perm {j}/{J}", flush=True)
        return out

    def adjusted(self, null: np.ndarray) -> np.ndarray:
        """Adjusted p = fraction of null minima at least as extreme."""
        obs = self.observed()
        srt = np.sort(null)
        rank = np.searchsorted(srt, obs, side="right")
        return (rank + 1) / (null.size + 1)

    # ---- reporting -------------------------------------------------------
    def report(self, null: np.ndarray, top: int = 20):
        """
        Significant rules, ranked correctly.

        Adjusted p cannot go below 1/(J+1), so strong rules all pile up on
        that floor.  Rank by (adjusted p, raw p) so ties at the floor are
        broken by the evidence that actually distinguishes them.
        """
        adj = self.adjusted(null)
        raw = self.observed()
        idx = np.flatnonzero(self.testable)
        order = sorted(idx, key=lambda i: (adj[i], raw[i]))
        return [
            dict(rule=int(i), support=int(self.support[i]),
                 a=int(self.counts(self.y)[i]),
                 raw_p=float(raw[i]), adj_p=float(adj[i]))
            for i in order[:top]
        ]

    @property
    def p_floor(self) -> float:
        """Smallest adjusted p a given J can express."""
        return None
