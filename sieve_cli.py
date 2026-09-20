"""
SIEVE command line entry point.

    python3 sieve_cli.py                    # default: PJM, 200 permutations
    python3 sieve_cli.py --perms 1000       # the canonical run
    python3 sieve_cli.py --naive            # also run the naive row shuffle
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from engine import Problem, load_cached


def bar(done, total, width=34):
    f = int(width * done / total)
    return "#" * f + "." * (width - f)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="sieve",
        description="Significant pattern mining with a permutation-based "
                    "family-wise error correction.")
    ap.add_argument("--perms", type=int, default=200,
                    help="permutations in the null (default 200)")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="family-wise error rate (default 0.05)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--naive", action="store_true",
                    help="also run the naive row shuffle, for comparison")
    ap.add_argument("--top", type=int, default=8,
                    help="rules to report after greedy cover")
    ap.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto",
                    help="auto picks the GPU when one is present (default auto)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    print("loading cached PJM lattice ...", end=" ", flush=True)
    rules, labels, y, X = load_cached()
    months = pd.to_datetime(X.index).month.values
    print(f"{time.time()-t0:.1f}s")

    p = Problem(rules, labels, y, months=months, alpha=args.alpha,
                device=args.device)
    print()
    print(f"  device           {p.backend.label}"
          f"   (batch {p.backend.batch})")
    print(f"  hours            {p.n:,}")
    print(f"  high-error hours {p.n1:,}  ({100*p.y.mean():.1f}%)")
    print(f"  candidate rules  {len(p.labels):,}")
    print(f"  testable         {int(p.s.testable.sum()):,}"
          f"   (Tarone prunes {100*(1-p.s.testable.mean()):.2f}%)")
    print()

    def run(kind, seed):
        t = time.time()
        state = {"t": t}

        def cb(j, J, v):
            if args.quiet:
                return
            el = time.time() - state["t"]
            eta = el / j * (J - j)
            sys.stdout.write(f"\r  {kind:5s} [{bar(j,J)}] {j}/{J}  eta {eta:4.1f}s")
            sys.stdout.flush()

        nul = p.null(args.perms, kind=kind, seed=seed, cb=cb)
        if not args.quiet:
            sys.stdout.write("\r" + " " * 70 + "\r")
        print(f"  {kind:5s} null: {args.perms} permutations in "
              f"{time.time()-t:.1f}s  ({1000*(time.time()-t)/args.perms:.0f} ms each)")
        return nul

    nb = run("block", args.seed)
    est, sd = p.count_ci(nb)
    print()
    print(f"  significant at {int(100*args.alpha)}% FWER (block permutation): "
          f"{len(p.significant(nb)):,}")
    print(f"      bootstrap spread at J={args.perms}: {est:,} +/- {sd:,}")

    if args.naive:
        nr = run("row", args.seed)
        sr = len(p.significant(nr))
        sb = max(len(p.significant(nb)), 1)
        print(f"  significant (naive row shuffle):           {sr:,}")
        print(f"      inflation: {sr/sb:.2f}x")

    print()
    print("  NON-REDUNDANT FINDINGS (greedy cover)")
    chosen, cum = p.cover(nb, top=args.top)
    print(f"  {'lift':>5} {'support':>8} {'new hrs':>8} {'log p':>9}  rule")
    for c in chosen:
        print(f"  {c['lift']:5.2f} {c['support']:8,} {c['new_hours']:8,} "
              f"{c['logp']:9.1f}  {c['rule']}")
    print(f"    -> {cum:,} of {p.n1:,} high-error hours covered "
          f"({100*cum/p.n1:.0f}%)")
    print()
    print(f"  total {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
