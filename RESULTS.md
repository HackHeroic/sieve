# SIEVE: what the code actually does

Built and measured 2026-09-17 on PJM, real EIA-930 data, CPU only.
Everything below is a number off this machine, not an estimate.

## It works

18 of 18 correctness tests pass: bitset popcount, Fisher tail against
scipy, Tarone self consistency, permutation validity, stratified permutation.

**Calibration on synthetic data with a known answer**, 60 trials:

| null | false positive rate |
|---|---|
| circular block permutation | **0.067** (target 0.05) |
| row shuffle (what a naive implementation does) | **1.000** |

The naive shuffle fires on every single null dataset. That is the money chart
and it is real.

**Power**: a planted rule is recovered as the top hit, raw p of 2.7e-123
against 9.5e-40 for the runner up.

## On real PJM data

n = 46,324 hours, 2019 to 2024, 85 forecast-time-available predicates,
75,729 rules, outcome = top decile of absolute day ahead forecast error
within its own hour and calendar month.

| null | rules significant at 5% FWER |
|---|---|
| circular block permutation | 29,199 |
| row shuffle | 49,533 |

The naive null inflates the count by **1.7x** on real data. See
`fig_two_nulls.png`.

## Two things that contradict the brief

### Tarone pruning is worthless here. Measured 0.55%.

| min support | rules | testable | pruned |
|---|---|---|---|
| 200 | 70,366 | 70,366 | 0.0% |
| 20 | 74,985 | 74,985 | 0.0% |
| 1 | 75,729 | 75,313 | **0.55%** |

The threshold it produces is 1.00x Bonferroni, i.e. no gain at all.

**Why, structurally.** Tarone prunes patterns too *rare* to ever reach
significance. Our predicates are quantile thresholds, so each covers 25 to 75
percent of rows, and conjunctions of dense predicates stay dense: median rule
support is 1,880 of 46,324. GWAS has rare variants. Grid quantile features do
not. Lowering the outcome rate to 0.6% does not rescue it either.

**L2 comes off the ladder for this problem.**

### The early exit is the whole project, and it is not where I put it.

The brief describes L5 as early exit on the *popcount*. Measured, the popcount
was never the problem:

| step | time per permutation | share |
|---|---|---|
| popcount over 439 MB of bitsets | 0.069 s | 4% |
| hypergeometric tail (scipy) | 1.733 s | **96%** |

So the bound belongs on the *p-value tail*, not the popcount. Since
P(A>=a) >= P(A=a), the first term alone lower bounds the tail, and for minP we
only want the smallest p, so any rule whose first term already exceeds the
running best can be skipped entirely. Same admissible bound argument as
Tarone, one level down.

| | |
|---|---|
| rules needing the full tail | **67 of 75,729 (0.09%)** |
| p-value step | 1.733 s to 0.0002 s = **8,815x** |
| per permutation | 1.802 s to 0.069 s = **26x** |
| J = 10,000 | **5.01 hours to 11.5 minutes**, single core |

Verified against `scipy.stats.hypergeom.logsf` to 1e-8.

## What this does to the GPU argument

It makes it cleaner. After the CPU optimisations the loop is **popcount
bound**: 0.069 s of the 0.069 s. Bitwise AND and popcount over a 439 MB matrix,
ten thousand times, is precisely what a GPU is for, and now there is no
cheaper CPU trick hiding behind it.

The honest ladder is therefore shorter and better evidenced than the brief
claims: L0 to L1 to L3+L5 (measured 26x) to L6 (GPU, to be measured Saturday).
L2 is a measured null result. L4 was already fast.

## The finding, such as it is

Top rule: **yesterday's forecast error in the top quartile predicts today's
error in the top decile, lift 2.17**. Adding forecast day range lifts it to
2.31.

Real, forecast time available, no look ahead. But it is error *persistence*,
which is not surprising. Needs work before it is a headline.

## Two defects to fix

1. **Redundant rules.** `err_lag24>q75` and `err_lag24>q25 AND err_lag24>q75`
   are the same set, because nested quantiles on one feature subsume. The
   lattice should refuse to conjoin two thresholds of the same base feature.
   Roughly a third of the top rules are duplicates of this kind.
2. **29,199 significant rules is too many to report.** They are variants of one
   strong effect. Needs a redundancy filter, closed patterns or a greedy cover.

## Files

`core.py` engine, `fastp.py` numba p-values, `features.py` EIA-930 to lattice,
`test_core.py` 18 tests, `run_pjm.py` end to end, `fig_two_nulls.png`.
Data cached in `data/` (661 MB, all 16 files 2019 to 2026).
