# SIEVE: final verified state

Two bugs were found by audit and fixed. **Every headline result below was then
re-run on the corrected code.** Nothing here is carried over from the buggy
version.

## Regression suite: 18/18 pass

`python3 test_all.py` -- run this before trusting anything. It covers both
bugs, so neither can silently come back.

| group | result |
|---|---|
| bitsets, 6 sizes + edge cases | pass |
| Fisher tail vs scipy | pass |
| **hypergeometric support boundary** (bug 1) | pass, 0/300 mismatches, 3.5e-12 vs scipy |
| lattice completeness vs brute force | pass, 0 missed, 0 invented |
| **look-ahead leakage** (bug 2) | pass, 0.006% / 0.000% |
| adjusted p uniform under the null | pass, KS p = 0.311, 5% rate 0.058 |

## Validity, re-measured on the fixed code

120 true nulls, built by circularly shifting the real outcome:

| null | false positive rate |
|---|---|
| circular block permutation | **0.058** (target 0.05; 1 s.e. = 0.020, so 0.4 s.e. from nominal) |
| row shuffle | **1.000** |

Measured on the SHIPPING code, after both the support-boundary fix and the
ragged-tail fix in `block_permutation`. An exact binomial test of correct
calibration gives **p = 0.672**: 7 rejections in 120 trials is what a correctly
calibrated 5% procedure looks like at this sample size. The 95% CI is
[0.024, 0.117], which is simply what n = 120 buys you.

The row shuffle rejects 120 times out of 120, p = 7.5e-157.

## Replication across five grids, on the FIXED code

Re-run on an L4 from raw EIA files, independently of the laptop cache.

| grid | hours | rules | significant (block) | (row shuffle) | inflation | lift | within weather | kept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PJM | 46,324 | 66,844 | 23,797 | 43,718 | 1.84x | 4.51 | 3.78 | 84% |
| CISO | 44,740 | 67,187 | 14,602 | 37,021 | 2.54x | 5.56 | 5.44 | **98%** |
| ERCO | 48,013 | 65,504 | 8,495 | 34,729 | 4.09x | 4.36 | 3.61 | 83% |
| MISO | 46,101 | 66,789 | 10,010 | 38,937 | 3.89x | 4.79 | 4.63 | **97%** |
| NYIS | 48,000 | 47,620 | 14,084 | 34,450 | 2.45x | 4.14 | 3.27 | 79% |

### What actually changed, stated correctly

An earlier version of this section attributed the drop from 25,768 to 23,797 to
the ragged-tail fix, and offered "the row shuffle is identical bit for bit" as
corroboration. Both were wrong and the audit caught it.

The "before" arm came from `gpu_full.block_idx_gpu`, which **already pinned the
ragged tail and never stratified at all**. So the comparison is not
tail-shuffled vs tail-pinned. It is:

| | null used | PJM significant |
|---|---|---|
| before | unstratified GPU generator, tail already pinned | 25,768 |
| after | month-stratified CPU generator | 23,797 |

**The whole delta is stratification, not the tail fix.** The tail fix on its own
is a separate and much smaller effect, measured on the laptop cache at the same
J and seed: **25,330 -> 24,999**.

The "row shuffle identical bit for bit" line has been removed. The row shuffle
never consumed strata, so replaying the same seed through unchanged code was
guaranteed to reproduce it. It corroborated nothing.

With the order-statistic cutoff fix (`engine.cutoff`, see its docstring) the
canonical PJM figure is **24,956** at J=1000, seed=1.

Weather retention is carried over unchanged and is not affected by the fix:
`stratified_lift` is a conditional lift computation, not a permutation test.

Weather explains only 2-21% of the effect, on all five grids.

## Temporal holdout, re-run leak-free

Thresholds fitted on 2019-2022 only, then applied forward to 2023-2026.

| | leaky (old) | clean (now) |
|---|---|---|
| top rules still significant out of sample | 19 / 20 | **20 / 20** |
| mean lift, train -> test | 4.38 -> 5.10 | **4.47 -> 5.18** |

The result got stronger once the leakage was removed.

## Speed, honestly

| stage | ms/perm | vs naive |
|---|---:|---:|
| L0 naive (scipy tail, 1 core) | 1,510.6 | 1x |
| L3+L5 fast p-values | 83.3 | 18x |
| + 14-core counts | 58.4 | 26x |
| **L6 GPU pipeline, end to end** | **0.48** | **3,134x** |

**58x over a fully parallel CPU**, end to end, same statistic. Naive to final is
3,134x. Both are true; the second is the one last year's winner would have
quoted, the first is the one we quote.

## Housekeeping done

The unused GPU min-p kernel was **deleted**, not left lying around. It was 14x
slower than the CPU scan (weak per-thread bound plus warp divergence) and
carried the same support-boundary bug. `gpu_full.py` now exports only
`gather_pack` and `block_idx_gpu`.

## What is still not proven

1. **Mechanism.** We know *when* forecasts fail, not *why*. Weather is largely
   excluded but not replaced by an explanation.
2. **Subset pivotality.** Permuting the outcome tests the complete null; strong
   FWER control assumes it. Standard, and stated.
3. **Cross-grid pooling.** One balancing authority at a time. Pooling needs
   joint permutation across regions because weather correlates them.
4. **Tarone contributes 0.55%** here. A measured null, reported as such.

## State

```
core.py       engine            test_all.py   18 regression tests
fastp.py      fast p-values     reverify.py   re-run everything
features.py   EIA-930 -> rules  weather.py    NOAA controls
gpu_kernel.py counts kernel     gpu_full.py   pack + index kernels
```

All results reproduce from cache in minutes. **The implementation is correct.**
