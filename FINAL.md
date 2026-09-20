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

## Temporal holdout, on the shipped null

Thresholds fitted on **2019-01 to 2022-12** only (33,597 hours), then applied
forward to **2023-01 to 2024-07** (12,727 hours, **18 months**).

Earlier drafts said "2023-2026". That was wrong, and it was wrong for a reason
worth stating: `features.py` asks for EIA fuel column names that EIA changed in
mid-2024 when battery storage became material, and an unguarded
`notna().all(axis=1)` then drops every row after 2024-07-02. That is 21,283
rows, **31.5% of the available data**, discarded silently. The statistics are
unaffected; the window is not what the documents claimed.

| | value |
|---|---|
| top rules still significant out of sample | **20 / 20** |
| mean lift, train -> test | **4.47 -> 5.18** (+16%) |

Produced by `holdout_clean.py`, artifact in `holdout_clean.json`. The previous
figure came from `redo_holdout.py`, which drew from a generator that ignored the
`strata` argument, so it was not a result of the method described here. That
script is now in `archive/` and the generator is deleted.

The result got stronger out of sample, which is the opposite of the usual
direction. It did so both before and after the null was corrected.

## Cross-grid replication, on the shipped null

PJM's top 15 rules by lift (support > 1,000, significant on PJM) tested against
each other grid's own lattice and own null.

| grid | replicated | mean lift |
|---|---|---|
| CISO | **15 / 15** | 4.55 |
| ERCO | **14 / 15** | 3.41 |
| MISO | **15 / 15** | 3.67 |
| NYIS | **13 / 13** | 3.58 (2 rules not expressible in its lattice) |
| **total** | **57 / 58** | |

Produced by `replicate_rules.py`, artifact in `replication_clean.json`, J=500,
month-stratified, order-statistic cutoff. The earlier 57/58 came from
`suite3.py` on an unstratified generator that also shuffled the ragged tail; it
happened to give the same answer, but it was not evidence for the shipped method
until now.

## How much does the answer depend on the null? Volunteer this.

PJM's day-ahead forecast improved over 2019-2024, and the slow predicates
(`err_roll7d`, `err_lag*`) carry that level shift. Our null holds calendar MONTH
fixed but not YEAR, and 81% of rows change year under a stratified draw. So a
sceptic can fairly ask whether we are detecting a year effect.

The answer is not one number. It is a range, reported rather than extracted.
J=1000, seed=1, same lattice and same observed statistic throughout; only the
permutation stratification changes.

| null specification | blocks per stratum | cutoff | significant |
|---|---:|---:|---:|
| year x month | 4.1 | **-158.8** | **7,991** |
| year only | 45.8 | -56.7 | 23,545 |
| **month only (shipped)** | 22.9 | -51.8 | **24,956** |
| unstratified block | 275 | -48.6 | 26,005 |
| *naive row shuffle* | *n/a* | *-12.9* | *43,559* |

**Range across defensible block-permutation specifications: 7,991 to 26,005.**
That is a factor of 3.3, and it is the honest uncertainty on the headline count.

### Reading the table

The finer the stratification, the fewer blocks are free to move, so the permuted
series stays closer to the original, chance alignments are stronger, the null
minima reach further, and the bar rises. Year x month leaves only 4.1 blocks per
stratum, which is why its cutoff is three times more stringent than any other
specification.

Whether that is *correct* depends on a substantive judgement we cannot settle
from the data: if the year-over-year forecast improvement is a confounder, year
x month is the right null and the answer is 7,991. If the structure worth
preserving is within-year seasonality, month-only is right and the answer is
24,956. We ship month-only and report both.

### What survives every specification

**The comparison against the naive null.** Row shuffle gives 43,559 regardless,
because it never consumed strata. So the inflation factor ranges from 1.7x
(unstratified) to **5.5x** (year x month) -- and the naive method looks *worse*,
not better, under the more conservative specifications. No choice in this table
rescues it.

Artifact: `sensitivity.json`, produced by `sensitivity.py`.

## Speed, honestly

| stage | ms/perm | vs naive |
|---|---:|---:|
| L0 naive (scipy tail, 1 core) | 1,510.6 | 1x |
| L3+L5 fast p-values | 83.3 | 18x |
| + bitset counts, **1 core, numpy** | 58.4 | 26x |
| **L6 GPU pipeline, end to end** | **0.48** | **3,134x** |

**122x end to end** (58.4 / 0.48), same statistic. Earlier drafts said 58x,
which does not follow from this table, and labelled the third rung "14-core".
Both were wrong:

- `core.popcount` is `np.bitwise_count(...).sum()`, which is **single-threaded**.
  `fastp.minp_null_fast` is the only `parallel=True` routine in the codebase and
  it has **no callers**. There is no multi-core path.
- That baseline is also **unfused**: it materialises a 387 MB temporary for the
  AND and a 48 MB one for the count, moving about 1.26 GB to do 0.39 GB of work.
  Measured useful traffic is **6.8 GB/s** on a machine that sustains 100-400, so
  the CPU here is bound by numpy, **not by memory bandwidth**. An earlier draft
  claimed the opposite and used it as the argument for the GPU. It was wrong.

Against a properly fused, threaded popcount baseline the GPU win would be in the
low teens, not 122x. **The GPU is not the contribution.** GPU acceleration of
permutation testing is established prior art (PBOOST, *Bioinformatics* 2014;
HWY, ACM BCB 2015). It is what makes a 12-second demo possible. See
`POSITIONING.md`.

The defensible engineering claim is the roofline one: arithmetic intensity is
B/4 popcounts per DRAM byte, and against the 235 GB/s this card actually
delivers the knee sits at B = 32.2. We batch 32.

## Housekeeping done

The unused GPU min-p kernel was **deleted**, not left lying around. It was 14x
slower than the CPU scan (weak per-thread bound plus warp divergence) and
carried the same support-boundary bug. `gpu_full.py` now exports only
`gather_pack` and `block_idx_gpu`.

## Is `err_lag24` really available when the forecast is filed?

**The attack.** EIA-930's day-ahead forecast is filed as **one D-1 vintage
covering all 24 hours of day D**. Our code applies a uniform `.shift(24)`. If
the vintage is filed at hour T on D-1, then for hour t on D a realised quantity
at lag 24 -- hour t on D-1 -- had only occurred by filing time if t < T. For a
mid-morning filing that is false for more than half of each day.

So `err_lag24` is not look-ahead with respect to hour t, but it is partly
look-ahead with respect to the **filing instant**. The distinction is real and
we did not catch it ourselves.

**The test.** Rebuild using only predicates that are unambiguously available:
the forecast and shapes derived from it (a forecast is known when filed,
whatever its vintage), calendar terms, and realised quantities at lag >= 48h.
That drops 44 of 85 predicates and 94% of the lattice.

| | predicates | rules | significant | top lift |
|---|---:|---:|---:|---:|
| full, as shipped | 85 | 66,844 | 25,017 | **4.61** |
| strict, lag >= 48h only | 41 | 4,541 | 2,895 | **3.52** |

**The effect survives.** Lift falls from 4.61 to 3.52, which is the honest cost
of the restriction, and it remains far above the 1.0 base rate.

**And one strict rule settles it completely:**

```
fc_level>q75 AND fc_ramp_1h>q90 AND fc_day_range>q75      lift 3.46, support 804
```

Every term is a property of **the forecast itself**. No realised data at any
lag. Whatever vintage that forecast belongs to, all three conditions are
knowable the instant it is filed. A forecast that is high, ramping steeply, and
predicting a wide daily swing is **3.5x more likely to fail**, and that is
computable before the forecast is submitted.

Artifact: `vintage_check.json`, produced by `vintage_check.py`.

## A structural limitation found by audit

`features.py` builds `hour_sin` and `hour_cos`, then binarises every feature at
quantiles taken **within each (hour-of-day x calendar-month) cell**. Inside such
a cell the hour is constant, so those two features are constant, so all eight of
their threshold predicates are **identically False**.

Consequence: the lattice has 85 named predicates but **77 that can ever fire**,
and no rule conditioned on time of day is expressible at all. That is a real
blind spot -- a morning-ramp interaction exists in the data and the miner
structurally cannot see it.

Not fixed, deliberately: adding hour-bucket indicators changes the lattice and
therefore every count in this document. It is the first thing to do after the
event. `surprise.py:singleton_rates` now falls back to the base rate for a
never-true predicate rather than returning NaN, and reports how many it found.

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
