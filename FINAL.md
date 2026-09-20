# SIEVE: verified results

Every number here traces to a JSON artifact in this repository. Where a number
changed during the audit, the old value and the reason are stated.

**All PJM figures below are on the CORRECTED series**: the day-ahead forecast is
shifted +1h from 2019-01-30 onward, and four overflow rows are removed. See
"The data defect" below.

---

## The problem instance

```
45,980 hours          PJM, 2019-01 to 2024-07
 4,660 positive       top decile of |forecast error| within each (hour, month) cell
66,817 rules          conjunctions of up to 3 predicates, after pruning
   504 hour blocks    circular block permutation, stratified by calendar month
```

## Calibration: the result the project rests on

120 datasets built to contain no signal, by circularly shifting the real outcome
so its autocorrelation survives but its relationship to the features does not.

| null model | false positive rate |
|---|---|
| naive row shuffle | **1.000** |
| circular block permutation, 504h | **0.067** |

Exact binomial test of correct calibration: **p = 0.397**, 95% CI
[0.029, 0.127], 0.85 standard errors from nominal. The naive method declares a
discovery on **every one** of 120 datasets containing nothing.

*Artifact: `final_results.json`.*

### Why 504 hours, and not the 168 we started with

The block length was re-chosen by measurement after the data correction, because
the corrected outcome is more autocorrelated than the one 168h was tuned for
(lag-1 0.654 -> 0.716, lag-6 0.082 -> 0.218).

| block | days | blocks | FPR |
|---|---:|---:|---:|
| 168h | 7 | 275 | 0.100 &nbsp; *(p = 0.019, miscalibrated)* |
| 336h | 14 | 137 | 0.075 |
| **504h** | **21** | **91** | **0.067** &nbsp; *(shipped)* |
| 720h | 30 | 64 | 0.067 |
| 1008h | 42 | 45 | 0.058 |

*Artifact: `block_sweep.json`.*

## Significant rules

| | 168h blocks | 504h blocks |
|---|---:|---:|
| circular block permutation | 5,561 | **3,928** |
| naive row shuffle | 30,883 | 30,883 |
| **inflation** | 5.55x | **7.86x** |

The naive method over-reports by nearly eightfold. Before the data correction
this figure was 1.78x, so **fixing our own defect quadrupled the size of the
effect the project exists to demonstrate.**

## Findings, after removing redundancy

Greedy cover over significant rules with support > 800, ranked by lift.

| lift | support | rule |
|---:|---:|---|
| **2.97** | 897 | `fc_vs_yesterday>q50 AND fc_day_range>q75 AND err_lag24>q90` |
| 2.94 | 1,011 | `fc_vs_yesterday>q75 AND dem_chg_24_48>q50 AND err_lag24>q90` |
| 2.92 | 821 | `err_lag24>q90 AND interchange_lag24>q75 AND share_coal_lag24>q25` |
| 2.90 | 1,316 | `fc_level>q50 AND err_lag24>q90 AND share_nuclear_lag24>q25` |
| 2.88 | 825 | `fc_vs_yesterday>q25 AND err_lag24>q90 AND share_solar_lag24>q75` |
| 2.83 | 1,208 | `dem_chg_24_48>q50 AND err_lag24>q90 AND interchange_lag24>q50` |

`err_lag24>q90` appears in **all six**. Error persistence is the robust core,
and it is also the part the power-systems literature already documents
(Makarov et al., PNNL-19229, 2010).

Before the correction the top lift was 4.61. **About 35% of that was artifact.**

---

## Block length must match the dependence horizon

This is the methodological result, and we found it by getting it wrong.

Using one block length everywhere looked reasonable. It is not. Each grid was
calibrated separately against 100 true nulls:

| grid | outcome ACF at 168h | calibrated block | FPR there | FPR at 504h |
|---|---:|---:|---:|---:|
| PJM | 0.007 | 504h (21d) | 0.067 | 0.067 |
| MISO | 0.084 | 504h | 0.083 | 0.083 |
| NYIS | &mdash; | 504h | 0.083 | 0.083 |
| ERCO | &mdash; | 504h | 0.067 | 0.067 |
| **CISO** | **0.211** | **4368h (182d)** | **0.033** | **0.850** |

**CISO at a 504-hour block has a false positive rate of 0.850.** Its dependence
runs far longer than the others (ACF still 0.152 at 504h and 0.116 at 1008h,
where PJM is at 0.001), so three-week blocks destroy structure that is really
there, the null becomes too easy, and the procedure fires on 85% of datasets
containing nothing.

Give CISO six-month blocks and it calibrates at 0.033.

**The practical claim:** block length is not a convention, it is a measurement,
and the same procedure that controls error at 0.067 on one series runs at 0.850
on another if you carry the parameter over. The tool measures this for you.

## Temporal holdout: it does not hold

| block | held | lift train -> test | test-set significant |
|---|---:|---|---:|
| 168h | **0 / 20** | 3.13 -> 2.86 | 512 |
| 504h | **0 / 20** | 3.13 -> 2.86 | 357 |

Identical at both block lengths, so this is a result and not an artifact of the
null we chose.

An earlier draft reported **20 of 20**. That was computed on the uncorrected
series, and a one-hour timestamp offset is perfectly stable over time, so it
transferred out of sample beautifully. **The holdout was largely measuring the
artifact's consistency.**

What is true now: **lift persists out of sample at 2.86** against a base of 1.0,
and 512 rules in the test window are significant. The phenomenon is there. The
specific top-20 training rules do not survive family-wise correction over 60,576
hypotheses on 12,726 hours.

## Cross-grid replication: the effect travels, the significance mostly does not

PJM's top 15 rules, each grid tested at **its own calibrated block length**.

| grid | replicated | block | FPR | mean lift |
|---|---:|---:|---:|---:|
| CISO | 0 / 15 | 4368h | 0.033 | **4.12** |
| ERCO | 6 / 15 | 504h | 0.067 | **3.02** |
| MISO | 8 / 15 | 504h | 0.083 | **3.66** |
| NYIS | 11 / 12 | 504h | 0.083 | **3.25** |
| **total** | **25 / 57** | | | |

**Read the lift column.** PJM's rules raise the failure rate on every grid
tested, 3.0 to 4.1 against a base of 1.0. The effect travels.

**Formal significance travels in 25 of 57.** The shortfall is concentrated in
CISO's 0/15, and that is a power problem rather than an effect problem: CISO
needs 182-day blocks to calibrate, which leaves only **10 exchangeable units**.
A permutation test on 10 units cannot resolve much, and its measured FPR of
0.033 confirms it is running conservative.

An earlier draft reported **57 of 58**. That was computed with one block length
across all grids, i.e. on grids where the procedure was not controlling error
(CISO at 0.850). It is withdrawn.

## The data defect

PJM's EIA-930 day-ahead demand forecast series **leads its realised demand by
exactly one hour, from 2019-01-30 onward.**

**How we know, and why the obvious test is not enough.** A lower error under a
shift is equally consistent with "the forecast is early" and with "demand is
late", and those require opposite fixes. Two structural tests settle it:

1. **Geography control, on first differences.** PJM's *demand* leads Eastern
   peers at lag 0 and Central peers at lag -1, which is real geography, so the
   instrument resolves one hour. PJM's *forecast* leads every peer at lag -1
   **regardless of timezone**. Timezone invariance is the signature of a label
   offset inside PJM's own file.
2. **The balance identity.** `Demand = NetGen - Interchange` holds for PJM at
   **lag 0**, median residual **7 MW**, 17.4% exact zeros. PJM's demand is
   welded to its own generation, so it is the forecast that moved.

**It is PJM alone.** 51 of 55 balancing authorities are optimal at shift 0 on
this same loader; the other three are degenerate flat curves. PJM is +1 in every
one of eight years; ERCO is 0 in every one.

**The date gate is not optional.** Before 2019-01-30 the series is correctly
aligned: MAPE 2.389 at shift 0 against 3.278 at +1, and shift 0 wins 28 of 28
days. Shifting those hours anyway takes their positive rate from 4.0% to 24.5%
against a 10.1% base, manufacturing false positives into the January cells.

**CISO is not also shifted.** An earlier draft said it was. CISO's MAPE is 8.611
at shift 0 against 8.749 at +1.

**Four overflow rows removed.** PJM carries four rows with impossible demand,
including 2021-10-19 04:00 UTC at **2,147,480,000 MW = 2^31**. They dragged mean
PJM demand from 92,518 to 153,448 MW and made every MAE-based statistic
meaningless.

Independently verified by 16 agents across 5 angles, all concluding misaligned
at 99% confidence.

---

## How much depends on our choices

Same lattice, same observed statistic; only the permutation stratification
changes. Measured before the data correction, so the absolute counts are the old
ones; the *shape* of the dependence is what matters.

| stratification | blocks per stratum | significant |
|---|---:|---:|
| year x month | 4.1 | 7,991 |
| year only | 45.8 | 23,545 |
| month only (shipped) | 22.9 | 24,956 |
| unstratified | 275 | 26,005 |

**Range 7,991 to 26,005**, a factor of 3.3. The finer the stratification, the
fewer blocks are free to move, so the bar rises. **What survives every choice is
the comparison itself** &mdash; the naive null is unaffected by stratification,
so it is too generous under every specification, and more so under the
conservative ones.

*Artifact: `sensitivity.json`.*

---

## Does it generalise beyond this dataset?

NOAA GHCN-Daily, Philadelphia, 1970-2026. Different agency, different domain,
daily rather than hourly, no shared feature code.

| | outcome ACF lag 1 | naive / block inflation |
|---|---:|---:|
| PJM, hourly (corrected) | **0.716** | **7.86x** |
| NOAA, daily | **0.028** | **1.03x** |

**The correction scales with the dependence it exists to correct for, and
vanishes when that dependence is absent.** On daily weather the outcome is
nearly white noise, and SIEVE reports that the naive method was fine all along.

*Artifact: `CROSS-DATASET.md`.*

---

## Speed

Real 66,844-rule lattice, NVIDIA L4.

| stage | ms/perm |
|---|---:|
| naive (scipy tail, 1 core) | 1,510.6 |
| fast p-values | 83.3 |
| bitset counts, 1 core numpy | 58.4 |
| **GPU pipeline, end to end** | **0.521** |

**112x end to end.** Earlier drafts said 58x and then 122x; neither reproduced.

### The kernel, including what failed

| kernel | ms/batch | GB/s | % of roof |
|---|---:|---:|---:|
| thread per rule | 8.691 | 44.5 | 19.0% |
| warp per rule | 5.253 | 73.7 | 31.4% |
| warp + shared memory | 5.616 | 68.9 | 29.3% &nbsp; *slower, reverted* |
| **warp + compile-time B** | **3.430** | **112.9** | **48.0%** |

All four produce bit-identical output. Two negative results kept:
shared-memory staging of the label matrix is **6% slower** (185 KB already lives
in L2, and the per-tile barrier costs more than the traffic saved), and an
unused `parallel=True` scanner in our own code is **3.6x slower** (it
oversubscribes the 6 vCPU the container allocates).

Profiling also showed the kernel was never the bottleneck: host-side permutation
index generation was **53%** of wall time. Caching that structure gave 4.67x with
bit-identical draws.

**The GPU is not the contribution.** GPU permutation testing is established prior
art (PBOOST, *Bioinformatics* 2014; HWY, ACM BCB 2015). Our CPU baseline is
single-threaded unfused numpy at 6.8 GB/s on a machine sustaining 100-400, so
against a tuned implementation the gap would be far smaller.

---

## What is still wrong

**The data window.** `features.py` requests EIA fuel column names that changed in
mid-2024, and an unguarded `notna().all()` drops every row after 2024-07-02 &mdash;
**31.5% of the available data.** Disclosed, not fixed; fixing it moves every
number here.

**A structural blind spot.** `hour_sin` and `hour_cos` are constant within each
(hour, month) binarisation cell, so all eight of their predicates are identically
false. The lattice has 85 named predicates but **77 that can fire**, and **no
time-of-day rule is expressible.**

**Forecast vintage.** EIA files one D-1 vintage covering all 24 hours, so lag-24
features are partly look-ahead relative to the filing instant. Tested: restricting
to forecast-derived, calendar and lag>=48h predicates drops 94% of the lattice and
lift falls 4.61 to 3.52 (measured pre-correction). The effect survives, and one
surviving rule uses only the forecast's own shape.

**Assumed, not proven.** Subset pivotality for strong FWER control. Block
permutation is asymptotically valid under stationarity, not exact &mdash; which is
why calibration is measured rather than claimed.
