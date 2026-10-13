# SIEVE

**Pattern mining that stays honest on autocorrelated time series.**

- False positive rate on signal-free data: **0.040** vs **1.000** for the standard row-shuffle correction.
- Given 36 years of VIX and no finance knowledge, it **rediscovered volatility clustering** (Engle ARCH LM 1,817.8, p = 0).
- Custom CUDA kernels on an NVIDIA L4: **112x** end to end vs single-core NumPy.
- Found and fixed two silent defects in the U.S. EIA-930 grid dataset.

SIEVE searches tens of thousands of candidate rules ("when the forecast is high
AND yesterday's error was high, today's error is extreme") and reports only the
ones that survive a family-wise error correction. The correction is done by
**circular block permutation** rather than analytically, because hourly and daily
rows are not independent, and analytic p-values that assume independent rows are
invalid.

Built solo at HackMIT 2026 for Voloridge's "Signal in the Noise" challenge. Not
submitted; developed further afterwards.

`Python` `NumPy` `Numba` `CuPy / raw CUDA` `Flask`

---

## The result

Build datasets that contain **no signal at all** (circularly shift the outcome,
so autocorrelation survives but the relationship dies) and ask each method
whether it found something. At a 5% family-wise error rate, the correct answer is
"yes" about 5% of the time.

| null datasets | standard row shuffle | SIEVE (block permutation) |
|---|---:|---:|
| PJM electricity, 64,690 hours (100 nulls) | **1.000** | **0.040** |
| VIX, 36 years of daily data (80 nulls) | **0.988** | **0.063** |

The standard approach reports a discovery on almost every empty dataset.

## On market data, it found a known law without being told

Same code, through the generic CSV adapter, on FRED's daily VIX (1990 to 2026,
90,307 candidate rules). The outcome: days when the naive "tomorrow's VIX equals
today's" forecast fails worst.

Top finding, lift **5.57**: `VIX_lag1>q90 AND perr_lag1>q90 AND perr_lag2>q90`.
Yesterday's VIX move was extreme and so was the day before's, so today's will be.
That is volatility clustering, Mandelbrot's "large changes are followed by large
changes, of either sign". Checked formally rather than asserted:

| test | result |
|---|---|
| ACF of **signed** daily VIX change, lags 1 / 2 / 5 | -0.125, -0.035, 0.008 |
| ACF of **absolute** daily VIX change, lags 1 / 2 / 5 | **0.366, 0.364, 0.283** |
| Ljung-Box Q(10) on absolute change | **7,446.7**, p = 0 |
| Engle ARCH LM, 10 lags | **1,817.8**, p = 0 |

It is more than persistence: conditioning on the previous two days alone gives a
lift of 4.09; SIEVE's three-way conjunction gives 5.57. Full write-up in
[`REPORT.md`](REPORT.md), section "Market data".

## The methods finding

**Block length has to cover the dependence horizon, and the autocorrelation
function does not tell you what that is.**

| series | ACF suggests | actually needs | false positive rate if too short |
|---|---|---|---:|
| VIX | near zero by 126 days | **504 days** | 0.362 at 126 days |
| CISO | 0.211 at 168 h | **4,368 h** | **0.850** at 504 h |

Same series, same code, one parameter: CISO's false positive rate is 0.850 at
504 h and 0.033 at 4,368 h. The standard data-snooping corrections in finance
(White's Reality Check, Hansen's SPA, Romano-Wolf StepM) run on a block or
stationary bootstrap and need this parameter, and the result depends on it.

## Headline numbers (PJM, repaired dataset)

```
64,690 hours, 2019-01-09 to 2026-09-17      6,532 positive hours (10.1%)
87,943 rules from 84 predicates
significant at 5% FWER:  10,252 block  vs  39,166 row shuffle   (3.82x inflation)
top rule, lift 3.21:     fc_level>q90 AND dem_chg_24_48>q50 AND err_lag24>q90
```

Source: [`repaired_results.json`](repaired_results.json).

## Speed

Pre-repair 66,844-rule lattice on an NVIDIA L4, milliseconds per permutation:

| stage | ms / perm |
|---|---:|
| naive (SciPy tail, 1 core) | 1,510.6 |
| fast p-values (Numba) | 83.3 |
| bitset counts (1 core, NumPy) | 58.4 |
| **GPU pipeline, end to end** | **0.521** |

**112x end to end** against my own single-core NumPy path.

The kernel ladder, including what did not work. All four produce bit-identical
output; roofline is against the 235 GB/s the card measured.

| kernel | ms / batch | GB/s | % of roofline |
|---|---:|---:|---:|
| thread per rule | 8.691 | 44.5 | 19.0% |
| warp per rule | 5.253 | 73.7 | 31.4% |
| warp + shared memory | 5.616 | 68.9 | 29.3% (slower, reverted) |
| **warp + compile-time batch** | **3.430** | **112.9** | **48.0%** |

Profiling showed the kernel was never the bottleneck: host-side permutation index
generation was **53% of wall time**, and caching it gave 4.67x with bit-identical
draws.

The GPU is engineering, not the contribution. GPU permutation testing is prior art
(PBOOST 2014, HWY 2015), and the CPU baseline is single-threaded and unfused, so a
tuned CPU implementation would narrow the gap. Details in [`FINAL.md`](FINAL.md)
and [`POSITIONING.md`](POSITIONING.md).

## Two defects found in EIA-930 data

1. **PJM's `Demand Forecast` leads its `Demand` by exactly one hour** from
   2019-01-30. Of the other 54 balancing authorities, 51 test clean and 3 are
   indeterminate. Uncorrected, a third of PJM's "forecast error" is a timestamp
   artifact. Fixed, date-gated, in `features.py`.
2. **EIA split the Solar, Wind and Hydro columns in mid-2024.** Old column names
   plus an unguarded `notna().all()` silently dropped every row after 2024-07-02,
   31.5% of the data. Now resolved by prefix.

Four PJM rows with impossible demand values (one is 2^31 MW, an overflow
sentinel) are also filtered.

## How it works

1. **Outcome:** the top decile of day-ahead forecast error, taken within each
   hour-of-day x month cell, so daily shape and seasonality cannot pass as
   discoveries.
2. **Predicates** use the forecast itself, quantities lagged at least 24 h, or
   calendar terms. (EIA files one day-ahead vintage for all 24 hours, so lag-24
   features are partly look-ahead; restricting to lag 48 h and beyond keeps the
   top lift at 3.52.)
3. **Lattice** of conjunctions up to arity 3, with apriori pruning, same-feature
   exclusion and closed-pattern reduction.
4. **Fisher's exact test ranks; it does not decide.** Validity comes from the
   permutation distribution.
5. **Westfall-Young min-P** controls family-wise error across the whole lattice.
6. **Circular block permutation**, stratified by calendar month, because hours
   are not exchangeable.
7. **Tarone's testability bound** drops rules that can never reach the threshold
   (measured contribution here: 0.36 to 0.55%, reported as a null result).

## Run it

```bash
pip install -r requirements.txt               # add cupy-cuda12x for the GPU path
(cd data && ./fetch.sh)                       # 650 MB of EIA-930, not in the repo
python3 rebuild_cache.py                      # builds the PJM lattice cache
python3 sieve_cli.py --perms 200 --naive      # PJM
python3 app.py                                # live demo at 127.0.0.1:5000
python3 test_core.py && python3 test_backends.py
```

`adapter.py` turns any time-indexed CSV into predicates; that is how the VIX run
([`fred/`](fred)) was done.

## Reports

| file | what it is |
|---|---|
| [`FINAL.md`](FINAL.md) | numbers reference, including what is still wrong |
| [`UNDERSTANDING.md`](UNDERSTANDING.md) | A to Z walkthrough; Part 6 is the code file by file, Part 7 every engineering choice |
| [`POSITIONING.md`](POSITIONING.md) | prior art, citations checked against Crossref |
| [`AUDIT.md`](AUDIT.md) | known-answer tests of the implementation |
| [`CROSS-DATASET.md`](CROSS-DATASET.md) | NOAA generalisation |
| [`REPORT.md`](REPORT.md) | consolidated report, pre-repair; its replication figures are withdrawn (see below) |

## Layout

```
core.py           Sieve, bitsets, Tarone sweep, block permutation
fastp.py          Numba Fisher tails, min-P scan with admissible early exit
features.py       EIA-930 -> predicates -> lattice (carries both data fixes)
adapter.py        any time-indexed CSV -> predicates
backends.py       CPU / GPU sharing one permutation-index generator
engine.py         Problem: null(), significant(), count_ci(), cover()
gpu_kernel_u.py   the fast kernel (warp per rule, compile-time batch)
sieve_cli.py      CLI          app.py + templates/   live demo
```

## Known limitations

- **Withdrawn results.** Holdout went from 20/20 to 0/20 once PJM's timestamp
  offset was corrected (lift still holds out of sample at 2.86). Cross-grid
  replication went from 57/58 to 25/57 once each grid got its own block length;
  effect sizes still travel (mean lift 3.02 to 4.12 on all four grids).
- **Some reports still carry the 45,980-hour numbers.** The 64,690-hour run in
  `repaired_results.json` supersedes them; replication, holdout and VIX have not
  been re-run on it yet.
- **The null holds calendar month fixed but not year.** Across defensible null
  specifications the PJM count spans 3.3x (7,991 to 26,005, pre-repair).
