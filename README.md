# SIEVE

Significant pattern mining for autocorrelated time series, where the
multiple-testing correction is done by **permutation** rather than analytically,
because the rows are not independent and the analytic p-value is not valid.

HackMIT 2026, Voloridge "Signal in the Noise".

## Run it

```bash
python3 sieve_cli.py --perms 200 --naive      # PJM, from cache
python3 app.py                                # demo at 127.0.0.1:5000
python3 test_core.py && python3 test_backends.py
```

Data is not in the repo (650 MB EIA + 71 MB NOAA). Fetch with `data/fetch.sh`.

## What it does

1. **Outcome** is the top decile of day-ahead forecast error, taken *within* each
   hour-of-day x calendar-month cell, so daily shape and seasonality cannot
   masquerade as discoveries.
2. **Predicates** must be knowable 24h ahead: the forecast itself, quantities
   lagged >= 24h, or calendar terms.
3. **Lattice** of conjunctions up to arity 3, with apriori pruning, same-feature
   exclusion, and closed-pattern reduction.
4. **Fisher's exact test ranks; it does not decide.** All validity comes from the
   permutation distribution.
5. **Westfall-Young min-P** controls family-wise error across the whole lattice.
6. **Circular block permutation**, 504h blocks stratified by calendar month,
   because hours are not exchangeable.
7. **Tarone's testability bound** removes rules that can never reach the
   threshold. Measured contribution here: 0.36-0.55%, reported as a null result.

## Headline numbers

| | |
|---|---|
| Calibration, 120 true nulls | **0.067** block vs **1.000** naive row shuffle (p = 0.40) |
| PJM significant at 5% FWER | **3,928** block vs **30,883** row shuffle, **7.86x** inflation |
| Cross-dataset | NOAA daily: outcome ACF 0.028, inflation **1.03x**. PJM hourly: ACF 0.716, inflation **7.86x** |

The cross-dataset result is the interesting one: **the correction scales with the
dependence it exists to correct for, and vanishes when that dependence is
absent.**

## Layout

```
core.py        Sieve, bitsets, FisherTable, Tarone sweep, block_permutation
fastp.py       numba log-factorial Fisher tails, minp_scan with early exit
features.py    EIA-930 -> predicates -> lattice
adapter.py     ANY time-indexed CSV -> predicates (persistence-forecast framing)
backends.py    CPU / GPU, sharing one permutation-index generator
engine.py      Problem: null(), significant(), count_ci(), cover()
gpu_kernel.py  CuPy popcount_batch, 32 permutations per loaded word
sieve_cli.py   CLI       app.py + templates/  live demo
```

## Known limitations

See `FINAL.md` for the full list. The load-bearing ones:

- **Data window.** `features.py` asks for EIA fuel column names that changed in
  mid-2024, so rows after 2024-07-02 are silently dropped. The working set is
  45,980 hours, not the 65,036 available.
- **PJM's forecast series is offset by one hour** from 2019-01-30; we correct it
  (`fc_shift=1`, date-gated) and four 2^31 overflow rows are removed. Four of the
  five grids were never affected.
- **The null holds calendar month fixed but not year.** Across defensible null
  specifications the PJM count spans roughly 16,700 to 30,400.
- GPU acceleration of permutation testing is established prior art (PBOOST 2014,
  HWY 2015). The GPU here is engineering, not the contribution. See
  `POSITIONING.md`.
