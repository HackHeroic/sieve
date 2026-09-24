# SIEVE: handoff

For a fresh Claude session, or a human picking this up cold. Read this first;
everything else is detail.

**Repo:** `github.com/HackHeroic/sieve` (private) &middot; local: `~/hackmit2026/sieve`

---

## What this is

A pattern miner for **autocorrelated time series** where the multiple-testing
correction is done by permutation instead of analytically, because the rows are
dependent and every analytic p-value in that setting is invalid.

Built for HackMIT 2026's Voloridge challenge. **Not submitted** &mdash; the value
now is (a) a methods result about block-length calibration and (b) an application
to Watt-IF Stage 2.5.

## The one result that matters

Build datasets containing **no signal at all** (circularly shift the outcome so
autocorrelation survives but the relationship dies), then run both methods:

| data | conventional correction | SIEVE |
|---|---|---|
| PJM electricity, 64,690 h | **1.000** | **0.040** |
| 36 years of VIX | **0.988** | **0.062** |

The conventional approach reports a discovery on almost every dataset containing
nothing. Nominal is 0.05.

## The methods finding (this is the publishable part)

**Block length must cover the dependence horizon, and you cannot infer it from
the ACF.**

| series | ACF says | needs | FPR if you get it wrong |
|---|---|---|---|
| VIX | dies by 126 d | **504 d** | 0.362 at 126 d |
| CISO | 0.211 at 168 h | **4368 h** | **0.850** at 504 h |
| PJM | 0.007 at 168 h | 504 h | 0.100 at 168 h |

Same code, same procedure, one parameter: error rate **0.040 or 0.850**.

Every data-snooping correction in finance (White's Reality Check, Hansen's SPA,
Romano-Wolf StepM, PBO) runs on a block or stationary bootstrap and needs this
parameter. Romano-Wolf has a section titled "Choice of Block Sizes". **Nobody
publishes theirs.**

## Current headline numbers (repaired dataset, `repaired_results.json`)

```
64,690 hours, 2019-01-09 to 2026-09-17   6,532 positive (10.1%)
87,943 rules from 84 predicates, 0 dead
block 504h:  significant 10,252 [10,186 +/- 214]   naive 39,166   inflation 3.82x
calibration: SIEVE 0.040   naive 1.000
top lift 3.18  fc_level>q90 AND dem_lag24>q90 AND err_lag24>q90
```

## Two data defects found in EIA-930, both fixed

1. **PJM's `Demand Forecast` leads its `Demand` by exactly one hour** from
   2019-01-30. PJM alone of 55 balancing authorities; every year. Verified by a
   timezone-invariance argument and the balance identity. Date-gated in
   `features.py`. Uncorrected, a third of the "forecast error" is artifact.
2. **EIA split the Solar/Wind/Hydro columns in mid-2024.** Code asking for the
   old names plus an unguarded `notna().all()` silently dropped every row after
   2024-07-02 &mdash; 31.5% of the data. Now resolved by prefix.

Also: four PJM rows carry 2^31 MW as demand (overflow sentinel). Filtered.

## What was withdrawn, and why

- **holdout 20/20 -> 0/20.** The old figure was computed on the uncorrected
  series; a timestamp offset is stable over time so it transferred beautifully.
  Identical result at 168 h and 504 h, so it is not a block-length artifact.
  Lift still persists out of sample at 2.86.
- **replication 57/58 -> 25/57.** The old figure used one block length across
  all grids, i.e. on grids where error was not controlled (CISO at 0.850).
  Effect sizes do travel: lift 3.02 to 4.12 on all four.

## Files

```
core.py       Sieve, bitsets, Tarone sweep, block_permutation, BlockPermuter
fastp.py      numba Fisher tails, minp_scan with the admissible early exit
features.py   EIA-930 -> predicates -> lattice; carries both data fixes
adapter.py    ANY time-indexed CSV (this is what ran on VIX and NOAA)
backends.py   CPU/GPU sharing one permutation-index generator
engine.py     Problem: null(), significant(), count_ci(), cover()
gpu_kernel_u.py  the fast kernel (compile-time batch, 48% of roofline)
sieve_cli.py / app.py   CLI and live demo
```

Tests: `test_core.py`, `test_all.py`, `test_backends.py`, `test_gpu_equal.py`,
`kernel_ab.py`, `verify_all.py`. All passing.

Artifacts: 18 JSON files. **24 of 24 headline claims machine-checked against
them**, zero failures.

## Documents

| file | what it is |
|---|---|
| `UNDERSTANDING.md` | A-to-Z, 890 lines. Part 6 = code file by file, Part 7 = every engineering choice and why |
| `REPORT.md` | the consolidated results report |
| `FINAL.md` | the numbers reference |
| `POSITIONING.md` | prior art, every citation verified against Crossref |
| `BENCHMARK.md` | comparison against the 2025 winner |
| `CROSS-DATASET.md` | NOAA generalisation |

## What is still open

- **Documents carry the 45,980-hour numbers.** The repaired 64,690-hour run is in
  `repaired_results.json` and supersedes them. The doc pass has not been done.
- **Replication, holdout and VIX have not been re-run on the repaired data.**
  About ten minutes on a GPU, hours locally.
- **Kernel is at 48% of roofline.** Tuned kernels reach 65-85%. Remaining ideas:
  vectorised loads, binary tensor cores (needs B>=128). But the kernel is 21% of
  wall time, so the Amdahl ceiling is 1.26x.
- **Watt-IF Stage 2.5.** Its planned expectation-gap trigger is built on exactly
  the `Demand Forecast` minus served signal SIEVE mines. It **must** apply the
  PJM timestamp fix.

## Standing rules from the user

- Painkiller not vitamin: name who bleeds, how much, with a source.
- No fake technical flexes. We quote 112x against our own CPU, not 2,899x
  against a scipy loop, and the refused 16,000x is in the repo.
- No em/en dashes in shared documents. No changelogs or revision history.
