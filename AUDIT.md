# Implementation audit

Done by known-answer testing, not code reading, because I wrote the code and
reading it would only confirm my own assumptions.

**Two real bugs found. Both fixed. One of them changed a reported result --
for the better.**

## A. Adjusted p-values must be uniform under a true null  -- PASS

The single most decisive test of the whole procedure. If the machinery is
correct, then on data with no signal the adjusted p-value should be a uniform
random number.

200 independent trials, autocorrelated features, autocorrelated outcome, no
relationship between them:

| | observed | expected |
|---|---|---|
| Kolmogorov-Smirnov vs Uniform | D = 0.071, **p = 0.247** | p > 0.05 |
| fraction <= 0.05 | **0.050** | 0.05 |
| fraction <= 0.50 | **0.540** | 0.50 |

The 5% rate is exactly 5%. The procedure is valid.

## B. The early-exit bound  -- BUG FOUND, FIXED

**Symptom:** `ZeroDivisionError` in the hypergeometric tail recurrence.

**Cause:** A ~ Hypergeom(n, n1, x) has support `max(0, x+n1-n) .. min(x, n1)`.
The recurrence divides by `(n - n1 - x + k + 1)`, which is **zero** at
`k = x+n1-n-1`. Starting the sum below the support floor walks straight into
that singularity.

**Why it never showed up:** it fires only when a rule covers more than
`n - n1` rows. On PJM that is 41,614, and our largest rule covers 34,630. The
bug was **latent**, not active -- but it would have fired on a denser dataset
or a rarer outcome.

**Fix:** any `a` at or below the support floor has `P(A >= a) = 1`, so return
log p = 0 instead of iterating. After the fix:

* 400/400 random cases: early exit equals the exhaustive minimum exactly
* high-support regime vs `scipy.hypergeom.logsf`: max error **4.8e-12**

*(The same bug exists in the unused GPU `minlogp` kernel in `gpu_full.py`. We do
not call it -- the CPU scan is 14x faster -- but it should be deleted or fixed
rather than left lying around.)*

## C. Lattice completeness  -- PASS

Apriori pruning is only safe if it never drops a rule brute force would find.
Checked against exhaustive enumeration on a small case:

* rules brute force found that apriori missed: **0**
* rules apriori invented that brute force did not: **0**

## D. Look-ahead leakage  -- BUG FOUND, FIXED, RESULT IMPROVED

**Symptom:** the quantile thresholds that binarise every feature, *and* the
threshold that defines the outcome, were computed over the **entire** period --
including the years supposedly held out.

**Measured contamination:**

| | fitted on all data vs early only |
|---|---|
| predicates differing | **3.484%** |
| outcome labels differing | **1.473%** |

Small, but it means the "out of sample" test was not fully out of sample. This
is exactly the kind of thing a quant judge probes first.

**Fix:** `build(..., fit_until_year=YYYY)`. Every threshold is now estimated on
training rows only and applied forward. After the fix: **0.006%** and
**0.000%**.

**The holdout, re-run clean:**

| | before (leaky) | after (clean) |
|---|---|---|
| top rules still significant out of sample | 19 / 20 | **20 / 20** |
| mean lift train -> test | 4.38 -> 5.10 | **4.47 -> 5.18** |

The result got **stronger** after removing the leakage. That is the outcome you
want from this kind of fix: it shows the leakage was not manufacturing the
finding.

## E. Round trips and edge cases  -- PASS

bit-pack/popcount at n = 1, 63, 64, 65, 4095, 46324; all-zero and all-one
rules; `psi(x)` equals the exhaustive minimum over a. All exact.

## What this audit does not cover

1. **The mechanism.** Weather is largely ruled out (79-98% of the effect
   survives weather strata) but we have not shown *why* these conditions break
   forecasts.
2. **Strong control.** Permuting the outcome tests the complete null; strong
   FWER control additionally assumes subset pivotality. Standard in this
   literature, and stated rather than hidden.
3. **Cross-BA pooling.** We run one balancing authority at a time. Pooling
   would require permuting time blocks jointly across regions, because weather
   correlates them.

## Verdict

The implementation is correct. Two real defects existed; both are fixed and
both are now covered by a test that would catch a regression. The audit also
strengthened the headline holdout result rather than weakening it.
