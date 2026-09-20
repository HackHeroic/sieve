# Every concept in SIEVE, and how to defend it

Read down the left column to explain. Read the right column when pushed.

---

## 1. The outcome: top decile, within hour-of-day by calendar-month

**What it does.** We flag an hour as a failure if its absolute day-ahead demand
forecast error is in the worst 10% *of its own cell*, where a cell is one
hour-of-day crossed with one calendar month. January 6am is compared only to
other January 6am hours.

**Why it is there.** Forecast error is much larger at 6pm than at 3am, and
larger in August than in April. A flat top-decile rule would discover "errors
are big in summer evenings", which is true, useless and not a pattern.
Stratifying the outcome removes the daily and seasonal shape before the search
starts, so anything found afterwards is deviation *within* a comparable group.

**If pushed.** Yes, this reduces power against effects that are purely seasonal.
That is the intended trade. We are looking for conditional structure, not the
climate.

---

## 2. Forecast-time-available features

**What it does.** Every predicate must be knowable 24 hours before the hour it
describes. Three legal kinds: the day-ahead forecast itself and shapes derived
from it, realised quantities lagged 24 hours or more, and calendar terms.

**Why it is there.** A day-ahead forecast is filed at roughly t-24h. If a
feature uses what actually happened at hour t, the model is explaining the past,
not predicting. The honest answer to "what could the operator have done?" would
be "nothing".

**How you know it holds.** It is enforced in `features.py`, not by convention.
Realised solar share, ramp and interchange all appear only as `_lag24`.

---

## 3. The rule lattice, and three prunings

**What it does.** Conjunctions of up to three predicates. 66,844 survive.

**The three prunings, and why each is not a heuristic:**

- **Apriori.** Support is anti-monotone: adding a condition can only shrink the
  covered set. So a candidate already below the support floor can never yield a
  viable superset. Exact, not approximate.
- **Same-feature exclusion.** Quantile thresholds on one feature *nest*.
  `err>q25 AND err>q75` is literally the same set of rows as `err>q75`. Allowing
  both fills the output with logical duplicates.
- **Closed-pattern reduction.** If a longer rule covers exactly the same rows as
  a shorter one, keep the shorter. The discarded rule carries no extra evidence.

**If pushed.** None of these change which rules are testable or what their
p-values are. They remove duplicates and impossibilities, so the multiple-testing
count is honest rather than inflated by redundancy.

---

## 4. Fisher's exact test as a RANKING statistic only

**What it does.** For each rule we build a 2x2 table (in rule / not in rule,
crossed with failure / not failure) and take the right-tailed Fisher p-value.

**Why it is there, and the key sentence to say out loud.** *We do not use this
p-value for inference.* Hourly rows are autocorrelated, so the independence
assumption behind the analytic p-value is false and the number is not valid as
a probability. It is used only to ORDER rules. Every claim of significance comes
from the permutation distribution instead.

**Why Fisher and not chi-square.** Margins are fixed under permutation, so the
hypergeometric is the exact conditional distribution, and it stays correct in
the small-support tail where chi-square degrades.

---

## 5. Westfall-Young minP

**What it does.** For each permutation, compute the p-value of every rule and
keep only the single SMALLEST one. Do that J times and you have the null
distribution of the best-looking rule you would find in data with no signal. A
real rule's adjusted p is its rank against that distribution.

**Why it is there.** Test 66,844 rules at 5% and you expect about 3,300 false
positives by construction. Bonferroni would control that but assumes
independence, and these rules overlap heavily, so it is far too conservative.
minP learns the actual dependence from the data, because the permutation
preserves the correlation structure among rules.

**Why minP and not maxT.** maxT compares raw test statistics across rules. Rules
with different support have Fisher statistics on non-comparable scales, so the
maximum is dominated by whichever support happens to produce big numbers. minP
compares p-values, which are on a common scale by construction.

**If pushed on strong control.** Weak control is exact under the complete null.
Strong control additionally assumes subset pivotality. That is standard in this
literature and we state it rather than hide it.

---

## 6. Circular block permutation, stratified by month

**What it does.** Instead of shuffling rows, shuffle contiguous 168-hour blocks,
and only exchange blocks that belong to the same calendar month.

**Why it is there.** It is the choice that makes the numbers valid. It is NOT
the novel one: Romano and Wolf (*Econometrica* 2005) has a section titled
"Choice of Block Sizes" that names the circular block bootstrap for stationary
series, and Sullivan, Timmermann and White (1999) applied the stationary
bootstrap across 7,846 trading rules. What is unclaimed is that the lattice,
the testability bound and the block null coexist. See POSITIONING.md. A row
shuffle destroys the autocorrelation, which produces a null that is far too easy
to beat. Measured on 120 datasets containing no real signal:

```
circular block permutation     7/120   = 0.058   (target 0.05, p = 0.67)
naive row shuffle            120/120   = 1.000   (p = 7.5e-157)
```

The naive null fires on every single null dataset. If a judge remembers one
number, make it this pair.

**Why 168 hours.** Chosen from where the autocorrelation decays: the measured
horizon was 76 hours, and 168 gives comfortable margin while staying a whole
number of weeks so the weekday pattern is preserved.

**If pushed on validity.** Correct: block permutation gives asymptotic validity
under stationarity, not exactness. That is precisely why we measured the false
positive rate on 120 nulls instead of claiming a theorem.

**Known limitation, say it first.** 168-hour blocks straddle month boundaries,
so about 18% of individual rows change calendar month under a stratified draw.
Blocks are assigned to a stratum by their first row. Inherent to blocks longer
than a day.

---

## 7. Tarone's testability bound

**What it does.** A rule with tiny support cannot reach a small p-value no
matter how the labels fall. Its minimum attainable p already exceeds the
threshold, so it can be removed from the correction count entirely rather than
tested and rejected.

**The self-consistency.** The threshold depends on how many rules are testable,
and testability depends on the threshold: `delta = alpha / k(delta)`. Solved by
sweeping the support threshold downward, because naive iteration oscillates.

**The honest part, and say it unprompted.** On this data Tarone buys **0.55%**.
Our predicates are quantile-based and therefore dense, so almost every rule has
enough support to be testable. It is a measured null result and we report it as
one rather than quietly dropping the machinery.

---

## 8. FWER, not FDR

**What it does.** We control the probability of making *even one* false
discovery, not the expected proportion among discoveries.

**Why.** A grid operator acting on a rule wants that rule to be real. FDR at 5%
means one in twenty of the rules you act on is noise, which is the wrong
guarantee when each action has an operational cost.

---

## 9. Greedy cover, because "25,000 significant" is not a finding

**What it does.** Take the significant rules with support above 800, rank by
lift, accept a rule only if it covers high-error hours no accepted rule already
covers.

**Why it is there.** 38% of the lattice clearing the threshold is arithmetically
correct and practically useless, because the survivors are variants of one
effect. Reporting is a separate problem from testing, and this is the reporting
step.

**Why the support floor.** Below about 800 hours, lift is estimated off a handful
of observations and is unstable. The unfiltered top rule has lift 8.2 on 43
hours, which is noise dressed as a discovery.

---

## 10. Surprise, via De Bie's FORSIED

**What it does.** Lift measures deviation from the base rate. That rewards a
rule for restating its own parts: if A raises the failure rate and B raises it,
nobody is surprised that A AND B does. So we score instead against a background
model in which the analyst already knows every single-predicate effect. The
maximum-entropy completion of that knowledge is additive in log-odds, so

```
surprise = logit(observed) - [ logit(p0) + SUM_i ( logit(p_i) - logit(p0) ) ]
```

A rule with high surprise is doing something its own components do not explain.
That is an interaction, and it is unexpected by construction.

**The result.** Lift and surprise are nearly orthogonal: the top 50 by each
overlap in 1 rule out of 200. So "significant" and "interesting" really are
different questions, measured rather than asserted.

---

## 11. The four validations

| Test | What it rules out | Result |
|---|---|---|
| Calibration on 120 true nulls | the method finds things in noise | **0.058** (95% CI 0.024-0.117) vs naive **1.000** |
| Temporal holdout, thresholds fit on 2019-2022 only, tested 2023-01 to 2024-07 (18 months) | overfitting to the sample | **20 of 20**, lift 4.47 -> **5.18** |
| Replication on CISO, ERCO, MISO, NYIS | a PJM-specific quirk | **57 of 58** (15/15, 14/15, 15/15, 13/13) |
| NOAA weather join, matched temp/rain/wind | "storms are just hard" | **79% to 98%** retained |

The holdout number is the one to lead with, because the effect got *stronger*
out of sample. That normally does not happen, and it happened only after we
removed a look-ahead leak in the quantile thresholds.

---

## 12. Speed, and the only claim worth making

Bitsets pack 64 hours per 64-bit word, so a rule-versus-outcome intersection is
an AND plus a population count. The GPU holds the 387 MB rule matrix resident
and tests a BATCH of 32 permuted label vectors against each loaded word.

**The win is reuse, not parallelism.** The naive port reads 387 MB once per
permutation; batching reads it once per 32.

Do NOT say the CPU was memory-bandwidth bound. It was not. Our CPU baseline is
single-threaded, unfused numpy running at 6.8 GB/s of useful traffic on a
machine that sustains 100-400. It is bound by numpy, not by the memory system.
Against a fused, threaded popcount the GPU win would be low double digits.

The defensible engineering claim is the roofline one: arithmetic intensity is
B/4 popcounts per DRAM byte, so against the 235 GB/s this card actually delivers
the knee sits at B = 32.2, and we batch 32. That explains why the batch size is
right instead of asserting it.

**Quote 122x end to end** (58.4 / 0.48 ms per permutation). Not 3,134x
against naive Python, and not the pod's own throttled CPU, which would have
produced 16,000x.

---

## What we do NOT know. Lead with these.

1. **Mechanism.** We know *when* forecasts fail, not *why*. Weather is largely
   excluded but has not been replaced with an explanation.
2. **Tarone contributed almost nothing here,** 0.55%, for a structural reason we
   can name.
3. **Subset pivotality** is assumed for strong FWER control.
4. **Cross-grid pooling** is not done. One balancing authority at a time, because
   pooling needs joint permutation across regions that weather correlates.
