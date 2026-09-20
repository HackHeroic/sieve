# Understanding SIEVE properly

Not a summary. This is the reasoning chain, so you can answer a question you
have not seen before instead of reciting an answer you have.

The method is five decisions. For each one: what it does, what breaks without
it, and the number that proves it.

---

## The worked example, first

Everything below is abstract until you have seen one rule go through the whole
machine. Here is a real one, with real numbers off the real data.

```
==========================================================================
WORKED EXAMPLE, ONE REAL RULE, END TO END
==========================================================================

RULE:  fc_level>q50 AND fc_day_range>q90 AND err_lag24>q90

  'the day-ahead forecast is above median, AND it predicts a daily swing
   in the top 10%, AND yesterday's forecast error was in the worst 10%'

STEP 1 -- the 2x2 table

                      high-error hour    normal hour      total
   rule fires                393              445           838
   rule does not           4,317           41,169        45,486
   total                   4,710           41,614        46,324

  base rate of high-error hours : 4,710/46,324 = 10.17%
  rate when this rule fires     : 393/838 = 46.90%
  LIFT = 4.61x

STEP 2 -- Fisher's exact test, right tail
  Under the null, A ~ Hypergeometric(n=46,324, n1=4,710, x=838)
  E[A] = x*n1/n = 85.2, we observed a = 393
  P(A >= 393) = 1.367e-166      log p = -381.9
  >>> This number is NOT used for inference. Rows are autocorrelated.
      It is a RANKING statistic only.

STEP 3 -- what the permutation null says
  1,000 circular block permutations of the outcome (168h blocks,
  stratified by calendar month). For EACH, take the smallest log p over
  all 66,844 rules. That is the null distribution of the
  best-looking rule you would find in data with no signal.

    null minima:  -91.3 (most extreme) .. -17.1 (least)
    5% quantile:  -51.7   <-- the significance cutoff
    this rule  :  -381.9

  How many of the 1,000 null permutations produced a minimum at least
  as extreme as this rule?   0
  ADJUSTED p = (0+1)/(1000+1) = 0.0010

STEP 4 -- why the naive method gets this wrong
  Row shuffle null minima: -16.8 .. -6.1
  its 5% cutoff: -12.9  vs block -51.7
  Destroying the autocorrelation makes extreme minima IMPOSSIBLE to reach,
  so the bar drops and far more rules clear it:
    block permutation : 24,999 rules significant
    row shuffle       : 43,603 rules significant
    the naive null manufactures 18,604 extra 'discoveries'

STEP 5 -- Tarone, and why it barely matters here
  A rule with support x can never achieve a p below psi(x) = P(A = min(x,n1)).
  If psi(x) > threshold, the rule is UNTESTABLE and leaves the correction count.
  Here: sigma = 7, delta = 7.507e-07
  testable 66,604 of 66,844  -> prunes only 0.36%
  Our predicates are quantile-based and therefore dense, so almost every
  rule has enough support to be testable. Reported as a measured null.
==========================================================================
```

---

## Now the five decisions

### Decision 1: stratify the outcome before you search

**What.** An hour is a failure if its forecast error is in the worst 10% *of
its own hour-of-day by calendar-month cell*.

**What breaks without it.** You rediscover that summer evenings are hard. True,
known, useless. The top decile taken globally would be almost entirely 5pm-8pm
in July and August, and every rule you found would just be a proxy for "it is
a summer evening".

**The number.** The base rate is 10.17% by construction, in every cell. So a
lift of 4.61 means something within a comparable group, not across groups.

---

### Decision 2: features must be knowable 24 hours ahead

**What.** Three legal kinds only: the day-ahead forecast and shapes derived
from it, realised quantities lagged >= 24h, calendar terms.

**What breaks without it.** You get attribution, not prediction. If a feature
uses what happened at hour t, then the honest answer to "what could the
operator have done?" is "nothing, you are describing the past".

**The trap we actually fell into.** Not the features, the THRESHOLDS. The
quantile cutoffs were originally fitted on all years including the test period,
so the test period helped define its own labels. Fixing it moved the holdout
from 19/20 to 20/20 and the lift from 4.38->5.10 to 4.47->5.18. The result got
STRONGER once the leak was removed, which is the opposite of the usual
direction and is worth saying out loud.

---

### Decision 3: Fisher ranks, permutation decides

**What.** The Fisher p-value orders rules. It is never used as a probability.

**What breaks without it.** Nothing breaks loudly, which is the danger. You get
p = 1.4e-166 on the example rule and it looks decisive. It is not, because the
hypergeometric assumes exchangeable rows and consecutive hours are strongly
autocorrelated. The number is a ranking score wearing the costume of a
probability.

**How to say it.** "We use Fisher's exact test as a ranking statistic. All
validity comes from the permutation distribution."

---

### Decision 4: Westfall-Young minP over the whole lattice

**What.** Per permutation, keep only the SMALLEST p across all 66,844 rules.
That builds the distribution of the best-looking rule findable in noise.

**What breaks without it.** Test 66,844 rules at 5% and roughly 3,300 fire by
construction. Bonferroni would fix that but assumes independence; these rules
overlap massively (they share predicates), so Bonferroni is far too
conservative. minP learns the real dependence because the permutation preserves
the correlation structure among rules for free.

**Why minP and not maxT.** maxT compares raw statistics. Rules of different
support have Fisher statistics on non-comparable scales, so the max is
dominated by whichever support happens to produce large numbers. p-values are
on a common scale by construction.

**The assumption to volunteer.** Weak control is exact under the complete null.
Strong control additionally needs subset pivotality. Standard, and stated.

---

### Decision 5: circular block permutation, stratified by month

**What.** Shuffle contiguous 168-hour blocks, exchanging only blocks that sit
in the same calendar month.

**What breaks without it.** Everything. This is the one that matters.

```
circular block permutation    7/120   = 0.058   target 0.05,  p = 0.67
naive row shuffle           120/120   = 1.000                 p = 7.5e-157
```

One standard error at n = 120 is 0.020, so 0.058 sits 0.4 s.e. from nominal.
If he asks "how precise is that?", the answer is the CI, not the point estimate.

120 datasets constructed to contain no signal at all. The naive method declares
a discovery on every single one.

**WHY it fails, and this is the part to understand rather than memorise.** A
row shuffle turns the outcome into white noise. A white-noise outcome cannot
accidentally line up with an autocorrelated rule for long stretches, so the
best-in-noise p-value never gets very extreme: the row-shuffle minima only
reach -16.8. Block permutation keeps the outcome's autocorrelation, so chance
alignments CAN be long and strong, and the minima reach -91.3.

The cutoff is the 5th percentile of those minima. Block: -51.7. Row: -12.9.
More negative is a harder bar. So the naive null sets a bar four times too
easy, and 18,604 extra rules walk under it.

**Say it in one line.** "The naive null is not just wrong, it is
anti-conservative, and we measured by how much."

---

## The four things that could still be wrong, and what we did

| Worry | Test | Result |
|---|---|---|
| It finds things in noise | 120 true nulls | **0.058**, CI [0.024, 0.117] vs naive **1.000** |
| It overfits the sample | thresholds fit on 2019-2022 only, tested forward | **20 of 20**, lift 4.47 -> **5.18** |
| It is a PJM quirk | rerun on CISO, ERCO, MISO, NYIS | **57 of 58** |
| It is just storms | NOAA join, matched temp/rain/wind | **79% to 98%** retained |

---

## Answering a question you have not seen

The pattern for anything he throws at you:

1. **Name which of the five decisions it attacks.**
2. **Say what would break if that decision were removed.**
3. **Give the number that proves the decision earns its place.**

If you do not have a number, say so. "We did not measure that" is a complete
and respectable answer. "I think it would probably be fine" is not.

---

## Your own nulls. Volunteer these before he finds them.

1. **Mechanism is unknown.** We know when forecasts fail, not why. Weather is
   largely excluded but not replaced with an explanation.
2. **Tarone bought 0.36% here** (0.55% on the fuller dataset). Quantile
   predicates are dense so nearly everything is testable. A measured null about
   a celebrated method, reported rather than buried.
3. **Subset pivotality** is assumed for strong FWER control.
4. **168h blocks straddle month ends**, so about 18% of rows change calendar
   month under a stratified draw. Inherent to blocks longer than a day.
5. **We found our own bug.** The ragged final block used to shift everything
   after it, so only 3 of 20 draws kept blocks inside their own month. Pinned
   it; 20 of 20 now; the headline moved 1.3%.
