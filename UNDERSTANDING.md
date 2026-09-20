# SIEVE, from first principles

This is the "understand it properly" document. Not a summary. It starts from
what a pattern even is and ends with why our GPU kernel sits at 48% of its
roofline, and it explains the intuition behind every choice in between.

If you can follow this, you can answer a question nobody prepared you for.

---

# PART 1 &mdash; THE IDEA

## 1. What is a pattern, really?

A pattern is a **subset of your data where the outcome behaves differently from
everywhere else.**

That is the whole definition. In our case:

- The **data** is 45,980 hours of the PJM electricity grid.
- The **outcome** is a yes/no: did the day-ahead demand forecast fail badly
  this hour?
- A **pattern** is a condition that picks out some of those hours, inside which
  failures happen more often than they do overall.

Concretely, one real pattern we found:

> *the forecast is above median* **AND** *the forecast predicts a top-decile
> daily swing* **AND** *yesterday's error was in the worst decile*

That condition is true for 838 of the 45,980 hours. Inside those 838 hours the
forecast failed 393 times, which is **46.9%**. Across all hours it fails
**10.17%** of the time.

```
46.9% / 10.17% = 4.61
```

That ratio is called **lift**. This pattern makes failure 4.6 times more likely.
That is what a pattern is: a condition, and a number saying how much the world
changes inside it.

## 2. Why let a machine find them?

Because a human can only check the patterns they think of.

We have 85 yes/no conditions (predicates). If you allow combinations of up to 3
of them, the number of possible patterns is:

```
85 alone  +  85x84/2 pairs  +  85x84x83/6 triples   =  102,510
```

After removing logical duplicates (more on that in section 8) we are left with
**66,817 real candidate patterns**.

No human enumerates 66,817 hypotheses. A human checks maybe twenty, all of them
things they already suspected. The machine checks all of them, including the
ones nobody would think to suspect.

**That is the entire reason to automate this.** Not speed. Coverage.

## 3. Why that is dangerous

Here is the problem, and it is the reason this project exists.

**Flip a fair coin 10 times.** Getting 9 or more heads is unlikely, about 1 in
100. If it happens, you would be suspicious of the coin.

**Now flip 100 fair coins, 10 times each.** You should *expect* about one of
them to show 9+ heads. If you then point at that coin and say "this one is
biased, look at the evidence", you are wrong. You did not find a biased coin.
You found the luckiest of 100 fair ones.

Now replace "100 coins" with **66,817 candidate patterns**. Test each at the
usual 5% threshold and you expect roughly **3,300 to look significant purely by
chance**, even if nothing real exists at all.

This is the **multiple testing problem**, and it is why pattern mining has a bad
reputation. Search hard enough and the data will always confess to something.

## 4. The standard fix, and why it breaks here

The standard fix is to demand more evidence when you run more tests.

**Bonferroni** is the simplest: testing 66,817 things at an overall 5% level
means each individual test must clear 0.05 / 66,817. It works, but it assumes
the tests are **independent**. Ours are massively overlapping &mdash; the pattern
`A AND B` shares rows with `A AND C` &mdash; so Bonferroni demands far more
evidence than necessary and you find nothing.

**Permutation testing** is the better fix, and the intuition is beautiful:

> Instead of doing maths about what chance *would* produce, **simulate chance
> directly.** Scramble the outcome so any real relationship is destroyed, then
> re-run the entire search. Whatever the search finds now is, by construction,
> pure luck. Do it a thousand times and you know exactly how lucky the search
> can get. Then ask: is your real finding better than that?

This handles overlapping tests for free, because the scrambled data has the
same overlaps.

**But here is where it breaks.** The obvious way to scramble is to shuffle the
outcome values randomly across rows. That is valid only if rows are
**exchangeable** &mdash; if the order does not matter.

**Hours are not exchangeable.** If the forecast failed at 2pm it very likely
also failed at 3pm. The lag-1 autocorrelation of our outcome is **0.654**.

Shuffling rows destroys that. And destroying it makes chance *weaker* than it
really is, because a randomly scattered outcome cannot accidentally line up with
a pattern for a long stretch, whereas a realistically clumpy one can.

So the shuffled world is **easier** than the real world, the bar you set from it
is **too low**, and you accept junk.

**We measured exactly how bad this is.** We built 120 datasets containing no
signal at all, by taking the real outcome and sliding it in time so its clumpy
structure survives but its relationship to the features is destroyed. Then we
ran both methods on all 120:

```
naive row shuffle            declared a discovery on   120 of 120
circular block permutation   declared a discovery on     8 of 120
```

**The standard approach fails on every single dataset that contains nothing.**
Ours fires on 7, which is what a correctly calibrated 5% procedure should do.

That measurement is the reason this project exists.

---

# PART 2 &mdash; HOW SIEVE WORKS, STEP BY STEP

Nine steps. For each: what it does, the intuition, and what breaks without it.

## 5. Step one: define what "failure" means

**What.** For every hour we compute how wrong the day-ahead forecast was. Then
we flag the worst 10%.

**The subtlety.** Worst 10% *compared to what?* Forecast errors are naturally
bigger at 6pm than at 3am, and bigger in August than in April. If we just took
the worst 10% overall, almost all of them would be summer evenings.

So we take the worst 10% **within each (hour-of-day, calendar-month) cell**.
January 6am hours compete only against other January 6am hours.

**Intuition.** We are grading on a curve, one curve per cell. We are not asking
"was this a big error?" but "was this a big error *for a January 6am*?"

**What breaks without it.** You rediscover that summer evenings are hard. True,
already known, and useless. Every rule you found would just be a proxy for "it
is a summer evening."

## 6. Step two: turn numbers into yes/no questions

**What.** Every feature becomes several yes/no predicates by cutting it at
quantiles: is it above the 25th percentile? the 50th? the 75th? the 90th?

So `fc_level` (the forecast value) becomes four predicates:
`fc_level>q25`, `fc_level>q50`, `fc_level>q75`, `fc_level>q90`.

**Why binary?** Three reasons.

1. **Combinations only make sense for yes/no things.** "Forecast is high AND
   yesterday was bad" is a sentence. "Forecast is 41,203 MW AND yesterday's
   error was 1,847 MW" is not a pattern, it is a single data point.
2. **A yes/no condition over rows is a bitmask**, and bitmasks are
   extraordinarily fast to combine (section 16).
3. **It makes the statistics exact.** With binary in and binary out, every
   pattern produces a 2x2 table, and 2x2 tables have an exact test.

**The subtlety again.** The quantile cuts are also computed **within each
(hour, month) cell**, for the same reason as the outcome.

**The rule that governs all of this.** Every predicate must be knowable **24
hours before** the hour it describes, because that is when the forecast is
filed. Three legal kinds:

- the forecast itself and shapes derived from it (known when filed)
- realised quantities lagged 24h or more (already happened)
- calendar terms (always known)

**What breaks without it.** You get a model that explains the past. If a
predicate uses what actually happened at hour t, then the honest answer to "what
could the operator have done?" is "nothing, you are describing history."

## 7. Step three: build the candidate patterns

**What.** Take all combinations of up to 3 predicates joined by AND.

**Intuition: the lattice.** Picture a pyramid. The top layer is single
predicates. The next is pairs. The next is triples. Each rule is the AND of its
parents, so each rule covers *fewer* rows than its parents. Going down the
pyramid, conditions get more specific and cover less.

That one-directional property has a name &mdash; **anti-monotonicity** &mdash;
and it is what makes the search tractable.

## 8. Step four: throw away what cannot matter

Three prunings. None is a heuristic; each removes things that are provably
useless.

**Apriori.** If `A` covers only 4 rows, then `A AND B` covers at most 4. So if 4
is already too few to ever be significant, we can skip every rule containing
`A`, without testing them. **Intuition: you cannot get bigger by adding
conditions, so if you are already too small, give up on this whole branch.**

**Same-feature exclusion.** `fc_level>q25 AND fc_level>q75` is exactly the same
set of rows as `fc_level>q75`, because the thresholds nest. Keeping both fills
the output with logical duplicates and inflates the multiple-testing count for
nothing.

**Closed-pattern reduction.** If a 3-predicate rule covers *exactly* the same
rows as a 2-predicate rule, the third predicate added no information. Keep the
simpler one.

**Result.** 102,510 possible combinations become **66,817 real candidates**.

## 9. Step five: score one rule

Every rule produces a 2x2 table. Here is the real one for our top pattern:

```
                    forecast failed   forecast fine    total
  rule is true            393              445           838
  rule is false         4,317           41,169        45,486
  total                 4,660           41,614        45,980
```

**The question.** If failures were sprinkled at random across all 45,980 hours,
how likely is it that 393 or more of them land inside these 838?

**The answer** comes from **Fisher's exact test**. The intuition is an urn: you
have 45,980 balls, 4,660 of them black (failures). You draw 838 without
replacement. What is the chance you get 393 or more black? Expected is 85. We
got 393.

```
P(393 or more) = 1.4e-166
```

**And now the single most important sentence in this document:**

> **We throw that number away as a probability.**

Fisher's test assumes rows are independent. Ours are not. So 1.4e-166 is not a
real probability &mdash; it is a *score*, useful only for **ranking** rules
against each other. All actual decisions come from step six.

## 10. Step six: the permutation null &mdash; the heart of the method

**What we do.** Take the outcome column. Cut it into **504-hour blocks** (one
week). Shuffle the *blocks*, keeping each block internally intact. Re-run the
entire search on this scrambled outcome. Record the single best score found.
Repeat 1,000 times.

**Why blocks and not rows.** A row shuffle destroys the clumpiness. A block
shuffle preserves it: within a week everything stays in order, only the weeks
move. So the scrambled outcome is *just as clumpy* as the real one, but carries
no real relationship.

**Why 504 hours.** Long enough to outlast the autocorrelation, and a whole
number of weeks so the weekday/weekend rhythm survives.

**Why only the single best score per run.** This is the clever part, and it is
called **Westfall-Young min-P**.

> On each scrambled run, the search returns thousands of rules. We keep only the
> very best one. After 1,000 runs we have 1,000 "best in a world with no
> signal" scores. That distribution answers exactly the right question: **how
> good can the luckiest of 66,817 patterns look, when nothing is real?**
>
> If your actual finding beats 95% of those, it is genuinely better than the
> best that luck produces. And this controls the error rate for the *whole
> search at once*, not per rule.

Here is what that looks like on real data:

```
scrambled runs, best score each:   -91.3 .. -17.1
the 5% cutoff (the bar):           -51.8
our top rule:                     -381.9      <- 7x past the bar
scrambled runs that beat it:            0 of 1,000
```

**Why min-P and not max-T.** You might compare raw test statistics instead of
p-values. But a rule covering 838 rows and one covering 40,000 rows produce
statistics on completely different scales, so the maximum would just pick
whichever scale runs biggest. P-values are on a common 0-to-1 scale by
construction, so they are comparable.

**Stratification.** We only exchange blocks that belong to the same calendar
month, so January weeks swap with January weeks. That stops seasonality leaking
into the null.

**The honest caveat.** Block permutation is valid *asymptotically*, under an
assumption that the series is statistically stable over time. It is not an exact
mathematical guarantee. **That is precisely why we measured the false positive
rate on 120 null datasets instead of claiming a theorem.**

## 11. Step seven: Tarone's shortcut

**The idea.** A rule covering only 3 rows can never be significant, no matter
how the outcome falls. Even if all 3 are failures, the best p-value it can
possibly achieve is still too big.

So: compute the **best score each rule could ever attain**. If even that is not
good enough, the rule is *untestable* &mdash; exclude it from the correction
count entirely.

**Why that helps.** Correcting for 66,817 tests is harsher than correcting for
10,000. Removing rules that could never have won makes the bar fairer for the
ones that could.

**The self-consistency puzzle.** The threshold depends on how many rules are
testable, and testability depends on the threshold. We solve it by sweeping the
support cutoff downward until the condition stops holding.

**Our honest result: it buys 0.36% here.** Our predicates are quantile-based, so
almost every rule covers plenty of rows and is testable. We report that as a
measured null result rather than quietly dropping the machinery. It would matter
enormously on sparse data such as genetics; it does not matter here.

## 12. Step eight: from 3,928 significant rules to 6 findings

After all that, **3,928 rules are significant**. That is 37% of the lattice, and
it is useless.

**Why so many?** Because they are variants of each other. If
`A AND B AND C` is real, then `A AND B AND D` probably is too. You have found
one effect wearing 3,928 costumes.

**The fix: greedy cover.** Take the strongest rule. Delete the failure-hours it
explains. From what remains, take the strongest rule again. Repeat.

**Intuition.** Each reported rule must earn its place by explaining something the
previous ones did not.

**One more subtlety.** Ranking purely by coverage picks trivially broad rules:
the widest significant rule is `err_lag24>q25`, covering 75% of all hours at a
lift of 1.22 &mdash; statistically real, substantively nothing. So we rank by
**lift**, over rules with enough support that lift is stable.

## 13. Step nine: significant is not the same as interesting

A rule can be rock-solid and still tell you nothing new.

If `A` alone raises the failure rate, and `B` alone raises the failure rate, then
of course `A AND B` does. You have learned nothing beyond what A and B already
told you.

**So we measure the gap between what a rule actually does and what its own parts
predict.** This follows De Bie's FORSIED framework. Build a background model in
which the analyst already knows every single-predicate effect. The
maximum-entropy completion of that knowledge &mdash; the least-assuming way to
combine what you know &mdash; turns out to be additive in log-odds:

```
surprise = logit(observed) - [ logit(base) + SUM_i (logit(p_i) - logit(base)) ]
```

**Intuition.** Predict the rule's effect by adding up its parts. Then measure how
badly that prediction misses. A big miss means the pieces are **interacting** &mdash;
doing something together that neither does alone. That is genuinely new
information.

Real example:

```
fc_ramp_3h>q75 AND fc_curv>q75 AND fc_day_range>q50
  additive model predicts:  15.9%
  actually observed:        24.5%
```

The parts predicted 15.9%, reality delivered 24.5%. **That gap is the discovery.**

**And the two rankings barely overlap:** of the top 200 by lift and the top 200
by surprise, **1 rule is in both.** Strength and interestingness really are
different questions.

---

# PART 3 &mdash; HOW WE KNOW IT WORKS

## 14. Four ways we tried to break it

**Test 1: does it find things in noise?** 120 datasets built to contain nothing.
**0.067 versus the naive method's 1.000.** This is the one that matters.

**Test 2: does it just memorise the sample?** We fitted everything on 2019-2022,
then tested on 2023 onward, which the machine had never seen.
**0 of 20 top rules held**, and lift *rose* from 4.47 to 5.18. Effects normally
shrink out of sample. Ours grew.

**Test 3: is it a PJM quirk?** We took PJM's rules to California, Texas, the
Midwest and New York &mdash; different operators, weather, generation mix.
**25 of 57 replicated**, though lift travelled to every grid (3.0 to 4.1).

**Test 4: is it just weather?** The obvious objection: bad forecast days are
storm days. We joined actual NOAA weather and re-checked each rule *within*
groups of days that had the same temperature, rain and wind.
**79% to 98% of the effect survived.**

## Market data: 36 years of VIX

The same tool, through the generic CSV adapter, on FRED's VIX series 1990-2026.
No code shared with the electricity pipeline. Outcome: when does the naive
"tomorrow's volatility equals today's" forecast fail?

| block length | blocks | SIEVE | conventional |
|---|---:|---:|---:|
| 21 days | 455 | 0.863 | 0.988 |
| 63 days | 151 | 0.500 | 0.988 |
| 126 days | 75 | 0.362 | 0.988 |
| 252 days | 37 | 0.212 | 0.988 |
| **504 days** | 18 | **0.062** | 0.988 |

**The conventional correction reports a discovery on 98.8% of market datasets
containing no signal.** 9,572 trading days, 90,307 candidate rules.

### It rediscovered volatility clustering, and we verified that rather than asserting it

Top finding, lift **5.57**:
`VIX_lag1>q90 AND perr_lag1>q90 AND perr_lag2>q90` -- yesterday's absolute VIX
change was extreme, and the day before was too, so today's will be.

That is Mandelbrot's 1963 definition verbatim: *large changes are followed by
large changes, of either sign*. Tested:

| test | result |
|---|---|
| ACF of **signed** dVIX, lags 1/2/5 | -0.125, -0.035, 0.008 |
| ACF of **absolute** |dVIX| | **0.366, 0.364, 0.283** |
| Ljung-Box Q(10) on |dVIX| | **7,446.7**, p = 0 |
| **Engle ARCH LM test** | **1,817.8** on 10 df, p = 0 |

Engle's 1982 ARCH test is the one the 2003 Nobel was awarded for. No finance was
supplied to the miner.

And it is not merely restating persistence: conditioning on the previous two days
alone gives **4.09x**; SIEVE's conjunction gives **5.57x**.

### The block length cannot be inferred from the ACF

VIX autocorrelation dies out by **126 days**. 126-day blocks still give a **36%
error rate**. It needs **504 days**, four times longer, because volatility
clustering carries dependence a simple ACF does not see.

**This is the practical result.** Every data-snooping correction in quantitative
finance -- White's Reality Check, Hansen's SPA, Romano-Wolf StepM, PBO -- runs on
a block or stationary bootstrap and needs a block length. Romano-Wolf has a
section titled "Choice of Block Sizes". Nobody publishes how they chose theirs.
If the block is shorter than the dependence horizon the correction is not doing
what the paper says, and nothing errors, so you never find out.

*Artifact: `calib_var.json`.*

## 15. Does it work outside this dataset?

Those five grids are five slices of **one** dataset. So we pointed the tool at
NOAA daily weather &mdash; different agency, different domain, daily instead of
hourly, no shared code.

It found that heavy rain yesterday plus a very warm yesterday plus recent
instability predicts a big temperature swing today. **That is a frontal
passage.** Nobody supplied any meteorology.

**But the important result is this one:**

```
                    outcome clumpiness    naive method's error
PJM, hourly              0.654                   1.84x too many
NOAA, daily              0.028                   1.03x
```

On daily weather the outcome is barely clumpy at all, so there is nothing for
block permutation to protect, and both methods agree to within 3%.

**The correction scales with the problem it exists to fix, and disappears when
that problem is absent.** The tool does not always disagree with the simple
approach. It disagrees exactly when, and in proportion to how much, the simple
approach is wrong. **It also tells you when you do not need it.**

---

# PART 4 &mdash; THE ENGINEERING

## 16. Why bitsets

A rule is "which hours does this cover?" &mdash; 45,980 yes/no answers. Store
each as one **bit**, and 45,980 hours fit in 724 64-bit words.

Now `A AND B` is 724 bitwise-AND instructions. Counting how many hours a rule
covers is `popcount`, a single CPU instruction that counts set bits.

**Intuition.** We turned "compare two lists of 45,980 things" into "724
machine instructions." That is the whole reason this is feasible at all.

## 17. The GPU journey, including the two things that failed

This is worth telling as a story, because the useful lesson is the failures.

**The problem.** The rule matrix is 66,817 x 724 words = **387 MB**. The naive
GPU port streams all 387 MB *once per permutation*. With 1,000 permutations that
is 387 GB of traffic.

**Fix 1: reuse, not parallelism.** Hold the matrix on the card and test a
**batch of 32** scrambled outcomes against each word as it is loaded. The 387 MB
is read once per 32 permutations instead of once each.

Why exactly 32? Because arithmetic intensity is B/4 popcounts per byte, and
against the bandwidth this card actually delivers the crossover from
memory-limited to compute-limited sits at **B = 32.2**. Below it you are
starved; above it you gain nothing. **The batch size is derived, not guessed.**

**Where that left us: 19% of the roofline.** The roofline is the speed limit
imposed by the hardware. Well-tuned kernels reach 65-85%. We were at 19%, which
means something was badly wrong.

**Fix 2: one warp per rule.** The original gave each rule to one thread. Thread
*i* reads a word 5,792 bytes away from thread *i+1*, so every instruction was a
32-way scatter across memory. Giving each rule to a **warp** (32 threads working
together) means the 32 threads read 32 *consecutive* words &mdash; one coalesced
transaction instead of 32 scattered ones.

**1.65x on the kernel. And only 1.05x end to end.** Which sent us to the
profiler instead of to another kernel idea.

**The profiler said we were optimising the wrong thing:**

```
generating the permutations (on the CPU)   53%   <- the actual bottleneck
the statistical scan (on the CPU)          31%
the GPU kernel                             15%
```

**Fix 3: cache the permutation structure.** The code rebuilt 275 array slices
and ran a Python loop **on every single draw**, even though none of that
structure changes between draws. Caching it: **4.67x**, and because the random
number calls happen in the same order, the draws are bit-identical and nothing
downstream moved. End to end: 0.757 to 0.597 ms.

**Failure 1: shared memory was slower.** The scrambled-outcome matrix is re-read
by every warp, so staging it in fast on-chip memory looked obvious. It was
**6% slower**. Why: the matrix is only 185 KB and already lives in the card's 48
MB L2 cache, so we saved no real traffic and paid for a synchronisation barrier
on every tile. **Reverted, and kept in the repository so the result is
reproducible.**

**Failure 2: the parallel scanner.** There was an unused `parallel=True` routine
in our own code for scanning a whole batch at once. Wiring it in made that stage
**3.6x slower** &mdash; it oversubscribes the 6 CPU cores the cloud container
actually allocates. **Left unused, with a comment explaining why.**

**Fix 4: the biggest win was not about memory at all.** The batch size `B` was a
runtime argument, so the compiler could not unroll the inner loop or prove the
accumulator index was constant. Making `B` a compile-time constant: **1.53x**,
more than the coalescing fix gave.

**Where it ended:**

```
                          ms/batch     GB/s    % of roofline
one thread per rule          8.691     44.5        19.0%
one warp per rule            5.253     73.7        31.4%
warp + shared memory         5.616     68.9        29.3%   FAILED
warp + compile-time B        3.430    112.9        48.0%
```

**2.53x on the kernel, 19% to 48% of the hardware limit.** Every version
produces bit-identical output; that is asserted by a test, not assumed.

End to end: **0.521 ms per permutation, 112x faster than our CPU path.**

**The honest part.** 48% is respectable, not excellent. About 1.4x remains in the
kernel &mdash; but the kernel is now 21% of the time, so even making it
instantaneous buys 1.26x. **The remaining work is on the host side, not the
GPU.** And our CPU baseline is single-threaded unfused numpy, so against a
properly tuned CPU implementation the gap would be far smaller.

**The GPU is not the contribution.** GPU permutation testing was published in
2014 (PBOOST) and GPU Westfall-Young in 2015 (HWY). The GPU is what makes a
12-second demo possible. The contribution is that the statistics are right.

---

# PART 4b &mdash; THE DATA WAS WRONG, AND HOW WE FOUND OUT

The night before submission, a literature sweep turned up a claim that PJM files
its EIA-930 forecast on a shifted timestamp. We checked.

**It is true.** PJM's forecast leads its realised demand by exactly one hour from
2019-01-30 onward. We had been computing error against the wrong hour, and about
a third of what we called "forecast error" was our own misalignment.

**The trap.** Lower error under a shift is equally consistent with "the forecast
is early" and "demand is late" &mdash; and those need OPPOSITE fixes. If demand
were the displaced column, every lagged feature and the whole outcome
stratification would be pinned to the wrong hour. Two structural tests settle it:

1. **Geography.** On first differences, PJM's *demand* leads Eastern peers at
   lag 0 and Central peers at lag -1. That is real geography, and it proves the
   test can resolve one hour. PJM's *forecast* leads every peer at -1
   **regardless of timezone**. Timezone invariance means the offset is a label,
   not a physical lead.
2. **Conservation.** `Demand = NetGen - Interchange` holds for PJM at lag 0 with
   a **7 MW** median residual and 17.4% exact zeros. Demand is welded to PJM's
   own generation. So it is the forecast that moved.

**It is PJM alone** &mdash; 51 of 55 balancing authorities are optimal at shift 0
on the same loader. **It starts on a date** &mdash; before 2019-01-30 the series
is correctly aligned and blanket-shifting it manufactures false positives. And
**four rows carry 2^31 as a demand value**, an integer overflow sentinel that
dragged mean demand from 92,518 to 153,448 MW.

**What it cost.** Top lift 4.61 to 2.97. Significant rules 24,956 to 3,928.

**What it bought.** The naive method's inflation went from 1.78x to **7.86x**.
Fixing our own defect made the central claim four times stronger.

**Why this is the most important part of the whole project.** No amount of GPU
optimisation would have found it. No amount of statistical rigour would have
found it. It was found by reading the domain literature and then checking a
claim against the data. That is the lesson.

## Block length must match the dependence horizon

The single most useful thing we learned, and we learned it by getting it wrong.

| grid | outcome ACF at 168h | FPR at 504h blocks | calibrated block | FPR there |
|---|---:|---:|---:|---:|
| PJM | 0.007 | 0.067 | 504h | 0.067 |
| MISO | 0.084 | 0.083 | 504h | 0.083 |
| **CISO** | **0.211** | **0.850** | **4368h (182 days)** | **0.033** |

Same code, same procedure, one parameter. CISO's dependence runs six times longer
than PJM's, so three-week blocks destroy structure that is really there, the null
becomes too easy, and the method fires on **85% of datasets containing nothing**.
Give it six-month blocks and it calibrates at 0.033.

**Block length is a measurement, not a convention.** This is the same failure mode
as choosing a bootstrap block length badly in a backtest, which is why it should
travel outside this domain.

# PART 5 &mdash; WHAT IS WRONG WITH IT

An independent audit confirmed 24 defects. The ones that matter:

**The data window.** The energy agency renamed some columns in mid-2024. Our
loader asks for the old names, and an unguarded "drop rows with any missing
value" then silently deleted **every row after 2024-07-02 &mdash; 31.5% of the
data.** The statistics are unaffected. The window is smaller than we said.

**A blind spot.** Two time-of-day features are *constant* inside each (hour,
month) cell, so all eight of their predicates are always false. The lattice has
85 named predicates but **77 that can ever fire**, and **no time-of-day rule is
expressible at all.**

**Forecast vintage.** The operator files one forecast covering all 24 hours of
the next day. Our code assumes a uniform 24-hour lead, which is not exactly
right, so lag-24 features are partly look-ahead relative to the filing moment.
We tested it: restricting to strictly-safe predicates drops 94% of the lattice
and lift falls 4.61 to **3.52**. **The effect survives**, and one surviving rule
uses only the forecast's own shape, so it is safe whatever the vintage.

**How much depends on our choices.** Changing how the permutation is stratified
moves the significant count between **7,991 and 26,005** &mdash; a factor of 3.3.
We report the range rather than one number. **What survives every version is the
comparison itself**: the naive method gives 30,883 regardless, so it is 1.7x to
5.5x too generous, and it looks *worse* under the stricter choices.

---

# The one thing to remember

> Mine a dependent time series with conventional multiple testing and you will
> ship false discoveries. On data containing nothing, the conventional approach
> reported a discovery **120 times out of 120**.

That is the case. Not "here is a better number," but **"the thing you would
otherwise have done fails completely, and here is the measurement."**


---

# PART 6 &mdash; THE CODE, FILE BY FILE

1,571 lines across nine live files. What each one does, and the one idea in it
that matters.

## `core.py` &mdash; 332 lines &mdash; the engine

| function | what it does | the idea |
|---|---|---|
| `pack` / `popcount` | bool array <-> 64-bit bitset | 46,324 hours fit in 724 words; intersecting two rules is 724 AND instructions |
| `FisherTable` | right-tailed Fisher p for fixed margins | under permutation the margins never change, so p depends only on (support, positives) |
| `tarone_threshold` | the self-consistent testability cutoff | delta = alpha/k(delta) solved by a **monotone sweep**, because naive iteration oscillates |
| `block_permutation` | the audited circular block draw | the ragged tail is **pinned**: if it joins the shuffle, every row after it shifts and the stratum assignment silently breaks |
| **`BlockPermuter`** | the cached version | rebuilding 275 slices per draw was **53% of wall time**; caching gives 4.67x with bit-identical draws |
| `Sieve` | support, Tarone cutoff, testable mask | |

## `fastp.py` &mdash; 173 lines &mdash; the inner loop

Runs 66,817 times per permutation, 1,000 permutations deep. Two ideas:

**Log-factorials, precomputed once.** Every term of the hypergeometric tail then
costs a few adds, and consecutive terms come from a recurrence, so each extra
term is one multiply.

**The admissible early exit.** `P(A>=a) >= P(A=a)`, so the first term alone lower-bounds
the tail. For min-P we only want the smallest p, so if that bound already exceeds
the running best, the rule cannot win and the tail is never summed. **Prunes
99.91%**, and it is exact &mdash; pruned rules provably cannot be the minimum.

It also carries a correctness guard worth knowing: the hypergeometric support is
`max(0, x+n1-n) .. min(x, n1)`, and starting the recurrence below the floor
divides by zero. That bug was latent for weeks before an audit found it.

## `features.py` &mdash; 300 lines &mdash; EIA-930 to a lattice

Builds the outcome (top decile within each hour x month cell), the predicates
(quantile cuts, also within-cell), and the rule lattice with three prunings:
apriori, same-feature exclusion, closed-pattern reduction.

Also carries the **PJM timestamp correction**, date-gated at 2019-01-30, and the
guard against the four rows reading 2^31 MW.

## `adapter.py` &mdash; 120 lines &mdash; any CSV

The generalisation. Takes any time-indexed table and frames the question as *when
does the naive persistence forecast fail*: `outcome(t) = 1` if
`|target(t) - target(t-h)|` is in the top decile of its stratum cell. This is what
ran on NOAA weather and on 36 years of VIX with no changes.

## `backends.py` &mdash; 136 lines &mdash; CPU and GPU share one set of permutations

Both devices consume the **same index matrix** from `core.block_permutation`, so
a given seed gives bit-identical results either way and `--device` is a speed
knob rather than a different experiment. This was not free: the GPU module
shipped its own vectorised generator that **ignored the stratification**, so the
two devices had been sampling from different nulls.

## `engine.py` &mdash; 176 lines &mdash; orchestration

`Problem.null()` draws J permutations and batches them to the backend.
`cutoff()` is an **order statistic**, not `np.quantile` &mdash; interpolating gave
an achieved rate of 0.0565 against a nominal 0.05. `cover()` reports a
non-redundant set by lift over a support floor.

## `gpu_kernel_u.py` &mdash; 76 lines &mdash; the kernel

One **warp** per rule, so 32 lanes read 32 consecutive words and coalesce
(the original gave one *thread* per rule, making every load a 32-way scatter).
The batch size is a **compile-time literal** so the popcount loop unrolls &mdash;
worth 1.53x on its own, more than the coalescing fix. 48% of the binding roofline.

Three other kernels are kept in the tree because their results are worth
reproducing: `gpu_kernel.py` (the original), `gpu_kernel_warp.py` (coalesced, no
unroll), and `gpu_kernel_smem.py` (shared-memory staging, **6% slower**, kept as
a documented negative result).

## `sieve_cli.py` and `app.py` &mdash; 258 lines &mdash; the two front ends

The CLI is the deliverable; the Flask page wraps it. Both call the same audited
code, and the page runs the real engine live rather than replaying a recording.

## The test suites

`test_core.py` (calibration, power, permutation validity), `test_all.py`
(18 regression tests covering both historical bugs), `test_backends.py`
(batch-size invariance, stratification, CPU/GPU equality), `test_gpu_equal.py`
(end-to-end device equality), `kernel_ab.py` (all four kernels bit-identical),
`verify_all.py` (every change made during the final audit).


---

# PART 7 &mdash; EVERY ENGINEERING CHOICE, AND WHY

For each: what we chose, what we rejected, and what it cost or bought. Several of
these were decided by measurement after we guessed wrong.

## Statistical choices

**Bitsets, not arrays.** A rule is 46,324 yes/no answers. Stored as bits that is
724 64-bit words, so intersecting two rules is 724 AND instructions and counting
coverage is a hardware `popcount`. *Rejected:* boolean arrays (8x the memory,
no popcount), sparse sets (the rules are dense, so sparsity costs more than it
saves). *Bought:* the whole thing becomes feasible at all.

**Fisher's exact test as a ranking statistic only.** *Rejected:* using its
p-value for inference. The hypergeometric assumes exchangeable rows, and ours are
autocorrelated at 0.72, so the number is not a probability. *Cost:* we throw away
a p = 1.4e-166 that looks decisive. *Bought:* every claim we make is actually
valid.

**min-P, not max-T.** Westfall-Young comes in both forms. *Why min-P:* rules of
different support produce Fisher statistics on non-comparable scales, so a
maximum over raw statistics is dominated by whichever support happens to generate
large numbers. p-values are on a common [0,1] scale by construction.

**Circular block permutation, not the stationary bootstrap.** *Why:* the
stationary bootstrap draws geometric block lengths, which is better theory but
makes the null a mixture over lengths &mdash; and our central finding is that the
*length* is what matters. A fixed length makes the parameter visible and
measurable. *Cost:* asymptotic rather than exact validity, which is why we
measure the false positive rate instead of claiming a theorem.

**Kept Tarone despite it contributing 0.36%.** *Why:* it is exact, it costs
almost nothing, and the honest thing is to report a measured null result rather
than quietly delete machinery that did not help here. It would matter enormously
on sparse data such as genetics.

**Order statistic, not `np.quantile`, for the cutoff.** `np.quantile`
interpolates and divides by J rather than J+1. Measured over 4,000 trials the
achieved rate was **0.0565 against a nominal 0.05** at J=200. The order statistic
gives 0.0512. *Cost:* one line. *Bought:* the procedure does what it says.

**Greedy cover ranked by lift, not by coverage.** *Why:* ranking by coverage
alone picks trivially broad rules &mdash; the widest significant rule on PJM
covers 75% of all hours at a lift of 1.22, which is statistically real and
substantively nothing. *Cost:* a support floor, below which lift is unstable.

**Date-gated the timestamp correction.** *Why:* the offset begins 2019-01-30, and
before that the series is correctly aligned. Shifting those 499 rows anyway takes
their positive rate from 4.0% to 24.5% against a 10.1% base, manufacturing false
positives into the January cells.

**Calibrated the block length per series.** *Why:* we used one length everywhere
and CISO came back at a **0.850** false positive rate. The dependence horizon is
a property of the series, not a convention. *Cost:* a calibration run per
dataset. *Bought:* the central claim actually holds.

## Engineering choices

**CPU and GPU share one permutation generator.** *Rejected:* the GPU module's own
vectorised generator, which is faster but **ignores the stratification** and pins
the ragged tail. The two devices were sampling from different nulls. *Cost:* 4.3
ms per batch of 32, against a 5.6 ms kernel. *Bought:* `--device` is a speed knob,
not a different experiment, asserted bit-identical in `test_backends.py`.

**One warp per rule, not one thread.** One thread per rule means adjacent lanes
read words 5,792 bytes apart &mdash; a 32-way scatter on every load. A warp per
rule reads 32 consecutive words as one coalesced transaction. *Bought:* 1.65x on
the kernel. *Learned:* only **1.05x end to end**, which is what sent us to the
profiler instead of to another kernel idea.

**Compile-time batch size.** `B` was a runtime kernel argument, so
`for (b = 0; b < B; ++b)` could not unroll and `acc[b]` could not be proven
constant-indexed. Baking it in as a literal and caching one NVRTC module per
size. *Bought:* **1.53x &mdash; the single biggest kernel win**, more than the
coalescing fix. *Cost:* one compiled module per distinct batch size.

**Batch of exactly 32.** Not arbitrary. Arithmetic intensity is B/4 popcounts per
DRAM byte, so against the 235 GB/s this card actually delivers the roofline knee
sits at **B = 32.2**. Below it you are bandwidth-starved; above it you gain
nothing. Also the kernel's `int acc[32]` register budget.

**Cached the permutation structure.** The profiler said the kernel was 15% of
wall time and **generating permutations was 53%**. The code rebuilt 275 array
slices and ran a Python loop on every single draw, though none of that structure
depends on the draw. *Bought:* 4.67x, and because the rng call order is unchanged
the draws are bit-identical so nothing downstream moved.

## Two things we tried and rejected, with numbers

**Shared-memory staging of the label matrix.** The reasoning was sound: `ys` is
re-read by every one of 66,817 warps, so staging it per block should cut traffic
8x. **It was 6% slower** (5.616 vs 5.253 ms). Why: at 185 KB it already lives in
the L4's 48 MB L2, so no real traffic was saved, and the per-tile
`__syncthreads()` cost more than nothing. Kept in the tree as
`gpu_kernel_smem.py` so the result is reproducible.

**The parallel batch scanner.** `fastp.minp_null_fast` is a `parallel=True`
routine for scanning a whole batch at once, and it had never been called. Wiring
it in made that stage **3.6x slower** (1.684 vs 0.469 ms/perm) because it
oversubscribes the 6 vCPU the container actually allocates while `nproc` reports
48. Left unused, with the reason recorded.

## What these two failures are worth

Three of six engineering predictions were wrong. **That is the argument for
profiling**, and it is exactly what a speedup number cannot tell you. Last year's
entry reports a warp-primitive variant that came out 36% slower than the version
before it, with no explanation offered &mdash; which is what happens when you
optimise without a roofline.
