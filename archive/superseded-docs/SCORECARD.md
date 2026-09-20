# How good is this, in plain terms

## The one-paragraph version

We built a machine that searches data for patterns and then tells you honestly
whether each one is real or just luck. We found a rule about when electricity
grid operators' own forecasts fail. Then we tried to break it four different
ways: we hid recent years from ourselves and re-tested, we took it to four
completely different power grids, we checked whether the weather was secretly
doing the work, and we ran the whole thing on data we had deliberately
scrambled so nothing could possibly be real. **It survived all four.**

## What we found

> When the operator's own day-ahead forecast predicts an unusually large daily
> swing **and** yesterday's forecast error was in the worst 10%, today's
> forecast fails about **4 to 5 times** more often than normal.

Both facts are known **before the forecast is submitted**, so this is a
prediction, not hindsight.

## How hard we tried to break it

**Test 1 -- hide the future.** We found the rule using only 2019 to 2022, then
tested it on 2023 to 2026, which the machine had never seen.
**19 of the top 20 rules still worked.** The effect got *stronger*, not weaker
(lift 4.38 -> 5.10). Normally a finding shrinks when you take it out of sample.
Ours did not.

**Test 2 -- try it somewhere else entirely.** We took the rules found in PJM
(the mid-Atlantic grid) and tested them on California, Texas, the Midwest and
New York. Different operators, different weather, different power plants.
**57 out of 58 replicated.**

**Test 3 -- is it just weather?** This was the big worry: bad forecast days are
probably storm days, so maybe we just rediscovered "storms are hard to
predict". We joined actual NOAA weather records and re-checked the rule
*within* groups of days that had the same temperature, rain and wind.
**79% to 98% of the effect survived.** Weather explains only a small slice.

**Test 4 -- feed it nothing.** We scrambled the data 120 different ways so
there was genuinely nothing to find. A correct machine should report a false
alarm about 5% of the time.
**Ours: 6.7%. The naive version everyone else uses: 100%.** It cried wolf on
every single scrambled dataset.

## How fast

| | time for 10,000 permutation tests |
|---|---|
| plain Python | 5 hours |
| optimised CPU, 14 cores | 4.7 minutes |
| our GPU pipeline | **4.8 seconds** |

**58x faster than a fully parallel CPU**, end to end. Not the kernel in
isolation -- the whole pipeline, same statistic, same answer.

## What is still weak, honestly

**We know *when* forecasts fail, not *why*.** Weather is mostly ruled out as
the cause, but we have not proven the mechanism. We can say "this predicts
failure", not "this causes failure".

**Tarone pruning, one of our six optimisations, does nothing here.** It
contributes 0.55%. It prunes rare patterns, and our patterns are common. We are
reporting that as a measured null rather than quietly dropping it, and we are
not going to manufacture a use case for it.

**It is engineering, not new mathematics.** Every statistical piece dates from
1990 to 2015. What is ours is making it fast enough to use, and the honesty
machinery around it.

## The grade

| | |
|---|---|
| Is the engine correct? | Yes. 18/18 tests, GPU matches CPU exactly |
| Is it calibrated? | Yes. 6.7% against a 5% target on 120 real nulls |
| Does the finding hold out of sample? | Yes. 19/20 |
| Does it replicate elsewhere? | Yes. 57/58 |
| Is it just weather? | No. 79-98% survives weather controls |
| Is it fast? | Yes. 58x end to end, honestly measured |
| Is the mechanism proven? | **No** |
| Is it new mathematics? | **No, and we say so** |

Six yes, two honest no.
