# SIEVE, explained simply

For the team. No statistics background needed. About ten minutes.

---

## 1. The problem we are solving

Suppose you search a dataset for patterns. You try one idea, it looks good.
But what if you quietly tried sixty-six thousand ideas and showed only the best
one?

Put a million people in a room, have each flip a coin ten times, and about 976
of them get ten heads in a row. None of them is psychic. They are the lucky tail
of a big room.

**Every automated pattern-finder does this.** Power BI, Tableau, ThoughtSpot:
they test thousands of ideas and show you the winners without telling you how
big the room was.

**SIEVE tells you how big the room was.**

---

## 2. What we actually built

We took eight years of US electricity grid data, hourly, free and public. It
contains something unusual: **each grid operator's own forecast of tomorrow's
demand, sitting next to what actually happened.** So we can ask when their
forecast fails.

Then we let the machine write the rules itself. Nothing was hand-written:

* **22 things we can measure** (forecast level, how steeply it ramps, yesterday's
  error, coal/gas/solar share, demand last week, and so on)
* each cut at **4 thresholds** -> **85 yes/no conditions**
* every combination of one, two or three of them -> **66,844 rules**

One rule looks like: *IF the forecast says today will swing a lot AND yesterday's
error was in the worst 10%.*

We test all 66,844.

![The funnel](fig1_funnel.png)

---

## 3. How we tell real from lucky

This is the whole trick, and it is simple.

**Step 1.** Score every rule on the real data. The best scores **-1045**
(more negative = stronger).

**Step 2.** Now destroy the truth. Shuffle the answer column in week-long blocks
so nothing *can* genuinely be related. Re-run the entire search over all 66,844
rules. Write down the best rule that fake world produced.

**Step 3.** Do that a thousand times. Now you know exactly what luck alone
produces.

![The null distribution](fig2_null.png)

In a world where nothing is true, the luckiest of 66,844 rules scores about
**-51**. Our real best scores **-1045**. Not close. It is real.

**26,715 rules clear the bar. 40,129 do not** -- they look promising and are
exactly what luck produces when you try 66,844 things.

---

## 4. Why the week-long shuffle matters

The obvious way to shuffle is hour by hour. That is wrong here, and the error is
big.

Hours are not independent -- 2pm looks a lot like 1pm. Shuffling individual hours
builds a world that is **more random than reality**, which sets the bar too low
and lets rubbish through.

We proved it. We fed the machine 120 datasets containing **nothing at all**:

![Two nulls](fig6_twonulls.png)

A correct method should cry wolf about 5% of the time. Ours does.
**The standard shortcut cries wolf on every single dataset.**

---

## 5. What we found

> **When the operator's own forecast predicts a big daily swing AND yesterday's
> error was in the worst 10%, today's forecast fails 4 to 5 times more often
> than normal.**

Both facts are known **before the forecast is submitted**, so this is a
prediction, not hindsight.

### We then tried hard to break it

**Hid the future.** Found the rule using only 2019-2022, tested on 2023-2026
which the machine had never seen. **20 of 20 rules still worked**, and the effect
got *stronger*, not weaker.

**Took it somewhere else.** Tested the mid-Atlantic rules on four completely
different grids -- different operators, weather, power plants:

![Replication](fig3_replication.png)

**57 of 58 replicated.**

**Checked whether it is just weather.** The obvious objection: bad forecast days
are probably storm days. We joined real NOAA weather records and re-checked the
rule *within* groups of days that had the same temperature, rain and wind:

![Weather](fig5_weather.png)

**79% to 98% of the effect survives.** Weather explains only a small slice.

---

## 6. The surprising part

We also searched for **reversals** -- cases where something that normally makes
forecasts worse appears to make them dramatically better. Analysts love these.

**We found 1,111 of them. One showed error dropping 68%.**

**Not a single one is real.** Zero survive the correction.

That is the finding worth leading with: *this dataset contains a genuine pattern
worth 4-5x and 1,111 fake ones that look more impressive, and you cannot tell
which is which by looking.*

---

## 7. Speed

Doing this honestly is expensive -- you re-run the whole search a thousand times.
So we made it fast, in four steps, each measured on the same machine:

![The ladder](fig4_ladder.png)

**4.7 minutes of CPU work becomes 4.8 seconds.** That is the difference between
starting it and walking away, and running the whole thing live in front of
someone.

---

## 8. What is NOT true

Being straight about the limits, because we will be asked:

1. **We know *when* forecasts fail, not *why*.** Weather is largely ruled out,
   but we have not replaced it with an explanation.
2. **The headline rule is real but not surprising.** "Yesterday was bad and today
   looks volatile" is two obvious facts together. The *surprising* result is the
   1,111 fakes.
3. **It is engineering, not new mathematics.** Every statistical piece is from
   1990-2015. What is ours is making it fast and proving it works.
4. **One of our six optimisations does nothing** on this data (0.55%). We report
   that rather than hide it.

---

## 9. Honesty log

We found and fixed **three of our own bugs**, and said so each time:

| bug | effect | after fixing |
|---|---|---|
| divide-by-zero in the maths | never fired on our data, but latent | 400/400 exact |
| the test period helped set its own thresholds | 3.5% of labels contaminated | holdout went 19/20 -> **20/20** |
| tested reversals with the wrong-direction test | would have reported a fake null | re-tested correctly, answer held |

Every fix made the result **stronger**. That is worth saying out loud.

---

## 10. One sentence

> *"This grid data contains one real pattern worth 4-5x and 1,111 fake ones that
> look better. We built the machine that tells them apart, proved it works by
> feeding it 120 datasets containing nothing, and it is 3,134x faster than the
> obvious implementation."*
