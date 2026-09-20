# Does it work on anything other than EIA-930?

## The honest starting position

PJM, CISO, ERCO, MISO and NYIS are five balancing authorities from **one
dataset**: same agency, same schema, same outcome definition, same predicate
construction. "57 of 58 replicate" is replication across *regions*, not across
*datasets*, and it should be described that way.

So we pointed the tool at something genuinely different.

## The second dataset

**NOAA GHCN-Daily**, Philadelphia International Airport, 1970 to 2026.

| | EIA-930 | NOAA GHCN-Daily |
|---|---|---|
| agency | Energy Information Administration | NOAA |
| domain | electricity demand | meteorology |
| cadence | **hourly** | **daily** |
| rows | 46,324 | 20,697 |
| schema | balancing-authority generation and demand | station weather observations |
| outcome | day-ahead operator forecast fails | **persistence forecast fails** |

No code in `features.py` was reused. `adapter.py` builds predicates from any
time-indexed CSV, with the outcome defined as *the naive forecast failing*:

```
outcome(t) = 1  if  |target(t) - target(t-h)|  is in the top decile
                    of its own stratum cell
```

For EIA the naive forecast is the operator's own filed forecast. For an
arbitrary series it is persistence. That is one substantive assumption, stated
rather than hidden.

## It runs

```
20,697 days, 2,231 positive (10.8%)
125 predicates -> 119,136 rules (309 MB)
Tarone prunes 3.74%      (vs 0.36% on PJM)
```

And the rules are physically sensible. Top finding, lift 3.82:

> precipitation yesterday in the top quartile, AND yesterday's maximum
> temperature in the top decile, AND the recent persistence error already
> elevated

That is a warm, wet air mass followed by a frontal passage. Nobody told the
miner any meteorology.

## The result that actually matters

| | outcome ACF lag 1 | lag 2 | lag 3 | naive / block inflation |
|---|---:|---:|---:|---:|
| EIA-930 PJM, hourly | **0.654** | 0.432 | 0.290 | **1.84x** |
| EIA-930 ERCO, hourly | | | | **4.09x** |
| EIA-930 MISO, hourly | | | | 3.89x |
| EIA-930 CISO, hourly | | | | 2.54x |
| EIA-930 NYIS, hourly | | | | 2.45x |
| NOAA daily | **0.028** | 0.030 | 0.003 | **1.03x** |

**The correction is proportional to the dependence it corrects for.**

On hourly electricity demand the outcome is strongly autocorrelated, the naive
row shuffle destroys that structure, and it over-reports by 1.8x to 4.1x. On
daily weather the outcome is already close to white noise, there is almost
nothing for the block structure to preserve, and block permutation and row
shuffle agree to within 3%.

This is a **dose-response relationship**, and it is much stronger evidence than
"our method produces different numbers." The method does not always disagree
with the naive one. It disagrees exactly when, and to the degree that, the
assumption behind the naive one is violated.

**It also tells you when you do not need it.** On the NOAA task, a plain row
shuffle would have been very nearly adequate, and SIEVE says so.

## What this does not show

- One station, one target variable. Not a survey of weather data.
- The block length for NOAA (30 days) was chosen by hand, not by the ACF
  procedure used for EIA. Given an ACF of 0.028 at lag 1 almost any block
  length gives the same answer, which is the point, but it was not derived.
- Still only two dataset families. A third from a different discipline
  (finance, industrial process, epidemiology) would be the natural next step.
