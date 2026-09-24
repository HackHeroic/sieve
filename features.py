"""
EIA-930 -> forecast-time-available features + binary outcome + rule lattice.

THE RULE THAT GOVERNS THIS FILE
-------------------------------
A day-ahead forecast for hour t is filed at roughly t-24h.  Therefore any
feature used to predict its failure must be knowable at t-24h.  Realised
solar share / ramp / interchange AT HOUR t are NOT, so using them would make
this attribution rather than prediction, and the honest answer to "what could
the operator have done?" would be "nothing".

Every feature below is either
  (a) the day-ahead forecast itself, or a shape derived from it, or
  (b) a realised quantity lagged >= 24h, or
  (c) a calendar term.
"""

from __future__ import annotations

import glob
import itertools
import os

import numpy as np
import pandas as pd

from core import pack

DATA = os.path.join(os.path.dirname(__file__), "data")

# EIA renamed these from the 2024 Jul-Dec file onward:
#   "Solar" -> "Solar with/without Integrated Battery Storage"
#   "Wind"  -> "Wind  with/without Integrated Battery Storage"
#   "Hydropower and Pumped Storage" -> "Hydropower Excluding Pumped Storage" + "Pumped Storage"
# Asking for the old names and then dropping rows with any NaN silently deleted
# every hour after 2024-07-02 -- 31.5% of the data, with no error and no log line.
# Resolve by PREFIX and sum the split variants instead.
FUELS = ["Coal", "Natural Gas", "Nuclear", "Wind", "Solar", "Hydropower"]
FUEL_PREFIX = "Net Generation (MW) from "


def _fuel_cols(columns, fuel):
    """Every column belonging to this fuel, across both schema eras."""
    want = FUEL_PREFIX + fuel
    return [c for c in columns
            if c.startswith(want) and "Imputed" not in c and "Adjusted" not in c]


def load_ba(ba: str, years=range(2019, 2027)) -> pd.DataFrame:
    """Load and concatenate one balancing authority, hourly, sorted by time."""
    frames = []
    for f in sorted(glob.glob(os.path.join(DATA, "EIA930_BALANCE_*.csv"))):
        yr = int(os.path.basename(f).split("_")[2])
        if yr not in years:
            continue
        df = pd.read_csv(
            f, low_memory=False, thousands=",",
            usecols=lambda c: (
                c in {"Balancing Authority", "UTC Time at End of Hour",
                      "Demand Forecast (MW)", "Demand (MW)",
                      "Net Generation (MW)", "Total Interchange (MW)"}
                or (c.startswith(FUEL_PREFIX) and "Imputed" not in c
                    and "Adjusted" not in c)),
        )
        df = df[df["Balancing Authority"] == ba]
        if len(df):
            frames.append(df)
    if not frames:
        raise SystemExit(f"no rows for {ba}")

    d = pd.concat(frames, ignore_index=True)
    d["t"] = pd.to_datetime(d["UTC Time at End of Hour"], errors="coerce",
                            format="mixed", utc=True)
    d = d.dropna(subset=["t"]).sort_values("t").drop_duplicates("t")

    num = [c for c in d.columns if c not in ("Balancing Authority",
                                             "UTC Time at End of Hour", "t")]
    for c in num:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # regular hourly index so lags mean what they say
    d = d.set_index("t").asfreq("1h")
    return d


def build(ba: str, years=range(2019, 2027), q: float = 0.90, fit_until_year=None,
          fc_shift: int = 0):
    """
    Returns (X dataframe of binary predicates, y bool outcome, meta dict).

    Outcome: absolute day-ahead demand forecast error at hour t, flagged if it
    is in the top q of its OWN (hour-of-day, calendar-month) cell.  Stratifying
    by month as well as hour is what stops seasonality leaking into the null.
    """
    d = load_ba(ba, years)

    fc = d["Demand Forecast (MW)"]
    dem = d["Demand (MW)"]
    # --- corrupt rows -------------------------------------------------
    # PJM has 4 rows with impossible demand, including 2021-10-19 04:00 UTC at
    # 2,147,480,000 MW -- that is 2**31, an integer overflow sentinel.  They drag
    # mean PJM demand from 92,518 to 153,448 MW and destroy any MAE-based
    # statistic.  MAPE is barely affected (3.6205 vs 3.6115), which is why they
    # went unnoticed.
    dem = dem.mask(dem > 3e5).mask(dem <= 0)

    if fc_shift:
        # PJM's EIA-930 forecast series LEADS its realised demand by exactly one
        # hour, from 2019-01-30 onward.  Measured, not assumed:
        #
        #   * 51 of 55 balancing authorities are optimal at shift 0 on this same
        #     loader.  PJM is the only one optimal at +1.  So it is not our code.
        #   * On first differences, PJM's DEMAND leads Eastern peers at lag 0 and
        #     Central peers at lag -1 (real geography), while PJM's FORECAST
        #     leads every peer at -1 regardless of timezone.  Timezone-invariance
        #     is the signature of a label offset inside PJM's own file.
        #   * The balance identity Demand = NetGen - Interchange holds for PJM at
        #     lag 0 with a 7 MW median residual and 17.4% exact zeros, so it is
        #     the FORECAST that is displaced, not demand.  This matters: if
        #     demand were displaced the fix would be the opposite one.
        #
        # THE DATE GATE IS NOT OPTIONAL.  Before 2019-01-30 the series is
        # correctly aligned: MAPE@0 2.389 vs MAPE@+1 3.278 on those 690 hours,
        # and shift 0 wins on 28 of 28 days.  Shifting them anyway takes their
        # positive rate from 4.0% to 24.5% against a 10.2% base, manufacturing
        # 122 false positives into the January hour x month cells.
        #
        # Note fc.shift(k) computes |demand(t) - forecast(t-k)|, i.e. it pulls
        # the forecast FORWARD by k hours.  An earlier comment described this
        # with the sign inverted.
        BREAK = pd.Timestamp("2019-01-30", tz="UTC")
        shifted = fc.shift(fc_shift)
        fc = fc.where(fc.index < BREAK, shifted)

    err = (dem - fc).abs()

    f = pd.DataFrame(index=d.index)

    # (a) the forecast itself and its shape -- all known at filing time
    f["fc_level"] = fc
    f["fc_ramp_1h"] = fc.diff()
    f["fc_ramp_3h"] = fc.diff(3)
    f["fc_curv"] = fc.diff().diff()
    f["fc_vs_yesterday"] = fc - fc.shift(24)
    f["fc_day_range"] = fc.rolling(24).max() - fc.rolling(24).min()

    # (b) realised quantities, lagged >= 24h
    f["dem_lag24"] = dem.shift(24)
    f["dem_lag168"] = dem.shift(168)
    f["dem_chg_24_48"] = dem.shift(24) - dem.shift(48)
    f["err_lag24"] = err.shift(24)
    f["err_lag48"] = err.shift(48)
    f["err_roll7d"] = err.shift(24).rolling(168).mean()
    ng = d["Net Generation (MW)"]
    f["netgen_lag24"] = ng.shift(24)
    f["interchange_lag24"] = d["Total Interchange (MW)"].shift(24)
    tot = ng.replace(0, np.nan)
    for fuel in FUELS:
        cols = _fuel_cols(d.columns, fuel)
        if not cols:
            continue
        gen = d[cols].sum(axis=1, min_count=1)      # sum the split variants
        f[f"share_{fuel.split()[0].lower()}_lag24"] = (gen / tot).shift(24)

    # (c) calendar
    hour = d.index.hour
    month = d.index.month
    # hour_sin / hour_cos were CONSTANT inside every (hour x month) binarisation
    # cell, so all eight of their quantile predicates were identically False and
    # no time-of-day rule was expressible at all.  Binary indicators pass through
    # the <=2-unique-values branch below untouched, so they survive.
    f["is_morning_ramp"] = ((hour >= 10) & (hour <= 14)).astype(float)   # UTC
    f["is_evening_peak"] = ((hour >= 21) | (hour <= 1)).astype(float)
    f["is_overnight"] = ((hour >= 5) & (hour <= 9)).astype(float)
    f["is_weekend"] = (d.index.dayofweek >= 5).astype(float)

    ok = f.notna().all(axis=1) & err.notna()
    f, err_ok = f[ok], err[ok]
    hour, month = f.index.hour, f.index.month

    # outcome: top-q error within its own hour x month cell.
    # If fit_mask is given, the THRESHOLD is estimated on those rows only and
    # then applied to everything, so a later period never informs its own label.
    # All quantile thresholds -- for the outcome AND for every predicate -- are
    # estimated on rows up to fit_until_year only, then applied everywhere.  Without
    # this the later period helps define its own labels, which is look-ahead.
    fm = (np.ones(len(f), bool) if fit_until_year is None
          else np.asarray(f.index.year <= fit_until_year))
    y = np.zeros(len(f), dtype=bool)
    cellv = np.asarray(hour) * 100 + np.asarray(month)   # integer cell code
    ev = err_ok.values
    for key in np.unique(cellv):
        sel = cellv == key
        ref = ev[sel & fm]
        if ref.size < 10:
            ref = ev[sel]
        y[sel & (ev >= np.quantile(ref, q))] = True

    # binarise each feature at its own within-(hour,month) quantiles
    preds, names = [], []
    for c in f.columns:
        v = f[c].values
        if np.unique(v).size <= 2:                       # already binary
            preds.append(v > 0.5); names.append(f"{c}=1")
            continue
        for qq in (0.25, 0.50, 0.75, 0.90):
            thr = np.full(len(v), np.nan)
            for key in np.unique(cellv):
                pos = np.flatnonzero(cellv == key)
                ref = v[pos][fm[pos]]
                if ref.size < 10:
                    ref = v[pos]
                thr[pos] = np.quantile(ref, qq)   # threshold from fit rows only
            preds.append(v > thr); names.append(f"{c}>q{int(qq*100)}")

    X = pd.DataFrame(np.array(preds).T, columns=names, index=f.index)
    meta = dict(ba=ba, n=len(f), n1=int(y.sum()), base=float(y.mean()),
                start=str(f.index[0]), end=str(f.index[-1]),
                months=month, hours=hour, err=err_ok.values)
    return X, y, meta


def _base(name: str) -> str:
    """'err_lag24>q75' -> 'err_lag24'.  Threshold predicates on one feature are
    nested, so conjoining two of them is a subsumption, not a new rule."""
    return name.split(">")[0].split("=")[0]


def lattice(X: pd.DataFrame, max_arity: int = 3, min_support: int = 1,
            verbose: bool = True, closed: bool = True):
    """
    Conjunctions up to max_arity, with three things suppressed:

    1. APRIORI pruning -- support is anti-monotone, so a candidate already
       below min_support can never yield a viable superset.
    2. SAME-FEATURE conjunctions -- 'err>q25 AND err>q75' is exactly the set
       'err>q75', because quantile thresholds on one feature nest.  Allowing
       them fills the output with logical duplicates.
    3. NON-CLOSED rules -- if a longer rule covers exactly the same rows as a
       shorter one, only the shorter (more general) one is kept.  Standard
       closed-pattern reduction; the discarded rules carry no extra evidence.
    """
    from core import popcount

    cols = list(X.columns)
    bases = np.array([_base(c) for c in cols])
    base = np.stack([pack(X[c].values.astype(bool)) for c in cols])
    sup1 = popcount(base)

    keep = np.flatnonzero(sup1 >= min_support)
    rules = [base[i] for i in keep]
    labels = [(cols[i],) for i in keep]
    level = [((i,), base[i]) for i in keep]
    if verbose:
        print(f"  arity 1: {len(level):,} of {len(cols)}")

    for k in range(2, max_arity + 1):
        nxt = []
        for combo, m in level:
            used = {bases[i] for i in combo}
            for j in keep:
                if j <= combo[-1] or bases[j] in used:      # (2)
                    continue
                cand = m & base[j]
                if int(popcount(cand)) >= min_support:      # (1)
                    nxt.append((combo + (j,), cand))
        if not nxt:
            break
        rules.extend(m for _, m in nxt)
        labels.extend(tuple(cols[i] for i in c) for c, _ in nxt)
        level = nxt
        if verbose:
            print(f"  arity {k}: {len(nxt):,}")

    rules = np.stack(rules)

    if closed:                                              # (3)
        seen, keep_idx = {}, []
        for i in range(rules.shape[0]):
            key = rules[i].tobytes()
            prev = seen.get(key)
            if prev is None:
                seen[key] = i
                keep_idx.append(i)
            elif len(labels[i]) < len(labels[prev]):
                keep_idx[keep_idx.index(prev)] = i
                seen[key] = i
        keep_idx.sort()
        dropped = rules.shape[0] - len(keep_idx)
        rules = rules[keep_idx]
        labels = [labels[i] for i in keep_idx]
        if verbose:
            print(f"  closed reduction: dropped {dropped:,} duplicate-coverage rules")

    return rules, labels


def greedy_cover(labels, sup, a, sig_idx, y, top=10):
    """
    Report a non-redundant set.

    38% of rules being 'significant' is arithmetically right and practically
    useless, because they are variants of one effect.  Pick the strongest rule,
    remove the positive rows it already explains, and repeat, so each reported
    rule earns its place by covering something new.
    """
    from core import popcount
    remaining = y.copy()
    chosen = []
    pool = list(sig_idx)
    for _ in range(top):
        best, best_gain = None, 0
        for i in pool:
            gain = int((MASKS[i] & remaining).sum())
            if gain > best_gain:
                best, best_gain = i, gain
        if best is None or best_gain == 0:
            break
        chosen.append((best, best_gain))
        remaining = remaining & ~MASKS[best]
        pool.remove(best)
    return chosen
