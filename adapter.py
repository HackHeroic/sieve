"""
Generic CSV -> SIEVE.  Point it at any time-indexed table.

The framing that makes this general
-----------------------------------
Every SIEVE question has the same shape: *when does a naive forecast fail?*
For EIA-930 the naive forecast is the operator's own filed day-ahead forecast.
For an arbitrary series it is PERSISTENCE: the value h steps ago.  So

    outcome(t) = 1  if  |target(t) - target(t-h)|  is in the top q of its own
                        stratum cell

and every predicate is a lagged quantity, knowable at t-h.  That is one
substantive assumption -- persistence is the baseline -- and it is stated rather
than hidden.

The look-ahead rule from features.py is preserved exactly: thresholds for the
outcome AND for every predicate are fitted on `fit_until` rows only, then
applied everywhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core import pack


def load_csv(path, time_col, cols=None, tz=None):
    df = pd.read_csv(path, low_memory=False)
    if time_col not in df.columns:
        raise SystemExit(f"--time-col {time_col!r} not in {list(df.columns)[:12]}")
    t = pd.to_datetime(df[time_col], errors="coerce", format="mixed", utc=bool(tz))
    df = df.assign(_t=t).dropna(subset=["_t"]).sort_values("_t").drop_duplicates("_t")
    df = df.set_index("_t")
    num = df.select_dtypes("number")
    if cols:
        missing = [c for c in cols if c not in num.columns]
        if missing:
            raise SystemExit(f"not numeric or not present: {missing}")
        num = num[cols]
    return num.loc[:, num.notna().mean() > 0.5]        # drop mostly-empty columns


def build_generic(df, target, horizon=1, q=0.90, lags=(1, 2, 7),
                  fit_until=None, stratify="month"):
    """
    Returns (X binary predicates, y bool, meta).

    `stratify` picks the cell the top-q outcome is taken within.  "month" is the
    sane default for daily data; use "hour_month" for sub-daily.
    """
    if target not in df.columns:
        raise SystemExit(f"--target {target!r} not in {list(df.columns)[:12]}")

    s = df[target].astype(float)
    err = (s - s.shift(horizon)).abs()                 # persistence-forecast error

    f = pd.DataFrame(index=df.index)
    for c in df.columns:
        v = df[c].astype(float)
        for L in lags:
            f[f"{c}_lag{L}"] = v.shift(L)              # every feature lagged >= 1
        f[f"{c}_chg_{lags[0]}"] = v.shift(lags[0]) - v.shift(lags[0] + 1)
        if len(lags) > 2:
            f[f"{c}_roll{lags[-1]}"] = v.shift(lags[0]).rolling(lags[-1]).mean()
    # the persistence error itself, lagged, is the strongest known predictor
    for L in lags:
        f[f"perr_lag{L}"] = err.shift(L)
    f[f"perr_roll{lags[-1]}"] = err.shift(lags[0]).rolling(lags[-1]).mean()

    # calendar
    f["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    f["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    if stratify == "hour_month":
        f["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        f["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    f["is_weekend"] = (df.index.dayofweek >= 5).astype(float)

    ok = f.notna().all(axis=1) & err.notna()
    f, err_ok = f[ok], err[ok]
    idx = f.index

    cell = (np.asarray(idx.month) if stratify == "month"
            else np.asarray(idx.hour) * 100 + np.asarray(idx.month))

    fm = (np.ones(len(f), bool) if fit_until is None
          else np.asarray(idx.year <= fit_until))
    if fm.sum() < 100:
        raise SystemExit(f"--fit-until {fit_until} leaves only {fm.sum()} fit rows")

    ev = err_ok.values
    y = np.zeros(len(f), dtype=bool)
    for k in np.unique(cell):
        sel = cell == k
        ref = ev[sel & fm]
        if ref.size < 10:
            ref = ev[sel]
        y[sel & (ev >= np.quantile(ref, q))] = True

    preds, names = [], []
    for c in f.columns:
        v = f[c].values
        if np.unique(v).size <= 2:
            preds.append(v > 0.5); names.append(f"{c}=1"); continue
        for qq in (0.25, 0.50, 0.75, 0.90):
            thr = np.full(len(v), np.nan)
            for k in np.unique(cell):
                pos = np.flatnonzero(cell == k)
                ref = v[pos][fm[pos]]
                if ref.size < 10:
                    ref = v[pos]
                thr[pos] = np.quantile(ref, qq)
            preds.append(v > thr); names.append(f"{c}>q{int(qq*100)}")

    X = pd.DataFrame(np.array(preds).T, columns=names, index=idx)
    meta = dict(n=len(f), n1=int(y.sum()), base=float(y.mean()), target=target,
                horizon=horizon, start=str(idx[0]), end=str(idx[-1]),
                months=np.asarray(idx.month), cell=cell)
    return X, y, meta
