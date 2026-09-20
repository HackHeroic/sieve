"""Is the finding just weather?

EIA-930 has no weather, so we join NOAA GHCN-Daily and ask the only question
that matters: does the rule still work INSIDE weather strata?

Using OBSERVED weather (not a forecast) is deliberate.  It is the strongest
possible control: if the effect survives conditioning on what the weather
actually did, it cannot be weather doing the work.
"""
import numpy as np, pandas as pd

STATION = {'PJM':'USW00013739','CISO':'USW00023174','ERCO':'USW00003927',
           'MISO':'USW00014922','NYIS':'USW00094728'}

def load_wx(code):
    d = pd.read_csv(f'wx/{code}.csv', low_memory=False,
                    usecols=lambda c: c in ('DATE','PRCP','TMAX','TMIN','AWND'))
    d['DATE'] = pd.to_datetime(d['DATE'], errors='coerce')
    d = d.dropna(subset=['DATE']).set_index('DATE')
    for c in ('PRCP','TMAX','TMIN','AWND'):
        if c in d: d[c] = pd.to_numeric(d[c], errors='coerce')
        else: d[c] = np.nan
    d['TRANGE'] = d['TMAX'] - d['TMIN']
    return d[['TMAX','TMIN','PRCP','AWND','TRANGE']]


def strata(idx, ba, nbins=5):
    """Weather stratum label per hourly row: temp quintile x precip x windy."""
    wx = load_wx(STATION[ba])
    day = pd.DatetimeIndex(idx).tz_localize(None).normalize()
    w = wx.reindex(day)
    tq = pd.qcut(w['TMAX'].values, nbins, labels=False, duplicates='drop')
    rq = (np.nan_to_num(w['PRCP'].values) > 0).astype(int)
    aq = w['AWND'].values
    wq = (aq > np.nanmedian(aq)).astype(int) if np.isfinite(aq).any() else np.zeros(len(w), int)
    s = pd.Series(tq).fillna(-1).astype(int).values * 4 + rq * 2 + wq
    return s, w


def stratified_lift(mask, y, s):
    """
    Marginal lift vs lift pooled WITHIN weather strata.

    Pooled = sum of observed positives inside the rule, over sum of positives
    EXPECTED inside the rule given each stratum's own base rate.  If weather
    explains the rule, expected catches up with observed and pooled -> 1.
    """
    marg = (y[mask].mean() / y.mean()) if mask.sum() else np.nan
    obs = exp = 0.0
    for u in np.unique(s):
        k = (s == u)
        if k.sum() < 50: continue
        base_u = y[k].mean()
        m = mask & k
        obs += y[m].sum()
        exp += m.sum() * base_u
    return marg, (obs / exp if exp > 0 else np.nan), obs, exp
