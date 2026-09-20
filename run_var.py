"""SIEVE on market risk data: when does a naive volatility forecast fail?"""
import io, time, numpy as np, pandas as pd, urllib.request, engine
from adapter import build_generic
from features import lattice
from engine import Problem

def fred(sid):
    u=f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
    d=pd.read_csv(io.StringIO(urllib.request.urlopen(u,timeout=30).read().decode()))
    d.columns=["date",sid]; d["date"]=pd.to_datetime(d["date"])
    d[sid]=pd.to_numeric(d[sid],errors="coerce"); return d.set_index("date")[sid]

t0=time.time()
df = pd.DataFrame({k: fred(k) for k in
    ("VIXCLS","DGS10","DGS3MO","BAMLH0A0HYM2","DTWEXBGS","DCOILWTICO")})
df = df[df.index >= "1997-01-01"].ffill(limit=5)
df["term_spread"] = df.DGS10 - df.DGS3MO
df = df.dropna(subset=["VIXCLS"]).ffill(limit=10).dropna()
print(f"{len(df):,} trading days, {df.index[0].date()} to {df.index[-1].date()}")
print(f"  series: {list(df.columns)}  ({time.time()-t0:.0f}s)")

# outcome: the naive "tomorrow's VIX = today's VIX" forecast fails badly
X, y, meta = build_generic(df, target="VIXCLS", horizon=1, q=.90,
                           lags=(1,2,5), stratify="month")
print(f"\n  outcome: top decile of |VIX(t) - VIX(t-1)| within calendar month")
print(f"  {meta['n']:,} days, {meta['n1']:,} positive ({100*meta['base']:.1f}%), "
      f"{X.shape[1]} predicates")
yy = y.astype(float)-y.mean()
acf1 = float(np.correlate(yy[:-1],yy[1:])[0]/np.dot(yy,yy))
print(f"  outcome autocorrelation lag 1: {acf1:.3f}")

r, lab = lattice(X, max_arity=3, min_support=1, verbose=False)
print(f"  {r.shape[0]:,} rules  ({time.time()-t0:.0f}s)")

for BLOCK in (21, 63, 126, 252):
    engine.BLOCK = BLOCK
    p = Problem(r, lab, y, months=meta["months"], alpha=0.05)
    nb = p.null(300, kind="block", seed=1); nr = p.null(300, kind="row", seed=1)
    sb, sr = len(p.significant(nb)), len(p.significant(nr))
    print(f"  block {BLOCK:4d}d ({meta['n']//BLOCK:3d} blocks): "
          f"significant {sb:6,}   naive {sr:6,}   inflation {sr/max(sb,1):5.2f}x", flush=True)

engine.BLOCK = 63
p = Problem(r, lab, y, months=meta["months"], alpha=0.05)
nb = p.null(500, kind="block", seed=1)
print(f"\n  FINDINGS (63-day blocks)")
ch, cum = p.cover(nb, top=5, min_support=150, min_new=15)
for c in ch:
    print(f"    lift {c['lift']:4.2f}  n={c['support']:5,}  {c['rule']}")
print(f"\n  total {time.time()-t0:.0f}s")
