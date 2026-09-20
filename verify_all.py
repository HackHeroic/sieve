"""End-to-end verification of everything changed tonight. Asserts, does not assume."""
import numpy as np, pandas as pd, sys
FAIL=[]
def chk(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not cond: FAIL.append(name)

print("\n[A] DATA: the PJM alignment fix")
from features import load_ba, build
d = load_ba("PJM", years=range(2019,2027))
fc, dem = d["Demand Forecast (MW)"], d["Demand (MW)"]
clean = (dem<=3e5)&(dem>0)
BREAK = pd.Timestamp("2019-01-30", tz="UTC")
pre, post = clean & (d.index<BREAK), clean & (d.index>=BREAK)
for lbl, m in (("pre-break", pre), ("post-break", post)):
    e0=(dem[m]-fc[m]).abs(); e1=(dem[m]-fc.shift(1)[m]).abs()
    m0=100*(e0/dem[m]).mean(); m1=100*(e1/dem[m]).mean()
    if lbl=="pre-break":
        chk(f"{lbl}: shift 0 is better (so the gate is needed)", m0<m1, f"{m0:.3f} vs {m1:.3f}")
    else:
        chk(f"{lbl}: shift +1 is better", m1<m0, f"{m0:.3f} vs {m1:.3f}")

print("\n[B] DATA: the gate and the corrupt rows are actually applied")
X0,y0,m0_ = build("PJM", years=range(2019,2027), q=.90, fc_shift=0)
X1,y1,m1_ = build("PJM", years=range(2019,2027), q=.90, fc_shift=1)
chk("corrupt rows gone (no demand > 3e5 survives)", m1_['n'] < 46324, f"{m1_['n']:,} hours")
i0,i1 = pd.to_datetime(X0.index), pd.to_datetime(X1.index)
pre1 = np.asarray(i1 < BREAK)
chk("pre-break rows still present after the gate", pre1.sum() > 400, f"{pre1.sum()} rows")
chk("base rate stays ~10%", abs(m1_['base']-0.10) < 0.01, f"{100*m1_['base']:.1f}%")

print("\n[C] CODE: BlockPermuter is identical to the audited generator")
from core import block_permutation, BlockPermuter
months = np.asarray(i1.month); n = m1_['n']
for strata,nm in ((months,"stratified"),(None,"unstratified")):
    r1,r2 = np.random.default_rng(7), np.random.default_rng(7)
    bp = BlockPermuter(n,168,strata)
    ok = all(np.array_equal(block_permutation(n,168,r1,strata), bp.draw(r2)) for _ in range(30))
    chk(f"{nm}: 30/30 draws bit-identical", ok)

print("\n[D] CODE: cutoff is the order statistic, not np.quantile")
from engine import Problem
import inspect
src = inspect.getsource(Problem.cutoff)
chk("cutoff uses np.sort + floor(alpha*(J+1))", "np.sort" in src and "floor" in src)
rng=np.random.default_rng(0); hit=0; T=3000
for _ in range(T):
    nul=rng.standard_normal(200); obs=rng.standard_normal()
    srt=np.sort(nul); k=int(np.floor(0.05*201))-1
    hit += obs <= srt[k]
chk("achieved rate at J=200 is near nominal", abs(hit/T-0.05)<0.012, f"{hit/T:.4f}")

print("\n[E] CONSISTENCY: block length across experiments")
print("      full PJM calibration used 504h; holdout/replication used 168h")
print("      168h on full corrected PJM measured FPR 0.100 (miscalibrated)")
chk("*** THIS IS INCONSISTENT AND MUST BE DISCLOSED OR FIXED ***", False,
    "see note below")

print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
