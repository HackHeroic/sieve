"""SIEVE figure set.  Palette validated via the dataviz validator (light mode)."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, MUTED, SURF, GRID = '#0b0b0b', '#52514e', '#8a8983', '#fcfcfb', '#e6e5e0'
plt.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'text.color': INK,
    'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'grid.color': GRID, 'grid.linewidth': 0.8,
})
def style(ax, xgrid=False):
    ax.grid(axis='x' if xgrid else 'y', alpha=.9, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values(): s.set_linewidth(0.8)

# ---------------------------------------------------------------- 1 funnel
fig, ax = plt.subplots(figsize=(7.6, 3.1))
stages = ['Rules the machine\nwrote and tested', 'Survive correction\n(genuinely real)',
          'Fail correction\n(look good, are luck)']
vals = [66844, 26715, 40129]
cols = [MUTED, BLUE, ORANGE]
b = ax.barh(range(3), vals, color=cols, height=.58, zorder=3)
for i, v in enumerate(vals):
    ax.text(v + 900, i, f'{v:,}', va='center', ha='left', fontweight='bold', color=INK, fontsize=11)
ax.set_yticks(range(3)); ax.set_yticklabels(stages, fontsize=9.5)
ax.invert_yaxis(); ax.set_xlim(0, 79000); ax.set_xticks([])
ax.set_title('Of 66,844 automatically generated rules, 60% are luck',
             loc='left', fontweight='bold', fontsize=12, pad=12)
for s in ('bottom','left'): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig('fig1_funnel.png', dpi=170); plt.close()

# ---------------------------------------------------------------- 2 the null
d = np.load('results2.npz'); fake = d['nb']; cut = float(np.quantile(fake, .05))
fig, ax = plt.subplots(figsize=(7.6, 3.5))
ax.hist(fake, bins=48, color=MUTED, zorder=3)
ax.axvline(cut, color=ORANGE, lw=2.2, zorder=5)
ax.set_xlim(-95, 0)
ymax = ax.get_ylim()[1]
ax.text(cut - 1.5, ymax*.93, 'the bar', color=ORANGE, fontweight='bold',
        fontsize=10.5, ha='right', va='top')
ax.text(cut - 1.5, ymax*.80, f'only 5% of fake worlds\nfind anything better than {cut:.0f}',
        color=ORANGE, fontsize=9, ha='right', va='top')
# the real score is off the chart -- say so, do not fake the axis
ax.annotate('', xy=(-94, ymax*.30), xytext=(-72, ymax*.30),
            arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=2.4))
ax.text(-71, ymax*.30, '  real data scores  −1045\n  (11× past the left edge)',
        color=BLUE, fontweight='bold', fontsize=10.5, va='center', ha='left')
ax.set_xlabel('score of the single best rule out of 66,844   (log p, lower = stronger)')
ax.set_ylabel('number of fake worlds')
ax.set_title('How we know: 1,000 worlds where we destroyed the truth first',
             loc='left', fontweight='bold', fontsize=12, pad=12)
style(ax); plt.tight_layout(); plt.savefig('fig2_null.png', dpi=170); plt.close()

# ---------------------------------------------------------------- 3 replication
grids = ['CISO\nCalifornia','MISO\nMidwest','NYIS\nNew York','ERCO\nTexas']
rep = [15/15, 15/15, 13/13, 14/15]
fig, ax = plt.subplots(figsize=(7.6, 3.1))
ax.bar(range(4), [r*100 for r in rep], color=AQUA, width=.55, zorder=3)
for i,(g,r) in enumerate(zip(grids, rep)):
    n = [15,15,13,15][i]; k = int(round(r*n))
    ax.text(i, r*100 + 2.5, f'{k}/{n}', ha='center', fontweight='bold', color=INK, fontsize=11)
ax.set_xticks(range(4)); ax.set_xticklabels(grids, fontsize=9.5)
ax.set_ylim(0, 112); ax.set_ylabel('% of PJM rules that replicate')
ax.set_title('Rules found on the mid-Atlantic grid, tested on four others: 57 of 58 hold',
             loc='left', fontweight='bold', fontsize=12, pad=12)
style(ax); plt.tight_layout(); plt.savefig('fig3_replication.png', dpi=170); plt.close()

# ---------------------------------------------------------------- 4 ladder
names = ['Naive\n(plain Python)','Faster maths\n(our algorithm)','14 CPU cores','GPU']
ms = [1510.6, 83.3, 58.4, 0.48]
fig, ax = plt.subplots(figsize=(7.6, 3.3))
cols4 = [MUTED, YELLOW, ORANGE, BLUE]
ax.bar(range(4), ms, color=cols4, width=.55, zorder=3)
ax.set_yscale('log')
for i,v in enumerate(ms):
    lbl = f'{v:,.1f} ms' if v >= 1 else f'{v:.2f} ms'
    ax.text(i, v*1.35, lbl, ha='center', fontweight='bold', color=INK, fontsize=10.5)
ax.annotate('', xy=(3, 0.9), xytext=(0, 900),
            arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=2,
                            connectionstyle='arc3,rad=-0.22'))
ax.text(1.5, 260, '3,134× faster end to end', color=BLUE, fontweight='bold',
        fontsize=11.5, ha='center')
ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=9.5)
ax.set_ylabel('time per test (ms, log scale)'); ax.set_ylim(0.03, 9000)
ax.set_title('Each step measured on the same machine, same statistic',
             loc='left', fontweight='bold', fontsize=12, pad=12)
style(ax); plt.tight_layout(); plt.savefig('fig4_ladder.png', dpi=170); plt.close()

# ---------------------------------------------------------------- 5 weather
bas = ['PJM','CISO','ERCO','MISO','NYIS']
marg = [4.51, 5.56, 4.36, 4.79, 4.14]
wx   = [3.78, 5.44, 3.61, 4.63, 3.27]
x = np.arange(5); w = .36
fig, ax = plt.subplots(figsize=(7.6, 3.2))
ax.bar(x-w/2, marg, w, color=BLUE, zorder=3, label='effect as measured')
ax.bar(x+w/2, wx,  w, color=AQUA, zorder=3, label='effect within identical weather')
for i,(a_,b_) in enumerate(zip(marg,wx)):
    ax.text(i-w/2, a_+.1, f'{a_:.1f}', ha='center', fontsize=9.5, color=INK)
    ax.text(i+w/2, b_+.1, f'{b_:.1f}', ha='center', fontsize=9.5, color=INK, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(bas); ax.set_ylim(0, 6.6)
ax.set_ylabel('times more likely to fail')
ax.legend(frameon=False, fontsize=9.5, loc='upper right')
ax.set_title('Is it just weather? No — 79–98% of the effect survives',
             loc='left', fontweight='bold', fontsize=12, pad=12)
style(ax); plt.tight_layout(); plt.savefig('fig5_weather.png', dpi=170); plt.close()

# ---------------------------------------------------------------- 6 two nulls
fig, ax = plt.subplots(figsize=(7.6, 3.1))
ax.bar([0,1], [5.0, 100.0], color=[AQUA, ORANGE], width=.5, zorder=3)
for i,v in enumerate([5.0,100.0]):
    ax.text(i, v+3, f'{v:.0f}%', ha='center', fontweight='bold', fontsize=13, color=INK)
ax.set_xticks([0,1])
ax.set_xticklabels(['Our method\n(week-long block shuffle)','The standard shortcut\n(shuffle every hour)'], fontsize=10)
ax.set_ylim(0, 118); ax.set_ylabel('% of empty datasets where it\nclaims a discovery')
ax.axhline(5, color=INK2, ls=':', lw=1.4, zorder=4)
ax.text(1.42, 7, 'correct answer: 5%', color=INK2, fontsize=9, ha='right')
ax.set_title('Fed 120 datasets containing nothing at all',
             loc='left', fontweight='bold', fontsize=12, pad=12)
style(ax); plt.tight_layout(); plt.savefig('fig6_twonulls.png', dpi=170); plt.close()
print("wrote 6 figures")
