"""Figure 12: convergence of the wrapper searches on the 200 ReliefF-ranked fused features.
Mean over the five folds of the best fitness (Eq. 16, D = 200) found after each fitness evaluation.
Usage: python -m figures.fig_convergence <search_budget_dir> <twostage_pipeline_dir> <out.png>"""
import glob, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R3, RFT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'


def curve(pattern):
    cs = [np.array(json.load(open(f))['curve']) for f in sorted(glob.glob(pattern))]
    assert len(cs) == 5, pattern
    n = min(len(c) for c in cs)
    return np.arange(1, n + 1), np.mean([c[:n] for c in cs], axis=0)


plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': 0.8})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True, dpi=300)
panels = [
    ('(a) WOA, agents × iterations',
     [(f'{R3}/sel/bwoa_K200_30x100_*.json', '30 × 100', '#86b6ef', '-'),
      (f'{R3}/sel/bwoa_K200_50x100_*.json', '50 × 100 (proposed)', '#2a78d6', '-'),
      (f'{R3}/sel/bwoa_K200_50x200_*.json', '50 × 200', '#1c5cab', '--'),
      (f'{R3}/sel/bwoa_K200_100x200_*.json', '100 × 200', '#0d366b', '-.')], 25000),
    ('(b) Wrapper methods, 5,050 evaluations',
     [(f'{RFT}/sel/fused+bwoa_*.json', 'WOA', '#2a78d6', '-'), (f'{RFT}/sel/fused+bpso_*.json', 'BPSO', '#eb6834', '--'),
      (f'{RFT}/sel/fused+bgwo_*.json', 'BGWO', '#1baf7a', '-.'), (f'{RFT}/sel/fused+ga_*.json', 'GA', '#eda100', ':'),
      (f'{RFT}/sel/fused+rand_*.json', 'Random search', '#898781', (0, (1, 1)))], 6000),
]
for ax, (title, series, xmax) in zip(axes, panels):
    for pat, label, color, ls in series:
        x, y = curve(pat)
        ax.plot(x, y, color=color, lw=2, ls=ls, label=label, solid_capstyle='round')
    ax.set_xscale('log'); ax.set_xlim(10, xmax)
    ax.set_title(title, fontsize=9, color=INK, loc='left')
    ax.set_xlabel('Fitness evaluations (log scale)')
    ax.grid(True, which='major', color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc='lower left', handlelength=2.4)
axes[0].set_ylabel('Best fitness (mean of 5 folds)')
fig.tight_layout()
fig.savefig(OUT, facecolor='white')
print('saved', OUT)
