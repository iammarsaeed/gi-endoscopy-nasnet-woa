"""Extra statistics used in the paper that report.py does not produce (McNemar tests, paired t-tests, fitness comparisons).
Usage: python -m analysis.twostage_stats <data.npz> <out.json>   (run after twostage_pipeline.py, search_budget.py and report.py)"""
import json, sys
import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

y = np.load(sys.argv[1])['y']; N = len(y)


def tasks(d, cfg, kern='CB-SVM'):
    return [json.load(open(f'{d}/tasks/{cfg}__{kern}__{k}.json')) for k in range(5)]


def pooled(rs, key='pred'):
    p = np.full(N, -1)
    for r in rs: p[np.array(r['te'])] = r[key]
    return p


def r3svm(tag):
    return [json.load(open(f'results_budget/svm/{tag}_{k}.json')) for k in range(5)]


def mc(ra, rb):
    ca, cb = pooled(ra) == y, pooled(rb) == y
    t = [[int((ca & cb).sum()), int((ca & ~cb).sum())], [int((~ca & cb).sum()), int((~ca & ~cb).sum())]]
    m = mcnemar(t, exact=False, correction=True)
    aa, ab = np.array([r['acc'] for r in ra]), np.array([r['acc'] for r in rb])
    return dict(diff_pp=round(100 * (aa.mean() - ab.mean()), 2), b=t[0][1], c=t[1][0], chi2=round(float(m.statistic), 2),
                p=round(float(m.pvalue), 4), t_p=round(float(stats.ttest_rel(aa, ab).pvalue), 4))


def msd(v, s=100, d=1):
    v = np.asarray(v, float) * s
    return f'{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}'


new = tasks('results_twostage', 'fused+woa')
out = {}
out['proposed'] = dict(acc=msd([r['acc'] for r in new]), f1=msd([r['f1'] for r in new]),
                       acc_single=msd([r['acc_single'] for r in new]), train=msd([r['train_s'] for r in new], 1, 1),
                       pred=msd([r['pred_speed'] for r in new], 1, 0), n=msd([r['n_features'] for r in new], 1, 0),
                       n_folds=[r['n_features'] for r in new], acc_folds=[round(100 * r['acc'], 2) for r in new])
out['vs_fused'] = mc(new, tasks('results_twostage', 'fused'))
out['vs_relieff_match'] = mc(new, tasks('results_twostage', 'fused+relieff'))
out['vs_relieff200'] = mc(new, r3svm('relieff_K200'))
for c in ['fused+bpso', 'fused+bgwo', 'fused+ga', 'fused+pca', 'fused+rand']:
    out[f'vs_{c}'] = mc(new, tasks('results_twostage', c))
# ReliefF top-200 alone
rk = r3svm('relieff_K200')
out['vs_random_ft'] = mc(new, tasks('results_twostage', 'fused+rand'))
out['relieff200'] = dict(acc=msd([r['acc'] for r in rk]), f1=msd([r['f1'] for r in rk]), train=msd([r['train_s'] for r in rk], 1, 1),
                         pred=msd([r['pred_speed'] for r in rk], 1, 0))
rank_s = [json.load(open(f'results_twostage/sel/fused+bwoa_{k}.json'))['rank_s'] for k in range(5)]
out['relieff200']['sel_s'] = msd(rank_s, 1, 1)
# selection stats for the proposed WOA
sel = [json.load(open(f'results_twostage/sel/fused+bwoa_{k}.json')) for k in range(5)]
out['woa_sel'] = dict(seconds=msd([s['seconds'] for s in sel], 1, 1), wrap_s=msd([s['wrap_s'] for s in sel], 1, 1),
                      rank_s=msd([s['rank_s'] for s in sel], 1, 1), evals=[s['evals'] for s in sel],
                      fitness=msd([s['best_fitness'] for s in sel], 1, 3), n=[s['n'] for s in sel])
# fitness comparison WOA vs same-size ReliefF and initial population (from search_budget.py)
b = [json.load(open(f'results_budget/sel/bwoa_K200_50x100_{k}.json')) for k in range(5)]
out['fitness_cmp'] = dict(init=msd([s['init_best'] for s in b], 1, 3), woa=msd([s['best_fitness'] for s in b], 1, 3),
                          relieff_same_n=msd([s['fitness_relieff_match'] for s in b], 1, 3),
                          all200=msd([s['fitness_all_K'] for s in b], 1, 3),
                          last_improve=[int(np.flatnonzero(np.diff(s['curve']) < 0)[-1] + 2) if np.any(np.diff(s['curve']) < 0) else 1 for s in b])
# search-budget table (reduced space) from search_budget.py
S3 = json.load(open('results_budget/summary.json'))
out['budget'] = {k: dict(evals=S3[k]['evals'][0], n=f"{S3[k]['n'][0]:.0f} ± {S3[k]['n'][1]:.0f}",
                         fitness=f"{S3[k]['best_fitness'][0]:.3f} ± {S3[k]['best_fitness'][1]:.3f}",
                         acc=f"{S3[k]['acc'][0]:.1f} ± {S3[k]['acc'][1]:.1f}",
                         sel=f"{S3[k]['seconds'][0]:.0f} ± {S3[k]['seconds'][1]:.0f}")
                 for k in S3 if k.startswith(('bwoa_', 'rand_'))}
for k in ['bwoa_K200_30x100', 'bwoa_K200_50x200', 'bwoa_K200_100x200', 'bwoa_K100_50x100', 'bwoa_K2112_10x20', 'bwoa_K2112_50x100', 'rand_K200_50x100']:
    out['budget'][k]['vs_proposed'] = mc(new, r3svm(k))
# frozen ablation row
fz = [json.load(open(f'results_twostage/frozen_fused+woa__CB-SVM__{k}.json')) for k in range(5)]
out['frozen'] = dict(acc=msd([r['acc'] for r in fz]), f1=msd([r['f1'] for r in fz]), train=msd([r['train_s'] for r in fz], 1, 1),
                     pred=msd([r['pred_speed'] for r in fz], 1, 0), n=msd([r['n_features'] for r in fz], 1, 0))
# baselines
B = json.load(open('results_ft/baselines.json')); a = np.array([r['acc'] for r in new])
for m in ['resnet50', 'efficientnet_b0', 'swin_tiny_patch4_window7_224']:
    bb = np.array([r['acc'] for r in sorted([r for r in B if r['model'] == m], key=lambda r: r['fold'])])
    out[f'vs_{m}'] = dict(t_p=round(float(stats.ttest_rel(a, bb).pvalue), 4), diff_pp=round(100 * (a.mean() - bb.mean()), 2))
half = stats.t.ppf(.975, 4) * a.std(ddof=1) / np.sqrt(5)
out['ci95'] = [round(100 * (a.mean() - half), 1), round(100 * (a.mean() + half), 1)]
json.dump(out, open(sys.argv[2], 'w'), indent=1)
print(json.dumps(out, indent=1))
