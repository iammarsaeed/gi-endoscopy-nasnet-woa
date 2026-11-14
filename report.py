"""Step 5: aggregate results -> summary.json, summary.md and figures.
Usage: python report.py <data.npz> <results_dir> <fig_dir>
"""
import json, os, sys
from collections import defaultdict
import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES = ['Dyed-lifted-polyps', 'Dyed-resection-margins', 'Esophagitis', 'Normal-cecum', 'Normal-pylorus',
         'Normal-z-line', 'Polyps', 'Ulcerative-colitis']
SHORT = ['DLP', 'DRM', 'ESO', 'NC', 'NP', 'NZL', 'P', 'UC']
KERNELS = ['L-SVM', 'Q-SVM', 'CB-SVM', 'MG-SVM', 'CG-SVM']


def ms(v, scale=100, d=1):
    v = np.asarray(v, float) * scale
    return f'{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}'


def pooled(rows, n):
    pred = np.full(n, -1)
    for r in rows: pred[np.array(r['te'])] = r['pred']
    return pred


def cm_fig(cm, path):
    rown = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.2, 6.6), dpi=200)
    ax.imshow(rown, cmap='Blues', vmin=0, vmax=1)
    for i in range(8):
        for j in range(8):
            if cm[i, j]: ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=9,
                                 color='white' if rown[i, j] > .5 else 'black', fontweight='bold' if i == j else 'normal')
    ax.set_xticks(range(8)); ax.set_yticks(range(8)); ax.set_xticklabels(SHORT)
    ax.set_yticklabels([f'{n}\n({s})' for n, s in zip(NAMES, SHORT)], fontsize=8.5)
    ax.set_xlabel('Predicted class'); ax.set_ylabel('True class')
    ax.set_xticks(np.arange(-.5, 8, 1), minor=True); ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
    ax.grid(which='minor', color='#999', linewidth=.5); ax.tick_params(which='minor', length=0)
    fig.tight_layout(); fig.savefig(path, bbox_inches='tight', pad_inches=0.08); plt.close(fig)


def bar_fig(vals, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    m = [np.mean(v) for v in vals]; s = [np.std(v, ddof=1) for v in vals]
    bars = ax.bar(KERNELS, m, yerr=s, capsize=3, color='#2E6DB4', width=.6)
    for b, v, e in zip(bars, m, s): ax.text(b.get_x() + b.get_width() / 2, v + e, f'{v:.1f}' if max(m) < 20 else f'{v:.0f}',
                                            ha='center', va='bottom', fontsize=8.5)
    ax.set_ylabel('Training time per fold (s)'); ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, color='#ddd'); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main(data_path, rdir, fdir):
    os.makedirs(fdir, exist_ok=True)
    y = np.load(data_path)['y']; n = len(y)
    R = json.load(open(f'{rdir}/svm_results.json'))
    B = json.load(open(f'{rdir}/baselines.json')) if os.path.exists(f'{rdir}/baselines.json') else []
    sel = json.load(open(f'{rdir}/selections.json'))
    g = defaultdict(list)
    for r in R: g[(r['cfg'], r['kernel'])].append(r)
    for k in g: g[k].sort(key=lambda r: r['fold'])
    S = {'experiments': {}, 'ablation': {}, 'feature_selection': {}, 'baselines': {}, 'stats': {}}
    md = []
    exp_cfg = {1: 'orig', 2: 'bbhe', 3: 'fused', 4: 'fused+woa'}
    cmfig = {1: 4, 2: 6, 3: 8, 4: 10}; tfig = {1: 5, 2: 7, 3: 9, 4: 11}
    for e, cfg in exp_cfg.items():
        md.append(f'\n## Experiment {e} ({cfg})\n| Kernel | Acc | P | R | F1 | Train s/fold | Pred obs/s | Features |')
        md.append('|---|---|---|---|---|---|---|---|')
        S['experiments'][e] = {}
        for kern in KERNELS:
            rs = g[(cfg, kern)]
            row = dict(acc=ms([r['acc'] for r in rs]), precision=ms([r['precision'] for r in rs]),
                       recall=ms([r['recall'] for r in rs]), f1=ms([r['f1'] for r in rs]),
                       train_s=ms([r['train_s'] for r in rs], 1, 1), pred_speed=ms([r['pred_speed'] for r in rs], 1, 0),
                       n_features=ms([r['n_features'] for r in rs], 1, 0), fold_acc=[round(r['acc'] * 100, 2) for r in rs])
            S['experiments'][e][kern] = row
            md.append(f"| {kern} | {row['acc']} | {row['precision']} | {row['recall']} | {row['f1']} | {row['train_s']} | {row['pred_speed']} | {row['n_features']} |")
        p = pooled(g[(cfg, 'CB-SVM')], n)
        cm = np.zeros((8, 8), int)
        for t, q in zip(y, p): cm[t, q] += 1
        S['experiments'][e]['CB_confusion'] = cm.tolist()
        cm_fig(cm, f'{fdir}/Fig{cmfig[e]}_confusion_matrix.png')
        bar_fig([[r['train_s'] for r in g[(cfg, k)]] for k in KERNELS], f'{fdir}/Fig{tfig[e]}_training_time.png')
    # per-class (proposed)
    cm = np.array(S['experiments'][4]['CB_confusion'])
    P = cm.diagonal() / cm.sum(0); Rr = cm.diagonal() / cm.sum(1); F = 2 * P * Rr / (P + Rr)
    S['per_class'] = {NAMES[i]: dict(P=round(P[i] * 100, 1), R=round(Rr[i] * 100, 1), F1=round(F[i] * 100, 1)) for i in range(8)}
    off = sorted(((cm[i, j] + cm[j, i], NAMES[i], NAMES[j]) for i in range(8) for j in range(i + 1, 8)), reverse=True)[:3]
    S['top_confusions'] = [dict(pair=f'{a}/{b}', errors=int(c)) for c, a, b in off]
    S['total_errors'] = int(cm.sum() - cm.trace())
    md.append('\n## Per-class (proposed, pooled)\n' + json.dumps(S['per_class'], indent=0))
    md.append('Top confusions: ' + json.dumps(S['top_confusions']) + f" of {S['total_errors']} errors")
    # ablation
    md.append('\n## Ablation (CB-SVM)\n| cfg | Acc | F1 | Train s | Pred/s | Features |\n|---|---|---|---|---|---|')
    for cfg in ['orig_noaug', 'orig', 'bbhe', 'fused', 'orig+woa', 'bbhe+woa', 'fused+woa']:
        rs = g[(cfg, 'CB-SVM')]
        row = dict(acc=ms([r['acc'] for r in rs]), f1=ms([r['f1'] for r in rs]), train_s=ms([r['train_s'] for r in rs], 1, 1),
                   pred_speed=ms([r['pred_speed'] for r in rs], 1, 0), n_features=ms([r['n_features'] for r in rs], 1, 0))
        S['ablation'][cfg] = row
        md.append(f"| {cfg} | {row['acc']} | {row['f1']} | {row['train_s']} | {row['pred_speed']} | {row['n_features']} |")
    # feature-selection comparison
    md.append('\n## Feature selection (CB-SVM)\n| method | features | Acc | F1 | select s | train s |\n|---|---|---|---|---|---|')
    for cfg in ['fused', 'fused+pca', 'fused+relieff', 'fused+bpso', 'fused+bgwo', 'fused+ga', 'fused+woa']:
        rs = g[(cfg, 'CB-SVM')]
        secs = [r.get('selection_s', sel.get(f"{cfg}|{r['fold']}", {}).get('seconds', 0)) for r in rs]
        row = dict(n_features=ms([r['n_features'] for r in rs], 1, 0), acc=ms([r['acc'] for r in rs]), f1=ms([r['f1'] for r in rs]),
                   selection_s=ms(secs, 1, 1), train_s=ms([r['train_s'] for r in rs], 1, 1))
        S['feature_selection'][cfg] = row
        md.append(f"| {cfg} | {row['n_features']} | {row['acc']} | {row['f1']} | {row['selection_s']} | {row['train_s']} |")
    # baselines
    gb = defaultdict(list)
    for r in B: gb[r['model']].append(r)
    if gb:
        md.append('\n## Baselines\n| model | params M | Acc | F1 | train s/fold | pred/s |\n|---|---|---|---|---|---|')
    for mname, rs in gb.items():
        rs.sort(key=lambda r: r['fold'])
        row = dict(params_m=round(rs[0]['params_m'], 1), acc=ms([r['acc'] for r in rs]), f1=ms([r['f1'] for r in rs]),
                   train_s=ms([r['train_s'] for r in rs], 1, 0), pred_speed=ms([r['pred_speed'] for r in rs], 1, 0),
                   epochs=rs[0]['epochs'], fold_acc=[round(r['acc'] * 100, 2) for r in rs])
        S['baselines'][mname] = row
        md.append(f"| {mname} | {row['params_m']} | {row['acc']} | {row['f1']} | {row['train_s']} | {row['pred_speed']} |")
    # statistics
    p3, p4 = pooled(g[('fused', 'CB-SVM')], n), pooled(g[('fused+woa', 'CB-SVM')], n)
    c3, c4 = p3 == y, p4 == y
    tab = [[int((c3 & c4).sum()), int((c3 & ~c4).sum())], [int((~c3 & c4).sum()), int((~c3 & ~c4).sum())]]
    mc = mcnemar(tab, exact=False, correction=True)
    S['stats']['mcnemar_exp3_vs_exp4'] = dict(table=tab, chi2=float(mc.statistic), p=float(mc.pvalue))
    a4 = np.array([r['acc'] for r in g[('fused+woa', 'CB-SVM')]])
    acc_mean = a4.mean(); half = stats.t.ppf(.975, 4) * a4.std(ddof=1) / np.sqrt(5)
    S['stats']['proposed_acc_ci95'] = [round((acc_mean - half) * 100, 2), round((acc_mean + half) * 100, 2)]
    S['per_fold_proposed'] = [dict(fold=r['fold'] + 1, acc=round(r['acc'] * 100, 2), f1=round(r['f1'] * 100, 2),
                                   n_features=r['n_features'], train_s=round(r['train_s'], 1))
                              for r in g[('fused+woa', 'CB-SVM')]]
    comps = {}
    for name, accs in [(k, [r['acc'] for r in sorted(v, key=lambda r: r['fold'])]) for k, v in gb.items()] + \
                      [(c, [r['acc'] for r in g[(c, 'CB-SVM')]]) for c in ['fused', 'fused+pca', 'fused+relieff', 'fused+bpso', 'fused+bgwo', 'fused+ga']]:
        accs = np.array(accs)
        t = stats.ttest_rel(a4, accs)
        w = stats.wilcoxon(a4, accs) if np.any(a4 != accs) else None
        comps[name] = dict(mean_diff_pp=round((a4 - accs).mean() * 100, 2), t_p=float(t.pvalue), wilcoxon_p=float(w.pvalue) if w else 1.0)
    S['stats']['paired_vs_proposed'] = comps
    md.append('\n## Stats\n' + json.dumps(S['stats'], indent=1) + '\nPer fold: ' + json.dumps(S['per_fold_proposed']))
    json.dump(S, open(f'{rdir}/summary.json', 'w'), indent=1)
    open(f'{rdir}/summary.md', 'w').write('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main(*sys.argv[1:4])
