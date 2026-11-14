"""Search-space and budget analysis for the WOA stage: binary WOA (S-shaped transfer function) on all 2,112 fused
features, the top 200 or the top 100 ReliefF features, with 10 x 20 up to 100 x 200 whales x iterations, plus random
search with the same budget. Same folds, fitness (Eq. 16) and CB-SVM + test-time-augmentation evaluation as
twostage_pipeline.py. Records the best fitness after every evaluation (convergence curves).
Usage: python search_budget.py <data.npz> <feats_pattern> <out_dir> <n_jobs>
"""
import json, os, sys, time
import numpy as np
from joblib import Parallel, delayed
import run_cv
from run_cv import SEED, stream, relieff
from run_cv_ft import fit_eval
from woa_budget import Tracked
import bwoa  # registers run_cv.bwoa and run_cv.rand

# (K, method, agents, iterations); K = 2112 searches all fused features (no ReliefF stage)
CONFIGS = [(200, 'bwoa', 30, 100), (200, 'bwoa', 50, 100), (200, 'bwoa', 50, 200), (200, 'bwoa', 100, 200),
           (100, 'bwoa', 50, 100), (2112, 'bwoa', 10, 20), (2112, 'bwoa', 50, 100), (200, 'rand', 50, 100)]
KS = sorted({c[0] for c in CONFIGS})
if os.environ.get('SMOKE'):
    CONFIGS = [(100, 'bwoa', 5, 3)]; KS = [100]


def rank_fold(pattern, y, tr, fold, out):
    if os.path.exists(out): return json.load(open(out))
    F = np.load(pattern.format(k=fold), mmap_mode='r'); X = np.asarray(stream(F, 'fused')[0][tr])
    t = time.perf_counter(); order = relieff(X, y[tr], X.shape[1], np.random.default_rng(SEED + fold))
    r = dict(fold=fold, seconds=time.perf_counter() - t, order=order.tolist())
    json.dump(r, open(out + '.tmp', 'w')); os.replace(out + '.tmp', out)
    return r


def run_wrapper(pattern, y, tr, fold, order, K, method, pop, iters, out):
    if os.path.exists(out): return json.load(open(out))
    run_cv.POP, run_cv.ITERS = pop, iters
    cand = np.array(order[:K])
    F = np.load(pattern.format(k=fold), mmap_mode='r'); X = np.asarray(stream(F, 'fused')[0][tr][:, cand]); yt = y[tr]
    fit = Tracked(X, yt); rng = np.random.default_rng(SEED + fold); t = time.perf_counter()
    mask = (getattr(__import__('bwoa'), method) if method in ('bwoa', 'rand') else getattr(run_cv, method))(fit, K, rng)
    sec = time.perf_counter() - t; n = int(mask.sum())
    evals = len(fit.curve); best = float(fit.best)
    f_woa = float(fit(mask))
    # same-size filter control: top-n of the ranking, scored with the same fitness function
    m_rel = np.zeros(K, bool); m_rel[:n] = True; f_rel = float(fit(m_rel))
    m_all = np.ones(K, bool); f_all = float(fit(m_all))
    r = dict(method=method, K=K, pop=pop, iters=iters, fold=fold, n=n, seconds=sec, evals=evals,
             best_fitness=best, fitness_selected=f_woa, fitness_relieff_match=f_rel, fitness_all_K=f_all,
             init_best=float(fit.curve[pop - 1]), curve=[float(c) for c in fit.curve[:evals]],
             idx=cand[mask].tolist(), idx_relieff_match=cand[:n].tolist())
    json.dump(r, open(out + '.tmp', 'w')); os.replace(out + '.tmp', out)
    return r


def run_svm(pattern, y, folds, fold, idx, tag, out):
    if os.path.exists(out): return json.load(open(out))
    tr, te = folds[fold]
    e = fit_eval(pattern, y, 'fused+sel', 'CB-SVM', fold, tr, te, np.array(idx))
    e.update(tag=tag)
    for key in ('pred_single',): e.pop(key)
    json.dump(e, open(out + '.tmp', 'w')); os.replace(out + '.tmp', out)
    return e


def main(data_path, pattern, out_dir, n_jobs):
    n_jobs = int(n_jobs)
    for d in ('rank', 'sel', 'svm'): os.makedirs(f'{out_dir}/{d}', exist_ok=True)
    y = np.load(data_path)['y']
    folds = [(np.array(a), np.array(b)) for a, b in json.load(open('results_ft/folds.json'))]
    t0 = time.time()
    ranks = Parallel(n_jobs=5)(delayed(rank_fold)(pattern, y, folds[k][0], k, f'{out_dir}/rank/{k}.json') for k in range(5))
    print(f'ranking done in {time.time()-t0:.0f}s', flush=True); t0 = time.time()

    tasks = sorted([(m, p, i, K, k) for K, m, p, i in CONFIGS for k in range(5)], key=lambda t: -t[1] * t[2] * t[3])
    sels = Parallel(n_jobs=n_jobs)(delayed(run_wrapper)(pattern, y, folds[k][0], k, ranks[k]['order'], K, m, p, i,
                                                        f'{out_dir}/sel/{m}_K{K}_{p}x{i}_{k}.json') for m, p, i, K, k in tasks)
    print(f'wrapper search done in {time.time()-t0:.0f}s', flush=True); t0 = time.time()

    svm_jobs = []
    for s in sels:
        base = f"{s['method']}_K{s['K']}_{s['pop']}x{s['iters']}"
        svm_jobs.append((s['fold'], s['idx'], base))
        svm_jobs.append((s['fold'], s['idx_relieff_match'], f'relieffmatch_{base}'))
    for K in KS:
        for k in range(5): svm_jobs.append((k, ranks[k]['order'][:K], f'relieff_K{K}'))
    res = Parallel(n_jobs=n_jobs)(delayed(run_svm)(pattern, y, folds, k, idx, tag, f'{out_dir}/svm/{tag}_{k}.json')
                                  for k, idx, tag in svm_jobs)
    print(f'SVM done in {time.time()-t0:.0f}s', flush=True)

    acc = {}
    for e in res: acc.setdefault(e['tag'], {})[e['fold']] = e
    summary = {}
    for tag, d in acc.items():
        es = [d[k] for k in range(5)]
        g = lambda f: [f(e) for e in es]
        summary[tag] = {name: [float(np.mean(x)), float(np.std(x, ddof=1))] for name, x in [
            ('n', g(lambda e: e['n_features'])), ('acc', g(lambda e: 100 * e['acc'])), ('f1', g(lambda e: 100 * e['f1'])),
            ('acc_single', g(lambda e: 100 * e['acc_single'])), ('train_s', g(lambda e: e['train_s'])),
            ('pred_speed', g(lambda e: e['pred_speed']))]}
        summary[tag]['acc_folds'] = g(lambda e: 100 * e['acc'])
    for s_tag in {f"{s['method']}_K{s['K']}_{s['pop']}x{s['iters']}" for s in sels}:
        ss = [s for s in sels if f"{s['method']}_K{s['K']}_{s['pop']}x{s['iters']}" == s_tag]
        for name in ('evals', 'seconds', 'best_fitness', 'fitness_relieff_match', 'fitness_all_K', 'init_best'):
            x = [s[name] for s in ss]; summary[s_tag][name] = [float(np.mean(x)), float(np.std(x, ddof=1))]
    json.dump(summary, open(f'{out_dir}/summary.json', 'w'), indent=1)
    for tag in sorted(summary):
        s = summary[tag]
        extra = (f"  evals {s['evals'][0]:5.0f}  fit {s['init_best'][0]:.4f}->{s['best_fitness'][0]:.4f}"
                 f"  (relief same-n {s['fitness_relieff_match'][0]:.4f}, all-K {s['fitness_all_K'][0]:.4f})"
                 f"  sel {s['seconds'][0]:.0f}s") if 'evals' in s else ''
        print(f"{tag:34s} n {s['n'][0]:5.0f}±{s['n'][1]:4.0f}  acc {s['acc'][0]:.1f}±{s['acc'][1]:.1f}"
              f"  train {s['train_s'][0]:.1f}s{extra}", flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:5])
