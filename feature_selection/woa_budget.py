"""Search budget of the wrapper methods on all 2,112 fused features with threshold binarization (positions in [0, 1],
feature selected when above 0.5). Also provides Tracked, the fitness wrapper that records convergence curves and is used
by the two-stage scripts.
Usage: python -m feature_selection.woa_budget <data.npz> <feats_pattern> <out_dir> <n_jobs>
"""
import json, os, sys, time
import numpy as np
from joblib import Parallel, delayed
from cross_validation import run_cv
from cross_validation.run_cv import SEED, stream, Fitness
from cross_validation.run_cv_ft import fit_eval

BUDGETS = [(10, 20), (30, 50), (30, 100), (50, 100)]
OTHERS = ['bpso', 'bgwo', 'ga']          # run at the largest budget for a fair comparison


class Tracked(Fitness):
    """Fitness that also records the best value found so far after every evaluation."""
    def __init__(self, X, y):
        super().__init__(X, y); self.best = np.inf; self.curve = []

    def __call__(self, mask):
        f = super().__call__(mask); self.best = min(self.best, f); self.curve.append(self.best)
        return f


def run_select(pattern, y, tr, fold, method, pop, iters, out):
    if os.path.exists(out): return json.load(open(out))
    run_cv.POP, run_cv.ITERS = pop, iters
    F = np.load(pattern.format(k=fold), mmap_mode='r'); X = np.asarray(stream(F, 'fused')[0][tr]); yt = y[tr]
    fit = Tracked(X, yt); rng = np.random.default_rng(SEED + fold); t = time.perf_counter()
    mask = getattr(run_cv, method)(fit, X.shape[1], rng)
    sec = time.perf_counter() - t; idx = np.flatnonzero(mask)
    k = len(idx); err = (fit(mask) - (1 - run_cv.ALPHA) * k / X.shape[1]) / run_cv.ALPHA
    r = dict(method=method, pop=pop, iters=iters, fold=fold, n=k, seconds=sec, evals=len(fit.curve) - 1,
             best_fitness=float(fit.best), inner_error=float(err), curve=[float(c) for c in fit.curve[:-1]], idx=idx.tolist())
    json.dump(r, open(out + '.tmp', 'w')); os.replace(out + '.tmp', out)
    return r


def run_svm(pattern, y, folds, r, out):
    if os.path.exists(out): return json.load(open(out))
    k = r['fold']; tr, te = folds[k]
    e = fit_eval(pattern, y, 'fused+sel', 'CB-SVM', k, tr, te, np.array(r['idx']))
    e.update(method=r['method'], pop=r['pop'], iters=r['iters'], n=r['n'], seconds=r['seconds'])
    for key in ('pred', 'pred_single', 'te'): e.pop(key)
    json.dump(e, open(out + '.tmp', 'w')); os.replace(out + '.tmp', out)
    return e


def main(data_path, pattern, out_dir, n_jobs):
    os.makedirs(f'{out_dir}/sel', exist_ok=True); os.makedirs(f'{out_dir}/svm', exist_ok=True)
    y = np.load(data_path)['y']
    folds = [(np.array(a), np.array(b)) for a, b in json.load(open('results_ft/folds.json'))]
    jobs = [('woa', p, i) for p, i in BUDGETS] + [(m, *BUDGETS[-1]) for m in OTHERS]
    tasks = sorted([(m, p, i, k) for m, p, i in jobs for k in range(5)], key=lambda t: -t[1] * t[2])
    t0 = time.time()
    sels = Parallel(n_jobs=int(n_jobs))(delayed(run_select)(pattern, y, folds[k][0], k, m, p, i,
                                                            f'{out_dir}/sel/{m}_{p}x{i}_{k}.json') for m, p, i, k in tasks)
    print(f'selection done in {time.time()-t0:.0f}s', flush=True); t0 = time.time()
    res = Parallel(n_jobs=int(n_jobs))(delayed(run_svm)(pattern, y, folds, r,
                                                        f"{out_dir}/svm/{r['method']}_{r['pop']}x{r['iters']}_{r['fold']}.json") for r in sels)
    print(f'SVM done in {time.time()-t0:.0f}s', flush=True)
    rows = {}
    for s, e in zip(sels, res):
        key = f"{s['method']} {s['pop']}x{s['iters']}"; rows.setdefault(key, []).append((s, e))
    summary = {}
    for key, v in rows.items():
        g = lambda f: [f(s, e) for s, e in v]
        summary[key] = {name: [float(np.mean(x)), float(np.std(x, ddof=1))] for name, x in [
            ('evals', g(lambda s, e: s['evals'])), ('n', g(lambda s, e: s['n'])),
            ('fitness', g(lambda s, e: s['best_fitness'])), ('inner_error', g(lambda s, e: s['inner_error'])),
            ('sel_seconds', g(lambda s, e: s['seconds'])), ('acc', g(lambda s, e: 100 * e['acc'])),
            ('f1', g(lambda s, e: 100 * e['f1'])), ('acc_single', g(lambda s, e: 100 * e['acc_single'])),
            ('train_s', g(lambda s, e: e['train_s'])), ('pred_speed', g(lambda s, e: e['pred_speed']))]}
        summary[key]['acc_folds'] = g(lambda s, e: 100 * e['acc'])
        summary[key]['n_folds'] = g(lambda s, e: s['n'])
    json.dump(summary, open(f'{out_dir}/summary.json', 'w'), indent=1)
    for key, s in summary.items():
        print(f"{key:16s} evals {s['evals'][0]:6.0f}  n {s['n'][0]:6.0f}±{s['n'][1]:4.0f}  fit {s['fitness'][0]:.4f}  "
              f"inner_err {100*s['inner_error'][0]:.2f}  acc {s['acc'][0]:.1f}±{s['acc'][1]:.1f}  "
              f"train {s['train_s'][0]:.1f}s  sel {s['sel_seconds'][0]:.0f}s", flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:5])
