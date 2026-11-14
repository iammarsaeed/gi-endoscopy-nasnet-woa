"""Two-stage feature selection: ReliefF keeps the 200 highest-ranked fused features, then a binary WOA with an S-shaped
transfer function (50 whales x 100 iterations, 5,050 fitness evaluations) searches them. Runs every result that depends
on feature selection (Experiment 4, ablation, comparison with ReliefF, BPSO, BGWO, GA and random search, frozen-feature
control), writing task files that report.py turns into the tables and figures. All selection uses the training
originals of each fold only. Configurations without selection (Experiments 1-3, PCA) are copied from run_cv_ft.py's output.
Usage: python twostage_pipeline.py <data.npz> <feats_pattern> <frozen_feats.npy> <out_dir> <n_jobs>
"""
import json, os, shutil, sys, time
import numpy as np
from joblib import Parallel, delayed
import run_cv
from run_cv import SEED, stream, relieff
import run_cv_ft
from woa_budget import Tracked
import bwoa  # registers run_cv.bwoa and run_cv.rand

K, POP, ITERS = 200, 50, 100
KERNELS = run_cv.KERNELS


def load_X(pattern, fold, base, tr):
    F = np.load(pattern.format(k=fold), mmap_mode='r')
    return np.asarray(stream(F, base)[0][tr])


def select(pattern, y, tr, fold, base, method, out):
    """Two-stage selection on one stream; returns global indices into that stream's feature vector."""
    if os.path.exists(out): return json.load(open(out))
    X = load_X(pattern, fold, base, tr); yt = y[tr]
    t = time.perf_counter(); order = relieff(X, yt, X.shape[1], np.random.default_rng(SEED + fold)); rank_s = time.perf_counter() - t
    cand = order[:K]
    run_cv.POP, run_cv.ITERS = POP, ITERS
    fit = Tracked(X[:, cand], yt); rng = np.random.default_rng(SEED + fold)
    t = time.perf_counter(); mask = (getattr(__import__('bwoa'), method) if method in ('bwoa', 'rand') else getattr(run_cv, method))(fit, K, rng); wrap_s = time.perf_counter() - t
    r = dict(base=base, method=method, fold=fold, n=int(mask.sum()), rank_s=rank_s, wrap_s=wrap_s, seconds=rank_s + wrap_s,
             best_fitness=float(fit.best), evals=len(fit.curve), curve=[float(c) for c in fit.curve],
             idx=cand[mask].tolist(), order=order[:K].tolist())
    json.dump(r, open(out + '.tmp', 'w')); os.replace(out + '.tmp', out)
    return r


def main(data_path, pattern, frozen, out_dir, n_jobs):
    n_jobs = int(n_jobs)
    os.makedirs(f'{out_dir}/tasks', exist_ok=True); os.makedirs(f'{out_dir}/sel', exist_ok=True)
    y = np.load(data_path)['y']
    folds = [(np.array(a), np.array(b)) for a, b in json.load(open('results_ft/folds.json'))]
    for f in ('folds.json', 'baselines.json'): shutil.copy(f'results_ft/{f}', f'{out_dir}/{f}')
    keep = ('orig__', 'bbhe__', 'fused__', 'orig_noaug__', 'fused+pca__')
    for f in os.listdir('results_ft/tasks'):
        if f.startswith(keep): shutil.copy(f'results_ft/tasks/{f}', f'{out_dir}/tasks/{f}')
    # ---- selection (fine-tuned features + frozen features for the ablation row)
    jobs = [(pattern, b, m) for b, m in [('fused', 'bwoa'), ('orig', 'bwoa'), ('bbhe', 'bwoa'), ('fused', 'rand'),
                                         ('fused', 'bpso'), ('fused', 'bgwo'), ('fused', 'ga')]]
    jobs.append((frozen, 'fused', 'bwoa'))
    t0 = time.time()
    tasks = [(p, b, m, k) for p, b, m in jobs for k in range(5)]
    res = Parallel(n_jobs=n_jobs)(delayed(select)(p, y, folds[k][0], k, b, m,
                                                  f"{out_dir}/sel/{'frozen_' if p == frozen else ''}{b}+{m}_{k}.json")
                                  for p, b, m, k in tasks)
    sels = {}
    for (p, b, m, k), r in zip(tasks, res):
        cfg = f"{b}+{m.replace('bwoa', 'woa')}" if p != frozen else 'frozen_fused+woa'
        sels[(cfg, k)] = (np.array(r['idx']), r['seconds'])
    for k in range(5):   # ReliefF: top-n of the same ranking, n = WOA subset size
        r = json.load(open(f'{out_dir}/sel/fused+bwoa_{k}.json'))
        sels[('fused+relieff', k)] = (np.array(r['order'][:r['n']]), r['rank_s'])
        sels[('fused+relieff200', k)] = (np.array(r['order'][:200]), r['rank_s'])
    json.dump({f'{c}|{k}': dict(n=len(i), seconds=s, idx=i.tolist()) for (c, k), (i, s) in sels.items()},
              open(f'{out_dir}/selections.json', 'w'))
    print(f'selection done in {time.time()-t0:.0f}s:',
          {c: [len(sels[(c, k)][0]) for k in range(5)] for c in sorted({c for c, _ in sels})}, flush=True)

    # ---- SVMs
    svm = [('fused+woa', kern, k) for kern in KERNELS for k in range(5)]
    svm += [(c, 'CB-SVM', k) for c in ['orig+woa', 'bbhe+woa', 'fused+relieff', 'fused+relieff200', 'fused+rand', 'fused+bpso', 'fused+bgwo', 'fused+ga']
            for k in range(5)]
    svm.sort(key=lambda t: t[1] != 'L-SVM')

    def one(cfg, kern, k):
        path = f'{out_dir}/tasks/{cfg}__{kern}__{k}.json'
        if not os.path.exists(path):
            r = run_cv_ft.fit_eval(pattern, y, cfg, kern, k, folds[k][0], folds[k][1], sels[(cfg, k)][0])
            json.dump(r, open(path + '.tmp', 'w')); os.replace(path + '.tmp', path)
        return json.load(open(path))

    def frozen_one(k):
        path = f'{out_dir}/frozen_fused+woa__CB-SVM__{k}.json'
        if not os.path.exists(path):
            r = run_cv.fit_eval(frozen, y, 'fused+woa', 'CB-SVM', k, folds[k][0], folds[k][1], sels[('frozen_fused+woa', k)][0])
            json.dump(r, open(path + '.tmp', 'w')); os.replace(path + '.tmp', path)
        return json.load(open(path))

    t0 = time.time()
    Parallel(n_jobs=n_jobs)([delayed(one)(c, kern, k) for c, kern, k in svm] + [delayed(frozen_one)(k) for k in range(5)])
    out = [json.load(open(f'{out_dir}/tasks/{f}')) for f in sorted(os.listdir(f'{out_dir}/tasks'))]
    for o in out:
        if (o['cfg'], o['fold']) in sels: o['selection_s'] = sels[(o['cfg'], o['fold'])][1]
    json.dump(out, open(f'{out_dir}/svm_results.json', 'w'))
    print(f'SVM stage done in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:6])
