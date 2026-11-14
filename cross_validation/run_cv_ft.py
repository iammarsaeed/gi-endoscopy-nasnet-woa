"""Step 3 (improved pipeline): same protocol and tasks as run_cv.py, but with per-fold fine-tuned NASNet-Mobile
pooled features (data/feats_ft_fold{k}.npy) and test-time augmentation (TTA): SVM decision scores are averaged
over the 5 rotated/flipped versions of each test image. Results without TTA are stored as *_single.
Usage: python -m cross_validation.run_cv_ft <data.npz> <feats_pattern> <out_dir> <n_jobs>
"""
import json, os, sys, time, warnings
import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from cross_validation import run_cv
from cross_validation.run_cv import SEED, KERNELS, stream, select, relieff, make_svm


def metrics(yt, p):
    pr, rc, f1, _ = precision_recall_fscore_support(yt, p, average='macro', zero_division=0)
    return float(accuracy_score(yt, p)), float(pr), float(rc), float(f1)


def fit_eval(pattern, y, cfg, kernel, fold, tr, te, sel):
    F = np.load(pattern.format(k=fold), mmap_mode='r')
    base = cfg.split('+')[0].replace('_noaug', ''); S = stream(F, base)
    augs = [0] if cfg.endswith('_noaug') else range(5)
    Xtr = np.concatenate([np.asarray(S[a][tr]) for a in augs]); ytr = np.tile(y[tr], len(augs))
    Xte = [np.asarray(S[a][te]) for a in range(5)]; extra = {}
    if cfg.endswith('+pca'):
        t = time.perf_counter(); sc0 = StandardScaler().fit(Xtr)
        pca = PCA(n_components=0.95, svd_solver='full', random_state=SEED).fit(sc0.transform(Xtr))
        Xtr = pca.transform(sc0.transform(Xtr)); Xte = [pca.transform(sc0.transform(x)) for x in Xte]
        extra['selection_s'] = time.perf_counter() - t
    elif sel is not None:
        Xtr = Xtr[:, sel]; Xte = [x[:, sel] for x in Xte]
    sc = StandardScaler().fit(Xtr); Xtr = sc.transform(Xtr); Xte = [sc.transform(x) for x in Xte]
    clf = make_svm(kernel, Xtr.shape[1])
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always', ConvergenceWarning)
        t = time.perf_counter(); clf.fit(Xtr, ytr); train_s = time.perf_counter() - t
    t = time.perf_counter(); p1 = clf.predict(Xte[0]); pred_s = time.perf_counter() - t
    dec = np.mean([clf.decision_function(x) for x in Xte], axis=0); ptta = clf.classes_[dec.argmax(1)]
    a, p, r, f = metrics(y[te], ptta); a1, p1m, r1, f1 = metrics(y[te], p1)
    return dict(cfg=cfg, kernel=kernel, fold=fold, n_features=int(Xtr.shape[1]), acc=a, precision=p, recall=r, f1=f,
                acc_single=a1, precision_single=p1m, recall_single=r1, f1_single=f1, train_s=train_s,
                pred_speed=len(te) / pred_s, pred=ptta.tolist(), pred_single=p1.tolist(), te=te.tolist(),
                converged=not any(issubclass(x.category, ConvergenceWarning) for x in wl), **extra)


def main(data_path, pattern, out_dir, n_jobs):
    os.makedirs(f'{out_dir}/tasks', exist_ok=True); n_jobs = int(n_jobs)
    y = np.load(data_path)['y']
    folds = [(np.array(a), np.array(b)) for a, b in json.load(open(os.environ.get('FOLDS', 'results/folds.json')))]
    json.dump([[a.tolist(), b.tolist()] for a, b in folds], open(f'{out_dir}/folds.json', 'w'))
    methods = ['fused+woa', 'orig+woa', 'bbhe+woa', 'fused+bpso', 'fused+bgwo', 'fused+ga']
    t0 = time.time(); jobs = [(m, k) for k in range(5) for m in methods]
    res = Parallel(n_jobs=n_jobs)(delayed(select)(m, np.load(pattern.format(k=k), mmap_mode='r'), y, folds[k][0], k)
                                  for m, k in jobs)
    sels = {(m, k): r for (m, k), r in zip(jobs, res)}
    for k in range(5):
        F = np.load(pattern.format(k=k), mmap_mode='r'); t = time.perf_counter()
        idx = relieff(np.asarray(stream(F, 'fused')[0][folds[k][0]]), y[folds[k][0]], len(sels[('fused+woa', k)][0]),
                      np.random.default_rng(SEED + k))
        sels[('fused+relieff', k)] = (idx, time.perf_counter() - t)
    json.dump({f'{m}|{k}': dict(n=len(i), seconds=s, idx=i.tolist()) for (m, k), (i, s) in sels.items()},
              open(f'{out_dir}/selections.json', 'w'))
    print(f'selection done in {time.time()-t0:.0f}s:', {m: [len(sels[(m, k)][0]) for k in range(5)]
                                                        for m in methods + ['fused+relieff']}, flush=True)
    tasks = [(c, kern, k) for c in ['orig', 'bbhe', 'fused', 'fused+woa'] for kern in KERNELS for k in range(5)]
    tasks += [(c, 'CB-SVM', k) for c in ['orig_noaug', 'orig+woa', 'bbhe+woa', 'fused+pca', 'fused+relieff',
                                          'fused+bpso', 'fused+bgwo', 'fused+ga'] for k in range(5)]
    tasks.sort(key=lambda t: (t[1] != 'L-SVM', 'fused' not in t[0]))

    def one(cfg, kern, k):
        path = f'{out_dir}/tasks/{cfg}__{kern}__{k}.json'
        if not os.path.exists(path):
            r = fit_eval(pattern, y, cfg, kern, k, folds[k][0], folds[k][1], sels[(cfg, k)][0] if (cfg, k) in sels else None)
            json.dump(r, open(path + '.tmp', 'w')); os.replace(path + '.tmp', path)
        return json.load(open(path))
    t0 = time.time()
    out = Parallel(n_jobs=n_jobs)(delayed(one)(c, kern, k) for c, kern, k in tasks)
    for o in out:
        if (o['cfg'], o['fold']) in sels: o['selection_s'] = sels[(o['cfg'], o['fold'])][1]
    json.dump(out, open(f'{out_dir}/svm_results.json', 'w'))
    print(f'SVM stage done in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:5])
