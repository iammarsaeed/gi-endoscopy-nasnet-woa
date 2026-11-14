"""Step 3: leakage-free 5-fold cross-validation.

Folds are defined on the 8,000 ORIGINAL images (stratified, seed 42). In each fold:
  - training set = all 5 augmented versions of the training images (6,400 x 5 = 32,000);
  - test set = the original (non-augmented) version of the held-out images (1,600);
  - feature selection (WOA and the comparison methods) uses training images only: its fitness is the error of a
    5-NN classifier on an inner stratified 80/20 split of the training originals;
  - the SVMs are trained on the selected training features and evaluated on the test fold.
Usage: python run_cv.py <data.npz> <feats.npy> <out_dir> <n_jobs>
"""
import json, os, sys, time
import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings

ALPHA, POP, ITERS, SEED = 0.99, 10, 20, 42
KERNELS = ['L-SVM', 'Q-SVM', 'CB-SVM', 'MG-SVM', 'CG-SVM']


def stream(F, name):
    if name == 'orig': return F[0]
    if name == 'bbhe': return F[1]
    return np.concatenate([F[0], F[1]], axis=-1)          # fused: (5, N, 2000)


# ---------------------------------------------------------------- feature-selection fitness
class Fitness:
    def __init__(self, X, y):
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
        sc = StandardScaler().fit(Xtr)
        self.Xtr, self.Xva, self.ytr, self.yva = sc.transform(Xtr), sc.transform(Xva), ytr, yva
        self.D = X.shape[1]; self.calls = 0

    def __call__(self, mask):
        self.calls += 1
        k = int(mask.sum())
        if k == 0: return 1.0
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=1).fit(self.Xtr[:, mask], self.ytr)
        err = 1.0 - (knn.predict(self.Xva[:, mask]) == self.yva).mean()
        return ALPHA * err + (1 - ALPHA) * k / self.D


def woa(fit, D, rng, b=1.0):
    """Binary WOA (Mirjalili & Lewis 2016) with positions in [0,1]; feature selected when position > 0.5."""
    X = rng.random((POP, D)); f = np.array([fit(x > 0.5) for x in X])
    bi = f.argmin(); best, bestf = X[bi].copy(), f[bi]
    for t in range(ITERS):
        a = 2 - 2 * t / ITERS
        for i in range(POP):
            A = 2 * a * rng.random() - a; C = 2 * rng.random(); p = rng.random(); l = rng.uniform(-1, 1)
            if p < 0.5:
                ref = best if abs(A) < 1 else X[rng.integers(POP)]
                X[i] = ref - A * np.abs(C * ref - X[i])
            else:
                X[i] = np.abs(best - X[i]) * np.exp(b * l) * np.cos(2 * np.pi * l) + best
            X[i] = np.clip(X[i], 0, 1)
            fi = fit(X[i] > 0.5)
            if fi < bestf: best, bestf = X[i].copy(), fi
    return best > 0.5


def bpso(fit, D, rng):
    X = rng.random((POP, D)) > 0.5; V = rng.uniform(-1, 1, (POP, D))
    f = np.array([fit(x) for x in X]); P, Pf = X.copy(), f.copy(); g = P[Pf.argmin()].copy(); gf = Pf.min()
    for t in range(ITERS):
        w = 0.9 - 0.5 * t / ITERS
        V = np.clip(w * V + 2 * rng.random((POP, D)) * (P.astype(float) - X) + 2 * rng.random((POP, D)) * (g.astype(float) - X), -6, 6)
        X = rng.random((POP, D)) < 1 / (1 + np.exp(-V))
        for i in range(POP):
            fi = fit(X[i])
            if fi < Pf[i]: P[i], Pf[i] = X[i].copy(), fi
            if fi < gf: g, gf = X[i].copy(), fi
    return g


def bgwo(fit, D, rng):
    X = rng.random((POP, D)); f = np.array([fit(x > 0.5) for x in X])
    best_mask, best_f = X[f.argmin()] > 0.5, f.min()
    for t in range(ITERS):
        a = 2 - 2 * t / ITERS
        order = np.argsort(f); alpha, beta, delta = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()
        for i in range(POP):
            new = np.zeros(D)
            for leader in (alpha, beta, delta):
                A = 2 * a * rng.random(D) - a; C = 2 * rng.random(D)
                new += leader - A * np.abs(C * leader - X[i])
            X[i] = np.clip(new / 3, 0, 1); f[i] = fit(X[i] > 0.5)
            if f[i] < best_f: best_mask, best_f = X[i] > 0.5, f[i]
    return best_mask


def ga(fit, D, rng):
    X = rng.random((POP, D)) > 0.5; f = np.array([fit(x) for x in X])
    for t in range(ITERS):
        elite = X[f.argmin()].copy(); ef = f.min(); kids = [elite]; kf = [ef]
        while len(kids) < POP:
            par = []
            for _ in range(2):
                i, j = rng.integers(POP, size=2); par.append(X[i] if f[i] < f[j] else X[j])
            child = np.where(rng.random(D) < 0.5, par[0], par[1]) if rng.random() < 0.9 else par[0].copy()
            child ^= rng.random(D) < 0.01
            kids.append(child); kf.append(fit(child))
        X, f = np.array(kids), np.array(kf)
    return X[f.argmin()]


def relieff(X, y, k_top, rng, m=1000, k=10):
    Xs = (X - X.min(0)) / (np.ptp(X, 0) + 1e-12); W = np.zeros(X.shape[1])
    classes, counts = np.unique(y, return_counts=True); prior = dict(zip(classes, counts / len(y)))
    m = min(m, len(y))
    for idx in rng.choice(len(y), size=m, replace=False):
        d = np.abs(Xs - Xs[idx]).sum(1); d[idx] = np.inf
        for c in classes:
            nn = np.where(y == c)[0]; nn = nn[np.argsort(d[nn])[:k]]
            diff = np.abs(Xs[nn] - Xs[idx]).mean(0)
            if c == y[idx]: W -= diff / m
            else: W += prior[c] / (1 - prior[y[idx]]) * diff / m
    return np.argsort(-W)[:k_top]


def select(method, F, y, tr, fold):
    """Feature selection on TRAINING originals only. Returns (indices or None, seconds, extra)."""
    rng = np.random.default_rng(SEED + fold)
    base = method.split('+')[0]; X = stream(F, base)[0][tr]; yt = y[tr]; D = X.shape[1]
    t = time.perf_counter()
    if method.endswith('+woa'): mask = woa(Fitness(X, yt), D, rng)
    elif method.endswith('+bpso'): mask = bpso(Fitness(X, yt), D, rng)
    elif method.endswith('+bgwo'): mask = bgwo(Fitness(X, yt), D, rng)
    elif method.endswith('+ga'): mask = ga(Fitness(X, yt), D, rng)
    else: raise ValueError(method)
    return np.flatnonzero(mask), time.perf_counter() - t


def make_svm(kernel, P):
    if kernel == 'L-SVM': return OneVsOneClassifier(LinearSVC(C=1, loss='hinge', dual=True, max_iter=100000, random_state=SEED))
    if kernel == 'Q-SVM': return SVC(kernel='poly', degree=2, max_iter=5_000_000, gamma=1 / P, coef0=1, C=1, cache_size=3000)
    if kernel == 'CB-SVM': return SVC(kernel='poly', degree=3, max_iter=5_000_000, gamma=1 / P, coef0=1, C=1, cache_size=3000)
    if kernel == 'MG-SVM': return SVC(kernel='rbf', gamma=1 / P, max_iter=5_000_000, C=1, cache_size=3000)
    if kernel == 'CG-SVM': return SVC(kernel='rbf', gamma=1 / (16 * P), max_iter=5_000_000, C=1, cache_size=3000)


def fit_eval(feats_path, y, cfg, kernel, fold, tr, te, sel):
    F = np.load(feats_path, mmap_mode='r')
    base = cfg.split('+')[0].replace('_noaug', '')
    S = stream(F, base)
    augs = [0] if cfg.endswith('_noaug') else range(5)
    Xtr = np.concatenate([np.asarray(S[a][tr]) for a in augs]); ytr = np.tile(y[tr], len(augs))
    Xte = np.asarray(S[0][te]); extra = {}
    if cfg.endswith('+pca'):
        t = time.perf_counter(); sc0 = StandardScaler().fit(Xtr)
        pca = PCA(n_components=0.95, svd_solver='full', random_state=SEED).fit(sc0.transform(Xtr))
        Xtr, Xte = pca.transform(sc0.transform(Xtr)), pca.transform(sc0.transform(Xte))
        extra['selection_s'] = time.perf_counter() - t
    elif sel is not None:
        Xtr, Xte = Xtr[:, sel], Xte[:, sel]
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    clf = make_svm(kernel, Xtr.shape[1])
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter('always', ConvergenceWarning)
        t = time.perf_counter(); clf.fit(Xtr, ytr); train_s = time.perf_counter() - t
    extra['converged'] = not any(issubclass(x.category, ConvergenceWarning) for x in wlist)
    t = time.perf_counter(); pred = clf.predict(Xte); pred_s = time.perf_counter() - t
    p, r, f1, _ = precision_recall_fscore_support(y[te], pred, average='macro', zero_division=0)
    return dict(cfg=cfg, kernel=kernel, fold=fold, n_features=int(Xtr.shape[1]), acc=float(accuracy_score(y[te], pred)),
                precision=float(p), recall=float(r), f1=float(f1), train_s=train_s, pred_speed=len(te) / pred_s,
                pred=pred.tolist(), te=te.tolist(), **extra)


def main(data_path, feats_path, out_dir, n_jobs):
    os.makedirs(out_dir, exist_ok=True); n_jobs = int(n_jobs)
    y = np.load(data_path)['y']; F = np.load(feats_path, mmap_mode='r')
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    json.dump([[a.tolist(), b.tolist()] for a, b in folds], open(f'{out_dir}/folds.json', 'w'))

    # Stage 1: feature selection inside each training fold
    sel_methods = ['fused+woa', 'orig+woa', 'bbhe+woa', 'fused+bpso', 'fused+bgwo', 'fused+ga']
    t0 = time.time()
    jobs = [(m, k) for k in range(5) for m in sel_methods]
    res = Parallel(n_jobs=n_jobs, verbose=0)(delayed(select)(m, F, y, folds[k][0], k) for m, k in jobs)
    sels = {}
    for (m, k), (idx, secs) in zip(jobs, res):
        sels[(m, k)] = (idx, secs)
    for k in range(5):   # ReliefF keeps as many features as WOA selected in that fold
        t = time.perf_counter()
        idx = relieff(stream(F, 'fused')[0][folds[k][0]], y[folds[k][0]], len(sels[('fused+woa', k)][0]),
                      np.random.default_rng(SEED + k))
        sels[('fused+relieff', k)] = (idx, time.perf_counter() - t)
    json.dump({f'{m}|{k}': dict(n=len(i), seconds=s, idx=i.tolist()) for (m, k), (i, s) in sels.items()},
              open(f'{out_dir}/selections.json', 'w'))
    print(f'selection done in {time.time()-t0:.0f}s:',
          {m: [len(sels[(m, k)][0]) for k in range(5)] for m in sel_methods + ['fused+relieff']}, flush=True)

    # Stage 2: SVMs
    tasks = []
    for cfg in ['orig', 'bbhe', 'fused', 'fused+woa']:
        for kern in KERNELS:
            for k in range(5): tasks.append((cfg, kern, k))
    for cfg in ['orig_noaug', 'orig+woa', 'bbhe+woa', 'fused+pca', 'fused+relieff', 'fused+bpso', 'fused+bgwo', 'fused+ga']:
        for k in range(5): tasks.append((cfg, 'CB-SVM', k))
    tasks.sort(key=lambda t: (t[1] != 'L-SVM', 'fused' not in t[0]))   # longest first
    t0 = time.time(); os.makedirs(f'{out_dir}/tasks', exist_ok=True)
    def one(cfg, kern, k):
        path = f'{out_dir}/tasks/{cfg}__{kern}__{k}.json'
        if not os.path.exists(path):
            r = fit_eval(feats_path, y, cfg, kern, k, folds[k][0], folds[k][1], sels[(cfg, k)][0] if (cfg, k) in sels else None)
            json.dump(r, open(path + '.tmp', 'w')); os.replace(path + '.tmp', path)
        return json.load(open(path))
    out = Parallel(n_jobs=n_jobs, verbose=0)(delayed(one)(cfg, kern, k) for cfg, kern, k in tasks)
    for o in out:
        if (o['cfg'], o['fold']) in sels: o['selection_s'] = sels[(o['cfg'], o['fold'])][1]
    json.dump(out, open(f'{out_dir}/svm_results.json', 'w'))
    print(f'SVM stage done in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:5])
