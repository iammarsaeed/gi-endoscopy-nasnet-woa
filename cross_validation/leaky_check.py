"""Control experiment: reproduce the ORIGINAL (leaky) protocol with the same features and classifier.

Rows = all 40,000 augmented images; 5 folds assigned at random over rows (stratified), so augmented versions of one
image fall into different folds; WOA (same settings) is run once on all 40,000 rows before cross-validation.
Usage: python -m cross_validation.leaky_check <data.npz> <feats.npy> <out.json> <n_jobs>
"""
import json, sys, time
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from cross_validation import run_cv


def fold(X, y, tr, te, k):
    sc = StandardScaler().fit(X[tr]); clf = run_cv.make_svm('CB-SVM', X.shape[1])
    t = time.perf_counter(); clf.fit(sc.transform(X[tr]), y[tr]); ts = time.perf_counter() - t
    p = clf.predict(sc.transform(X[te]))
    pr, rc, f1, _ = precision_recall_fscore_support(y[te], p, average='macro', zero_division=0)
    return dict(fold=k, acc=float(accuracy_score(y[te], p)), f1=float(f1), train_s=ts)


def main(data_path, feats_path, out, n_jobs):
    y0 = np.load(data_path)['y']; F = np.load(feats_path)
    fused = np.concatenate([F[0], F[1]], axis=-1)            # (5, 8000, 2000)
    X = fused.reshape(-1, fused.shape[-1]); y = np.tile(y0, 5)  # 40,000 rows, original image i at rows i + 8000*a
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    res = {}
    res['fused_CB'] = Parallel(n_jobs=int(n_jobs))(delayed(fold)(X, y, tr, te, k) for k, (tr, te) in enumerate(folds))
    rng = np.random.default_rng(42)
    t = time.perf_counter(); mask = run_cv.woa(run_cv.Fitness(X, y), X.shape[1], rng); ws = time.perf_counter() - t
    Xw = X[:, mask]
    res['fused_woa_CB'] = Parallel(n_jobs=int(n_jobs))(delayed(fold)(Xw, y, tr, te, k) for k, (tr, te) in enumerate(folds))
    res['woa_n_features'] = int(mask.sum()); res['woa_seconds'] = ws
    # how many test rows have an augmented sibling of the same image in the training folds
    img = np.tile(np.arange(len(y0)), 5); share = []
    for tr, te in folds:
        trimg = set(img[tr]); share.append(float(np.mean([i in trimg for i in img[te]])))
    res['test_rows_with_sibling_in_train'] = share
    for k in ('fused_CB', 'fused_woa_CB'):
        a = np.array([r['acc'] for r in res[k]]) * 100
        print(k, f'{a.mean():.1f} ± {a.std(ddof=1):.1f}', flush=True)
    print('WOA features (all data):', res['woa_n_features'], '| test rows with sibling in train:', np.round(share, 3), flush=True)
    json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:5])
