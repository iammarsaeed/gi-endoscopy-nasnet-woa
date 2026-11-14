"""Binary WOA with an S-shaped transfer function, and a random-search control.

bwoa: whales hold binary vectors x in {0,1}^D. The continuous WOA update (Eqs. 10-15: encircling, random-whale
exploration when |A| >= 1, spiral) is computed from the binary positions, and each bit is then resampled with
probability T(v) = 1 / (1 + exp(-10 (v - 0.5))) of being 1 (S-shaped transfer function centred on 0.5, cf.
Mirjalili & Lewis 2013). Population size and iterations come from run_cv.POP and run_cv.ITERS.

rand: POP x (ITERS + 1) masks drawn independently with each feature selected with probability 0.5 (the same
distribution as the initial population); the best mask is returned.

Importing this module registers both functions in run_cv, so getattr(run_cv, 'bwoa') works.
"""
import numpy as np
import run_cv


def bwoa(fit, D, rng, b=1.0):
    P, T = run_cv.POP, run_cv.ITERS
    X = (rng.random((P, D)) > 0.5).astype(float)
    f = np.array([fit(x > 0.5) for x in X]); bi = f.argmin(); best, bestf = X[bi].copy(), f[bi]
    for t in range(T):
        a = 2 - 2 * t / T
        for i in range(P):
            A = 2 * a * rng.random() - a; C = 2 * rng.random(); p = rng.random(); l = rng.uniform(-1, 1)
            if p < 0.5:
                ref = best if abs(A) < 1 else X[rng.integers(P)]
                v = ref - A * np.abs(C * ref - X[i])
            else:
                v = np.abs(best - X[i]) * np.exp(b * l) * np.cos(2 * np.pi * l) + best
            X[i] = (rng.random(D) < 1 / (1 + np.exp(-10 * (v - 0.5)))).astype(float)
            fi = fit(X[i] > 0.5)
            if fi < bestf: best, bestf = X[i].copy(), fi
    return best > 0.5


def rand(fit, D, rng):
    best, bestf = None, np.inf
    for _ in range(run_cv.POP * (run_cv.ITERS + 1)):
        m = rng.random(D) > 0.5
        f = fit(m)
        if f < bestf: best, bestf = m, f
    return best


run_cv.bwoa = bwoa
run_cv.rand = rand
