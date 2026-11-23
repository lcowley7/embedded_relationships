"""analogy_geometry.py -- the analogy RSA primitives shared by analogy_rsa.py, frames_pairs.py and
sweep_analogies.py.

  cloud(ds, cl, L)     unit-normalised vectors for one cloud, in the canonical order of the mapping T
  moves(X, perm)       all i<j chords in canonical order; perm relabels which row plays each slot
  rsa(A, B, kind)      positions or moves RSA against the correspondence-permutation null
  direction(A, B)      mean cosine between corresponding moves
"""
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from relgeom import analogies as W, model_switch
from relgeom.paths import AXBENCH

NPERM = 5000; SEED = 0
unit = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)


def cloud(ds, cl, L):
    """Unit-normalised vectors for one cloud, in canonical T order."""
    d = model_switch.vec_dir(ds, L)
    w = torch.load(d / "DiffMean_weight.pt", weights_only=False).float().numpy()
    ci = pd.read_parquet(AXBENCH / ds / "concept_index.parquet")
    idx = {v: i for i, v in zip(ci.concept_id, ci.vector_id)}
    return unit(np.array([w[idx[f"{cl}__{cid}"]] for cid in W.MOVE_ORDER[cl]]))


def moves(X, perm=None):
    """All i<j chords, in fixed canonical order. `perm` relabels which row of X plays
    each slot -- that is what the null destroys."""
    Y = X if perm is None else X[perm]
    return np.array([Y[j] - Y[i] for i, j in combinations(range(len(X)), 2)])


def rsa(A, B, kind):
    n = len(A); rng = np.random.default_rng(SEED)
    da = pdist(A if kind == "positions" else moves(A), "cosine")
    f = (lambda p: pdist(B[p], "cosine")) if kind == "positions" else \
        (lambda p: pdist(moves(B, p), "cosine"))
    obs = spearmanr(da, f(np.arange(n))).correlation
    null = np.array([spearmanr(da, f(rng.permutation(n))).correlation for _ in range(NPERM)])
    return dict(n_items=n, n_pairs=len(da), rho=round(float(obs), 4),
                null_mean=round(float(null.mean()), 4),
                null_p95=round(float(np.percentile(null, 95)), 4),
                z=round(float((obs - null.mean()) / (null.std() + 1e-12)), 2),
                p=round(float((1 + (null >= obs).sum()) / (NPERM + 1)), 5))


def direction(A, B):
    """Do corresponding moves POINT the same way? The paper reports this separately and
    does not expect it: distant families sit far apart (water/temp mean cos +0.08)."""
    ma, mb = unit(moves(A)), unit(moves(B))
    return round(float((ma * mb).sum(1).mean()), 3)
