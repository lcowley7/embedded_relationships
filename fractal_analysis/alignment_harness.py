"""alignment_harness.py — E0 foundations for the "shape predicted by cross-manifold alignment" track.

Builds on analysis_harness (H). Everything here exists to make ONE claim testable without fooling
ourselves: *part of a concept manifold's shape is forced by its members having to register at the
right places on OTHER manifolds.*  Three standing hazards, three tools:

  (1) SHARED SCAFFOLD.  Every steering vector loads ~+0.63 on a common residual-stream direction, so
      any two sit at cosine ~+0.36 for free (independence_vs_correlation).  -> `deflate` removes the
      single common direction (verified safe: it costs <3% of e_even/e_odd norm; stripping the full
      16-D scaffold over-strips, removing ~half — so default k=1).  Work gauge-free downstream.
  (2) MEASUREMENT JITTER.  DiffMean vectors are noisy estimates.  -> `reliability_ceiling` reports the
      reproducible-variance fraction of a manifold (≈0.85 on the dense sweeps); predictions are scored
      AGAINST that ceiling, never against 1.0.  Local speed/curvature is below the floor for N<=30, so
      use GLOBAL shape (RSA / Procrustes) at low N, curvature only on the dense sweeps.
  (3) GENERIC SHAPE.  A bounded ordinal concept is intrinsically a curved arc with an extremity fold;
      so is every null.  -> alignment must beat a `shuffled_correspondence` (common-cause) null and,
      where relevant, H.smooth_arc / H.ball_null.  The signal is DISPLACED structure (geometry where a
      neighbour's ground truth puts it), not "is curved".

cwd gotcha: absolute REPO paths throughout (Jupyter kernel cwd = the notebook's folder).
"""
from __future__ import annotations
import sys, numpy as np, pandas as pd
REPO = "."
HERE = REPO + "/fractal_analysis"
if HERE not in sys.path: sys.path.insert(0, HERE)
import analysis_harness as H
M = H.M
from scipy.stats import spearmanr

# --------------------------------------------------------------------------- #
# loading (registry + the off-registry %/shard sweeps)
# --------------------------------------------------------------------------- #
def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

def load_steer(ds, layer):
    """Unit-row steer cloud + intrinsic coordinate, for registry OR off-registry sweeps
    (probability_c1, prob_c1a-d). Returns dict(X, order, labels, name)."""
    if ds in H.DATASETS:
        cl = H.load_cloud(ds, layer)
        return dict(X=unit(cl["X"].astype(float)), order=cl["order"],
                    labels=cl["labels"], name=ds)
    W = H.load_weight(ds, layer).astype(float)
    ci = pd.read_parquet(f"{H.DATA_BASE}/{ds}/concept_index.parquet")
    if "pct" in ci.columns:
        coord = ci["pct"].values.astype(float)
    elif "perspective" in ci.columns and (ci["perspective"].astype(str) == "steer").any():
        sel = ci["perspective"].astype(str) == "steer"
        ci = ci[sel].reset_index(); W = W[ci["index"].values]
        coord = ci["concept_group"].astype(float).values
    else:
        coord = ci["concept_group"].astype(float).values
    si = np.argsort(coord)
    return dict(X=unit(W[si]), order=coord[si],
                labels=ci["concept_group"].astype(str).values[si], name=ds)

# --------------------------------------------------------------------------- #
# (1) shared-scaffold removal
# --------------------------------------------------------------------------- #
_SCAFFOLD = {}
_SCAFFOLD_POOL = ["temp_c1", "probability_c1", "frequency_v2", "wep_v1", "modal_v1",
                  "quantifier_v1", "logic_word_v1", "logic_sym_v1", "h2o_v1", "temp_c5", "temp_f5"]

def scaffold(layer, k=1):
    """Top-k common residual-stream directions from pooled per-level steer vectors. Cached.
    k=1 (default) = the single dominant common direction (the safe strip)."""
    key = (layer, k)
    if key in _SCAFFOLD: return _SCAFFOLD[key]
    pool = []
    for ds in _SCAFFOLD_POOL:
        try: pool.append(load_steer(ds, layer)["X"])
        except Exception: pass
    P = np.vstack(pool)
    if k == 1:
        B = unit(P.mean(0))[None, :]
    else:
        _, _, Vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
        B = unit(Vt[:k])
    _SCAFFOLD[key] = B
    return B

def deflate(X, layer, k=1):
    """Remove the shared scaffold span from rows of X (X is (N,D) or (D,))."""
    B = scaffold(layer, k); X = np.asarray(X, float)
    return X - (X @ B.T) @ B

# --------------------------------------------------------------------------- #
# (2) reliability ceiling
# --------------------------------------------------------------------------- #
def reliability_ceiling(ds, layer, smooth_rank=6):
    """Reproducible-variance fraction of a manifold. Two routes:
      lowrank: variance captured by the top-`smooth_rank` PCs (a smooth manifold is low-rank);
      diff   : 1 - successive-2nd-difference noise power / per-row power (upper bound on jitter).
    Returns dict(lowrank, diff, adjcos, N)."""
    cl = load_steer(ds, layer); X = cl["X"]; Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False); evr = (s**2 / (s**2).sum())
    d2 = Xc[:-2] - 2*Xc[1:-1] + Xc[2:]
    sig2 = (d2**2).mean() / 6.0
    per_row = (Xc**2).sum() / len(Xc)
    return dict(lowrank=float(evr[:smooth_rank].sum()),
                diff=float(1.0 - min(1.0, sig2 * X.shape[1] / per_row)),
                adjcos=float((X[:-1]*X[1:]).sum(1).mean()), N=len(X))

# --------------------------------------------------------------------------- #
# (3) gauge-free shape + Procrustes prediction + nulls
# --------------------------------------------------------------------------- #
def signature(X, metric="euclidean"):
    """Gauge-free distance fingerprint. metric: 'euclidean' (chord, flat baseline) or
    'geodesic' (curve-aware, connected-kNN)."""
    X = np.asarray(X, float)
    if metric == "geodesic":
        D, _ = H.geodesic_distance(X)
    else:
        D = H.euclidean_distance(X)
    return M.distance_signature(D)

def shape_rsa(Xa, Xb, metric="euclidean"):
    """Gauge-free RSA between two row-corresponded clouds (same N, row i <-> row i)."""
    return float(M.rsa(signature(Xa, metric), signature(Xb, metric)))

def shuffled_correspondence_null(Xa, Xb, metric="euclidean", n=500, seed=0):
    """RSA distribution when Xb's rows are randomly re-paired to Xa's = common-cause control."""
    rng = np.random.default_rng(seed); sa = signature(Xa, metric)
    null = [M.rsa(sa, signature(Xb[rng.permutation(len(Xb))], metric)) for _ in range(n)]
    return float(np.mean(null)), float(np.std(null)), np.array(null)

def fit_procrustes(A, B, scaling=True):
    """Orthogonal Procrustes: best rotation/scale mapping rows of A onto rows of B (centred).
    Returns a callable predict(A_new) and the train disparity (||pred-B||/||B||)."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    muA, muB = A.mean(0), B.mean(0); Ac, Bc = A - muA, B - muB
    U, s, Vt = np.linalg.svd(Ac.T @ Bc, full_matrices=False)
    R = U @ Vt
    scale = (s.sum() / (Ac**2).sum()) if scaling else 1.0
    def predict(A_new):
        return (np.asarray(A_new, float) - muA) @ R * scale + muB
    disp = float(np.linalg.norm(predict(A) - B) / (np.linalg.norm(Bc) + 1e-12))
    return predict, disp

def localization_percentile(pred_vec, candidates, true_idx):
    """Where does the TRUE candidate rank among all candidates by closeness to pred_vec?
    0 = predicted location nails the true point; 1 = worst. Gauge-free (cosine on deflated rows)."""
    pv = unit(pred_vec); sims = unit(candidates) @ pv
    rank = (sims > sims[true_idx]).sum()           # how many candidates are closer than the truth
    return float(rank / (len(candidates) - 1))

# --------------------------------------------------------------------------- #
def selftest():
    import numpy as np
    print("alignment_harness selftest")
    for layer in (10,):
        B = scaffold(layer, 1); print(f"  scaffold L{layer}: shape {B.shape}, |B|={np.linalg.norm(B):.3f}")
        for ds in ("temp_c1", "probability_c1"):
            r = reliability_ceiling(ds, layer)
            print(f"  reliability {ds:14s}: lowrank6={r['lowrank']:.2f} diff={r['diff']:.2f} N={r['N']} adjcos={r['adjcos']:.4f}")
    Xc5, Xf5, tC = H.c5_f5_twins(10)
    Xc5d, Xf5d = deflate(Xc5, 10), deflate(Xf5, 10)
    print(f"  twin shape RSA (deflated): euclid={shape_rsa(Xc5d, Xf5d):.2f}  geodesic={shape_rsa(Xc5d, Xf5d, 'geodesic'):.2f}")
    nm, ns, _ = shuffled_correspondence_null(Xc5d, Xf5d, n=200)
    print(f"  shuffled-correspondence null: {nm:+.2f} +/- {ns:.2f}")
    pred, disp = fit_procrustes(Xc5d, Xf5d)
    print(f"  Procrustes C5->F5 train disparity: {disp:.3f}  (0=perfect, 1=no better than mean)")

if __name__ == "__main__":
    selftest()
