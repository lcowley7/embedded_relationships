"""analysis_harness.py — shared loader + toolkit bridge for the real-data geometry analysis.

This is the foundation for the `fractal_analysis/` notebooks (N1 curvature … N5 pinning).
It reproduces the fractal_toy_model's analysis on our REAL Gemma-2-2B DiffMean direction
manifolds. The toy's instruments are pure (N, D) -> ... functions that import nothing from
the toy, so we put that directory on the path and call them directly:

    import analysis_harness as H
    M, S, L = H.M, H.S, H.L            # metrics / spectral / laplacian (the toolkit)
    cl = H.load_cloud("temp_c1", 20)   # -> dict(X, order, labels, ...)

WHAT THE CLOUD IS. Each "manifold" here is the DiffMean *direction* manifold: the steering
vector r(level) as the concept's level varies (temperature, frequency, ...). Rows are
L2-normalised, so the cloud lives on the unit sphere in R^2304. N = number of levels
(small: 4-131), D = 2304. This is the right object for "temperature is a curve, not a line".

LINEARITY POLICY (standing rule). These manifolds are curved. Lead with curve-aware /
gauge-free distance: geodesic (Isomap-graph) and diffusion. Raw Euclidean is a CHORD that
under-measures distance along a curve and assumes flatness — use it ONLY as a flagged
baseline, never as the headline. Helpers below expose all three so a notebook can show the
contrast explicitly.

cwd gotcha: a Jupyter kernel's cwd is the notebook's own folder, so everything here uses
absolute REPO paths.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths (relative to the repository root; run scripts from there)
# --------------------------------------------------------------------------- #
REPO = "."
TOOLKIT_DIR = os.path.join(REPO, "toy_model/fractal_toy_model")
DATA_BASE = os.path.join(REPO, "data/axbench")
LAYERS = [5, 10, 15, 20]

# import the toy's generic instruments (they import nothing from the toy model itself)
if TOOLKIT_DIR not in sys.path:
    sys.path.insert(0, TOOLKIT_DIR)
# The geometry toolkit (metrics/spectral/laplacian) is NOT part of this release and is not used by
# any analysis in the paper; the helpers below that call it (curvature, diffusion distance) are unused.
try:
    import metrics as M       # noqa: E402
    import spectral as S      # noqa: E402
    import laplacian as L     # noqa: E402
except ImportError:
    M = S = L = None

# --------------------------------------------------------------------------- #
# Domain knowledge (from build_review_notebook.py; verified by the data map)
# --------------------------------------------------------------------------- #
# Canonical ordinal ladders for the lexical "scale" families.
LEVEL_ORDER = {
    "frequency": ["never", "rarely", "sometimes", "often", "usually", "always"],
    "modal": ["cant", "might", "may", "should", "must"],
    "quantifier": ["none", "few", "some", "several", "many", "most", "all"],
    "wep": ["almost_no_chance", "very_unlikely", "unlikely", "roughly_even",
            "likely", "very_likely", "almost_certain"],
    "logic": ["forall", "exists", "not_all", "none"],  # NOT a total order (operators)
}

# h2o substances -> (phase group, NOMINAL temperature in C). Temps are ranking-only.
H2O_TEMP = {
    "ice": -10, "snow": -5, "hail": -5, "frost": -3, "sleet": 0, "slush": 1,
    "icy_water": 1, "cold_water": 8, "rain": 10, "dew": 10, "fog": 10, "puddle": 12,
    "cool_water": 15, "roomtemp_water": 21, "warm_water": 35, "hot_water": 55,
    "boiling_water": 100, "steam": 105,
}
H2O_GROUP = {
    "ice": "frozen", "snow": "frozen", "hail": "frozen", "frost": "frozen",
    "sleet": "frozen", "slush": "frozen",
    "rain": "liquid", "dew": "liquid", "fog": "liquid", "puddle": "liquid",
    "steam": "gas",
    "icy_water": "adj_water", "cold_water": "adj_water", "cool_water": "adj_water",
    "roomtemp_water": "adj_water", "warm_water": "adj_water", "hot_water": "adj_water",
    "boiling_water": "adj_water",
}
GROUP_COLOR = {"frozen": "#3b6fb6", "liquid": "#4fa36b", "gas": "#cc4444",
               "adj_water": "#b07cc6"}

# Approx percentage anchors so frequency / wep / probability can share a % axis.
PERCENT_ANCHOR = {
    "frequency": {"never": 0, "rarely": 7, "sometimes": 50, "often": 75,
                  "usually": 90, "always": 100},
    "wep": {"almost_no_chance": 3, "very_unlikely": 12, "unlikely": 32,
            "roughly_even": 50, "likely": 67, "very_likely": 87, "almost_certain": 97},
}

# Which family (LEVEL_ORDER key) each lexical dataset belongs to.
FAMILY_OF = {
    "frequency_v2": "frequency", "modal_v1": "modal", "quantifier_v1": "quantifier",
    "wep_v1": "wep", "logic_word_v1": "logic", "logic_sym_v1": "logic",
}

# Registry: family + a short description. `kind` flags whether a native numeric coord exists.
DATASETS = {
    # temperature family
    "temp_c1": dict(family="temperature", kind="numeric", desc="1C sweep -20..110C (131)"),
    "temp_c5": dict(family="temperature", kind="numeric", desc="5C ladder (27)"),
    "temp_f5": dict(family="temperature", kind="numeric", desc="5C-equiv in F (27 twins)"),
    "h2o_v1":  dict(family="temperature", kind="nominal", desc="18 water substances"),
    # frequency / scale family (ordered ladder = the steer sub-cloud)
    "frequency_v2": dict(family="frequency", kind="ordinal", desc="never..always (6)"),
    "modal_v1": dict(family="modal", kind="ordinal", desc="cant..must (5)"),
    "quantifier_v1": dict(family="quantifier", kind="ordinal", desc="none..all (7)"),
    "wep_v1": dict(family="wep", kind="ordinal", desc="almost_no_chance..almost_certain (7)"),
    "probability_steer": dict(family="probability", kind="numeric", desc="0..100% (21)"),
    "logic_word_v1": dict(family="logic", kind="unordered", desc="forall/exists/not_all/none"),
    "logic_sym_v1": dict(family="logic", kind="unordered", desc="logic symbols (4)"),
}


def slug_to_temp(slug: str):
    """vector_id slug -> signed number. 'neg20'->-20, '12'->12. None if not numeric."""
    s = str(slug)
    try:
        return -int(s[3:]) if s.startswith("neg") else int(s)
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------- #
# Weight / cloud loading
# --------------------------------------------------------------------------- #
def _weight_path(dataset, layer, data_base=None):
    # LT_MODEL switch (lt_model.py): redirects only the WEIGHTS to another model's
    # vectors; concept indices and every other convention stay canonical. Unset -> unchanged.
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    import lt_model as LT
    return os.path.join(data_base or LT.weight_base() or DATA_BASE, dataset,
                        f"diffmean_l{layer}", LT.weight_subdir(), "DiffMean_weight.pt")


def load_weight(dataset, layer, data_base=None):
    """Load DiffMean_weight.pt as float32 numpy (N, D). Rows are concept_id-ordered.

    data_base : override the tree the WEIGHTS come from, leaving every other convention
                (concept_index, ordering, labels) untouched. Used by the random-model null
                (random_null/), whose vectors live under data/random_null/<run>/
                but which must be read with exactly the real run's correspondences.
    """
    import torch
    w = torch.load(_weight_path(dataset, layer, data_base), map_location="cpu")
    return w.float().numpy()


def _concept_index(dataset):
    p = os.path.join(DATA_BASE, dataset, "concept_index.parquet")
    return pd.read_parquet(p)


def _order_array(dataset, df):
    """Intrinsic ordering coordinate for each row of df (concept_id order)."""
    name = dataset
    if name == "temp_c1" and "temp_c" in df.columns:
        return df["temp_c"].astype(float).values, "temp_C"
    if name in ("temp_c5", "temp_f5"):
        vals = np.array([slug_to_temp(str(v).split("__")[-1]) for v in df["vector_id"]],
                        dtype=float)
        return vals, ("temp_C" if name == "temp_c5" else "temp_F")
    if name == "h2o_v1":
        return (np.array([H2O_TEMP.get(g, np.nan) for g in df["concept_group"]], float),
                "nominal_temp_C")
    if name == "probability_steer":
        vals = []
        for g in df["concept_group"]:
            try:
                vals.append(float(g))
            except (ValueError, TypeError):
                vals.append(slug_to_temp(str(df.loc[df["concept_group"] == g, "vector_id"].iloc[0]).split("__")[-1]))
        return np.array(vals, float), "percent"
    fam = FAMILY_OF.get(name)
    if fam and fam in LEVEL_ORDER:
        lo = LEVEL_ORDER[fam]
        idx = []
        for g in df["concept_group"]:
            g = str(g)
            if g in lo:
                idx.append(lo.index(g))
            else:  # defensive: match the level token that the group contains
                hit = [i for i, t in enumerate(lo) if t in g or g in t]
                idx.append(hit[0] if hit else np.nan)
        return np.array(idx, float), f"{fam}_rank"
    return None, None


def load_cloud(dataset, layer, perspective="steer", sort=True, normalize=False, data_base=None):
    """Load one concept manifold as a (N, D) cloud + its intrinsic ordering.

    perspective : for lexical families, 'steer' = the clean ladder; 'magnitude' / 'anti_steer'
                  select the transition / anchor sub-manifolds. Ignored if the dataset has no
                  perspective column (temperature dirs are all steer).
    sort        : order rows by the intrinsic coordinate (so X[i] is the i-th level).
    normalize   : re-L2-normalise rows (they already are; default off).
    data_base   : read the WEIGHTS from an alternative tree (the concept_index, ordering
                  and labels still come from the real dataset). This is how the
                  random-model null reuses every correspondence unchanged; see
                  random_null/METHOD.md.

    Returns dict: X (N,D float32), order (N,), order_name, labels (level names), vector_ids,
    groups (h2o phase group or None), name, layer, df (the selected concept_index rows).
    """
    df = _concept_index(dataset).reset_index(drop=True)
    W = load_weight(dataset, layer, data_base)
    if len(df) != len(W):
        raise ValueError(f"{dataset} L{layer}: {len(df)} index rows vs {len(W)} weight rows")
    df["_row"] = np.arange(len(df))
    if perspective is not None and "perspective" in df.columns:
        sel = df["perspective"].astype(str) == perspective
        if sel.any():
            df = df[sel].reset_index(drop=True)
    rows = df["_row"].values
    X = W[rows]
    order, order_name = _order_array(dataset, df)
    if sort and order is not None and np.isfinite(order).all():
        si = np.argsort(order, kind="stable")
        df = df.iloc[si].reset_index(drop=True)
        X = X[si]
        order = order[si]
    if normalize:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    groups = (np.array([H2O_GROUP.get(g) for g in df["concept_group"]])
              if dataset == "h2o_v1" else None)
    return dict(
        X=np.ascontiguousarray(X, dtype=np.float32),
        order=order, order_name=order_name,
        labels=df["concept_group"].astype(str).values,
        vector_ids=df["vector_id"].astype(str).values,
        groups=groups, name=dataset, layer=layer, df=df,
    )


# --------------------------------------------------------------------------- #
# Cross-manifold correspondences
# --------------------------------------------------------------------------- #
def c5_f5_twins(layer):
    """temp_c5 and temp_f5 aligned row-by-row to the SAME physical temperature.
    Returns (Xc, Xf, temp_C)."""
    c = load_cloud("temp_c5", layer)
    f = load_cloud("temp_f5", layer)
    tC = c["order"]
    tC_from_f = (f["order"] - 32.0) * 5.0 / 9.0
    # both sorted ascending by their own scale -> same physical order
    assert np.allclose(tC, tC_from_f, atol=0.5), "c5/f5 twin temperatures do not match"
    return c["X"], f["X"], tC


def h2o_on_temp_axis(layer):
    """h2o cloud + its nominal temperature + phase group, for projection-onto-temperature
    tests. Returns dict(X, temp, group, labels)."""
    c = load_cloud("h2o_v1", layer)
    return dict(X=c["X"], temp=c["order"], group=c["groups"], labels=c["labels"])


def percent_axis_clouds(layer):
    """frequency / wep / probability on a shared 0-100% axis (approx anchors).
    Returns {family: dict(X, pct, labels)}."""
    out = {}
    for ds, fam in [("frequency_v2", "frequency"), ("wep_v1", "wep")]:
        cl = load_cloud(ds, layer)
        pct = np.array([PERCENT_ANCHOR[fam].get(str(g), np.nan) for g in cl["labels"]], float)
        out[fam] = dict(X=cl["X"], pct=pct, labels=cl["labels"])
    pr = load_cloud("probability_steer", layer)
    out["probability"] = dict(X=pr["X"], pct=pr["order"], labels=pr["labels"])
    return out


# --------------------------------------------------------------------------- #
# Distances — curve-aware first; Euclidean is a flagged baseline
# --------------------------------------------------------------------------- #
def euclidean_distance(X):
    """Chord distance (FLAT assumption). Baseline only — under-measures along a curve."""
    return M.euclidean_distance(X)


def connected_knn_k(X, k0=None, kmax=None):
    """Smallest kNN k that gives a SINGLE connected component, so Isomap/geodesic don't
    fragment. Our manifolds are smooth narrow-cone arcs (adjacent cosines ~0.996), so the
    graph needs more neighbours than intuition suggests — the toy warned: too-small k
    short-circuits/fragments and corrupts the 1-D order. Use this everywhere kNN is built."""
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import connected_components
    n = len(X)
    if k0 is None:
        k0 = int(min(max(10, n // 8), n - 1))
    if kmax is None:
        kmax = n - 1
    for k in range(max(2, k0), kmax + 1):
        G = kneighbors_graph(X, k, mode="distance", include_self=False)
        if connected_components(G, directed=False)[0] == 1:
            return k
    return n - 1


def geodesic_distance(X, k=None):
    """Geodesic (shortest-path on a kNN graph) distance — respects manifold curvature.
    Returns (D, k_used). k defaults to the smallest connected k (connected_knn_k)."""
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import shortest_path
    if k is None:
        k = connected_knn_k(X)
    G = kneighbors_graph(X, k, mode="distance", include_self=False)
    D = shortest_path(G, method="D", directed=False)
    return D, k


def diffusion_distance(X, t=1, **kw):
    """Diffusion distance (averages over all paths; robust, curvature-aware)."""
    return M.diffusion_distance(X, t=t, **kw)


def signature(D):
    """Gauge-free fingerprint (mean-normalised upper triangle) for RSA."""
    return M.distance_signature(D)


def rsa(sig_a, sig_b, method="spearman"):
    return M.rsa(sig_a, sig_b, method=method)


def ball_null(n, d=2304, seed=0):
    """N random unit vectors in R^d — the ball-packing / superposition null."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


def smooth_arc(n, d=2304, n_modes=2, amp_decay=1.2, seed=0):
    """A smooth OPEN arc on the unit sphere in R^d — intrinsically 1-D, embedded via a few
    low-frequency modes (dominant f=1 axis + gently-decaying higher modes), each point
    renormalised to the sphere. n_modes=1 -> a great-circle arc; 2-3 -> a gently curved arc.
    The deterministic backbone for the 'LayerNorm sphere + noise ball' baseline."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    U = rng.normal(size=(2 * n_modes, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    X = np.zeros((n, d))
    for f in range(1, n_modes + 1):
        a = 1.0 / (f ** amp_decay)
        X += a * np.cos(np.pi * f * t)[:, None] * U[2 * f - 2][None, :]
        X += a * np.sin(np.pi * f * t)[:, None] * U[2 * f - 1][None, :]
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


def noisy_sphere(X, noise, seed=0):
    """Add isotropic Gaussian noise of scale `noise`, then renormalise to the unit sphere
    (the 'noise ball at each point' + LayerNorm constraint)."""
    rng = np.random.default_rng(seed)
    Y = np.asarray(X, float) + noise * rng.normal(size=np.shape(X))
    return (Y / np.linalg.norm(Y, axis=1, keepdims=True)).astype(np.float32)


def pca_coords(X, n=2):
    """Top-n PCA coordinates. NOTE: LINEAR projection — flattens curvature; for
    visualisation/baseline only. Use Isomap/diffusion coords for the faithful low-D view."""
    Xc = np.asarray(X, float) - np.asarray(X, float).mean(0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (U[:, :n] * s[:n]), (s ** 2 / (s ** 2).sum())


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def selftest():
    print(f"toolkit @ {TOOLKIT_DIR}")
    print(f"  metrics={M.__file__ is not None} spectral={S.__file__ is not None} laplacian={L.__file__ is not None}")
    print(f"data    @ {DATA_BASE}\n")
    for ds, meta in DATASETS.items():
        try:
            cl = load_cloud(ds, 20)
        except Exception as e:  # noqa: BLE001
            print(f"  !! {ds}: {type(e).__name__}: {e}")
            continue
        X = cl["X"]
        nrm = np.linalg.norm(X, axis=1)
        o = cl["order"]
        if o is not None and np.isfinite(o).all():
            ostr = f"{cl['order_name']} {o.min():.0f}..{o.max():.0f}"
        else:
            ostr = f"labels={list(cl['labels'])}"
        # shape consistency across layers
        shapes = {ly: load_weight(ds, ly).shape for ly in LAYERS}
        ok = len({s[0] for s in shapes.values()}) == 1
        print(f"  {ds:16s} N={X.shape[0]:3d} D={X.shape[1]} |row|~{nrm.mean():.3f} "
              f"layers_ok={ok} | {ostr}")
    print("\ncorrespondences:")
    try:
        Xc, Xf, tC = c5_f5_twins(20)
        print(f"  c5/f5 twins: {len(tC)} temps {tC.min():.0f}..{tC.max():.0f}C, shapes {Xc.shape}/{Xf.shape}")
    except Exception as e:  # noqa: BLE001
        print(f"  !! c5_f5_twins: {type(e).__name__}: {e}")
    try:
        h = h2o_on_temp_axis(20)
        print(f"  h2o: {len(h['temp'])} substances, temp {np.nanmin(h['temp']):.0f}..{np.nanmax(h['temp']):.0f}C, "
              f"groups={sorted(set(h['group']))}")
    except Exception as e:  # noqa: BLE001
        print(f"  !! h2o_on_temp_axis: {type(e).__name__}: {e}")
    try:
        pax = percent_axis_clouds(20)
        for fam, d in pax.items():
            print(f"  %-axis {fam:11s}: N={len(d['pct'])} pct {np.nanmin(d['pct']):.0f}..{np.nanmax(d['pct']):.0f}")
    except Exception as e:  # noqa: BLE001
        print(f"  !! percent_axis_clouds: {type(e).__name__}: {e}")
    # one curvature-aware smoke check on temp_c1
    try:
        cl = load_cloud("temp_c1", 20)
        kappa = M.total_curvature(cl["X"], np.arange(len(cl["X"])))
        Dg, conn = geodesic_distance(cl["X"])
        print(f"\nsmoke: temp_c1 total_curvature={kappa:.1f}, geodesic connected={conn}")
    except Exception as e:  # noqa: BLE001
        print(f"\n!! smoke: {type(e).__name__}: {e}")


if __name__ == "__main__":
    selftest()
