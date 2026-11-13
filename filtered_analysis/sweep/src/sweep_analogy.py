"""sweep_analogy.py -- the analogy RSA (Section 'Analogies', Figure analogy_rsa) as a function of
the drop fraction f, at every trained layer (5, 10, 15, 20).

Items are the PAIRS of the correspondence T, not single vectors. A pair enters the source RDM
through its source vector and the target RDM through its target vector, so it is only as
trustworthy as the weaker of the two and is scored exactly as the temperature sweep scores a
water word with its numeric level:
    s_i = min( Overall(source_i), Overall(target_i) ).
At each f we drop the floor(f*n) lowest-scoring pairs (ties broken by position in T, so the
sequence is deterministic and nested), and recompute positions and moves RSA on the survivors,
with the correspondence-permutation null redrawn over the surviving pairs. Survivors keep their
canonical T order, so f = 0 IS the published analysis; the script checks that it reproduces
tables/wave1_rsa.csv on every row.

Frames: each of the 10 profession pairs is its own analysis over the 6 roles, and a role is
scored by the minimum over its two professions. With 6 items, f = 0.4 and 0.5 leave 4 and 3
roles (6 and 3 RDM entries): computed, but flagged not reportable (MIN_ITEMS).

cloud / moves / rsa are copied verbatim from wave1_rsa.py rather than imported, because that
module runs its whole analysis (and rewrites its tables) at import time.

With LT_MODEL set the vectors, judge scores, layers and outputs are that model's
(lt_model.py): the sweep asks of each model whether ITS geometry depends on ITS
own weak-steering tail. Layers with no steering evaluation are skipped, and the f = 0 check is
made against that model's own tables/<tag>/wave1_rsa.csv.

Run: PY filtered_analysis/sweep/src/sweep_analogy.py
Needs manifests[/<tag>]/judge_scores.csv with the wave-1 families (build_scores.py).
"""
import sys, math, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from itertools import combinations
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

REPO = "."
EXT = f"{REPO}/fractal_analysis/extension"
BASE = f"{REPO}/filtered_analysis/sweep"
sys.path.insert(0, EXT)
sys.path.insert(0, f"{REPO}")
import wave1_families as W   # noqa: E402
import lt_model as LT        # noqa: E402

NPERM, SEED = 5000, 0
FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
MIN_ITEMS = 5          # 4 items -> 6 RDM entries; Spearman on that is unconstrained
METAPHORS = ["size_imp_v1", "war_arg_v1", "food_ideas_v1"]
unit = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)


# ------------------------------------------------- verbatim from wave1_rsa.py --------------
def cloud(ds, cl, L):
    """Unit-normalised vectors for one cloud, in canonical T order."""
    d = LT.vec_dir(ds, L)
    w = torch.load(f"{d}/DiffMean_weight.pt", weights_only=False).float().numpy()
    ci = pd.read_parquet(f"{REPO}/data/axbench/{ds}/concept_index.parquet")
    idx = {v: i for i, v in zip(ci.concept_id, ci.vector_id)}
    return unit(np.array([w[idx[f"{cl}__{cid}"]] for cid in W.MOVE_ORDER[cl]]))


def moves(X, perm=None):
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
# --------------------------------------------------------------------------------------------


MANIFESTS, TABLES = str(LT.out_dir(f"{BASE}/manifests")), str(LT.out_dir(f"{BASE}/tables"))
sc = pd.read_csv(f"{MANIFESTS}/judge_scores.csv")
# only layers this model was steered at can be swept
LAYERS = [L for L in LT.layers([5, 10, 15, 20])
          if set(sc[sc.layer == L].family) >= {"size_imp_v1", "war_arg_v1", "food_ideas_v1", "frames_v1"}]
assert LAYERS, f"no layer has wave-1 judge scores in {MANIFESTS}/judge_scores.csv"
print(f"{LT.label()}: sweeping layers {LAYERS}", flush=True)


def scores(ds, L):
    s = sc[(sc.family == ds) & (sc.layer == L)]
    n = sum(len(W.CLOUDS[c]) for c in W.EXPERIMENT_CLOUDS[ds])
    assert len(s) == n and s.label.is_unique, f"{ds} L{L}: {len(s)} scores for {n} vectors"
    return dict(zip(s.label, s.overall))            # label = vector_id


def sweep_pair(A, B, ids_a, ids_b, s, meta):
    """A, B: unit rows in canonical T order (row i of A pairs with row i of B)."""
    n = len(A)
    ps = np.array([min(s[x], s[y]) for x, y in zip(ids_a, ids_b)])
    order = sorted(range(n), key=lambda i: (ps[i], i))
    out = []
    for f in FRACS:
        k = int(np.floor(f * n + 1e-9))
        drop = set(order[:k])
        keep = [i for i in range(n) if i not in drop]           # canonical order preserved
        for kind in ("positions", "moves"):
            out.append(dict(**meta, f=f, kind=kind, **rsa(A[keep], B[keep], kind),
                            n_relabelings=math.factorial(len(keep)),
                            reportable=len(keep) >= MIN_ITEMS,
                            min_score_kept=round(float(ps[keep].min()), 3),
                            dropped=";".join(f"{ids_a[i]}|{ids_b[i]}" for i in sorted(drop))))
    return out


rows = []
for ds in METAPHORS:
    a, b = W.EXPERIMENTS[ds]["A"], W.EXPERIMENTS[ds]["B"]
    assert list(zip(W.MOVE_ORDER[a], W.MOVE_ORDER[b])) == [tuple(t) for t in W.EXPERIMENTS[ds]["T"]]
    ids_a = [f"{a}__{c}" for c in W.MOVE_ORDER[a]]
    ids_b = [f"{b}__{c}" for c in W.MOVE_ORDER[b]]
    for L in LAYERS:
        rows += sweep_pair(cloud(ds, a, L), cloud(ds, b, L), ids_a, ids_b, scores(ds, L),
                           dict(experiment=ds, A=a, B=b, layer=L))
        print(f"  {ds} L{L} done", flush=True)

FR = W.F1_FRAMES
roles = [r["role"] for r in W.CLOUDS[FR[0]]]
assert all([r["role"] for r in W.CLOUDS[f]] == roles for f in FR)
for L in LAYERS:
    s = scores("frames_v1", L)
    X = {f: cloud("frames_v1", f, L) for f in FR}
    for f1, f2 in combinations(FR, 2):
        rows += sweep_pair(X[f1], X[f2], [f"{f1}__{c}" for c in W.MOVE_ORDER[f1]],
                           [f"{f2}__{c}" for c in W.MOVE_ORDER[f2]], s,
                           dict(experiment="frames_v1", A=f1, B=f2, layer=L))
    print(f"  frames_v1 L{L} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{TABLES}/analogy_sweep.csv", index=False)

fr = df[df.experiment == "frames_v1"]
frs = (fr.groupby(["layer", "f", "kind"])
         .agg(mean_rho=("rho", "mean"), n_sig=("p", lambda v: int((v < .05).sum())),
              n_items=("n_items", "first"), reportable=("reportable", "first"))
         .reset_index())
frs.to_csv(f"{TABLES}/analogy_sweep_frames.csv", index=False)

# ---- the wiring check: f = 0 must be the published analysis, row for row -------------------
pub = pd.read_csv(f"{LT.out_dir(f'{EXT}/tables')}/wave1_rsa.csv")
pub = pub[pub.layer.isin(LAYERS)].reset_index(drop=True)
key = ["experiment", "A", "B", "layer", "kind"]
m = df[df.f == 0].merge(pub, on=key, how="outer", suffixes=("", "_pub"), indicator=True)
assert (m._merge == "both").all() and len(m) == len(pub), "f=0 rows do not match wave1_rsa.csv"
cols = ["n_items", "n_pairs", "rho", "null_mean", "null_p95", "z", "p"]
worst = {c: float((m[c] - m[f"{c}_pub"]).abs().max()) for c in cols}
print("\nf=0 vs tables/wave1_rsa.csv, max |difference| per column over", len(m), "rows:", worst)
assert all(v == 0 for v in worst.values()), "f=0 does NOT reproduce the published analysis"
print("OK: f=0 reproduces the published analogy RSA exactly")
print(f"wrote {TABLES}/analogy_sweep.csv and analogy_sweep_frames.csv")
