"""direction_stats.py -- the "Direction" block of Table 2, for the analogies (paper Table 8).

Mirrors the temperature direction statistics (analysis/temperature/moves_rsa.py), so the analogy
numbers can sit beside the temperature ones like for like:

  cross_cos    mean cos(a-move, its b-move under T)      do corresponding moves point the same way?
  within_A/B   mean pairwise cos among one family's moves   is there one common axis inside a family?
  raw_A/B      mean pairwise cos among the raw positions    the reference row: what differencing removes
  top_pc_A/B   share of the moves' variance on the leading direction

plus one column the temperature table does not carry:

  cross_null   the same cross cos under the permutation null analogy_rsa.py uses for the RSA
               (relabel B, rebuild B's moves over the same A pairs), so "near-orthogonal" can be read
               against what an arbitrary pairing gives rather than against zero.

Orientation. Temperature orients every move cold -> hot, so its signs mean "agree on warmer". Here a
move runs low-index -> high-index in relgeom.analogies.MOVE_ORDER on BOTH sides of T -- exactly the
construction in analogy_rsa.py. For size_imp_v1 that is small -> large, a real direction. For the
unordered families it is a fixed convention: cross_cos is unaffected (flipping a move flips its partner
too), but the SIGN of the within-family cos means nothing there, so the orientation-free |cos| is
reported alongside it.

Check: cross_cos must reproduce analogy_rsa.csv's mean_cos_corresponding_moves (the body's 0.07-0.25).

Writes: outputs/tables/[<tag>/]analogy_direction_stats.csv (needs analogy_rsa.py first, for the check)
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root
from relgeom import analogies as W, model_switch
from relgeom.paths import AXBENCH, TABLES
OUT = model_switch.out_dir(TABLES)
LAYERS = model_switch.layers([5, 10, 15, 20]); NPERM = 5000; SEED = 0
unit = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)


def cloud(ds, cl, L):
    """Unit-normalised vectors for one cloud, in canonical T order (as analogy_rsa.py)."""
    d = model_switch.vec_dir(ds, L)
    w = torch.load(d / "DiffMean_weight.pt", weights_only=False).float().numpy()
    ci = pd.read_parquet(AXBENCH / ds / "concept_index.parquet")
    idx = {v: i for i, v in zip(ci.concept_id, ci.vector_id)}
    return unit(np.array([w[idx[f"{cl}__{cid}"]] for cid in W.MOVE_ORDER[cl]]).astype(float))


def moves(X, perm=None):
    Y = X if perm is None else X[perm]
    return np.array([Y[j] - Y[i] for i, j in combinations(range(len(X)), 2)])


def mean_pairwise_cos(M):
    U = unit(M); C = U @ U.T; iu = np.triu_indices(len(U), 1)
    return float(C[iu].mean()), float(np.abs(C[iu]).mean())


def top_pc_frac(M):
    s = np.linalg.svd(unit(M), compute_uv=False)
    return float(s[0] ** 2 / (s ** 2).sum())


def cross(A, B, perm=None):
    return (unit(moves(A)) * unit(moves(B, perm))).sum(1)


def row(ds, a, b, L, A, B):
    n = len(A); c = cross(A, B)
    rng = np.random.default_rng(SEED)
    null = np.array([cross(A, B, rng.permutation(n)).mean() for _ in range(NPERM)])
    wa, wa_abs = mean_pairwise_cos(moves(A)); wb, wb_abs = mean_pairwise_cos(moves(B))
    ra, _ = mean_pairwise_cos(A); rb, _ = mean_pairwise_cos(B)
    return dict(experiment=ds, A=a, B=b, layer=L, n_items=n, n_moves=len(c),
                cross_cos=round(float(c.mean()), 4), cross_cos_sd=round(float(c.std()), 4),
                cross_cos_abs=round(float(np.abs(c).mean()), 4),
                pct_positive=round(100 * float((c > 0).mean()), 1),
                cross_null_mean=round(float(null.mean()), 4),
                cross_null_p95=round(float(np.quantile(null, .95)), 4),
                cross_p=round(float((1 + (null >= c.mean()).sum()) / (NPERM + 1)), 5),
                within_A=round(wa, 4), within_A_abs=round(wa_abs, 4),
                within_B=round(wb, 4), within_B_abs=round(wb_abs, 4),
                top_pc_A=round(top_pc_frac(moves(A)), 4), top_pc_B=round(top_pc_frac(moves(B)), 4),
                raw_A=round(ra, 4), raw_B=round(rb, 4))


PAIRED = [("size_imp_v1", "size", "importance"),
          ("war_arg_v1", "war", "argument"),
          ("food_ideas_v1", "food", "ideas")]
rows = []
for L in LAYERS:
    for ds, a, b in PAIRED:
        rows.append(row(ds, a, b, L, cloud(ds, a, L), cloud(ds, b, L)))
    X = {f: cloud("frames_v1", f, L) for f in W.F1_FRAMES}
    for f1, f2 in combinations(W.F1_FRAMES, 2):
        rows.append(row("frames_v1", f1, f2, L, X[f1], X[f2]))
    print(f"layer {L} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "analogy_direction_stats.csv", index=False)

# ---- check against the number already in the paper -------------------------------------------
ref = pd.read_csv(OUT / "analogy_rsa.csv")
ref = ref[ref.kind == "moves"][["experiment", "A", "B", "layer", "mean_cos_corresponding_moves"]]
m = df.merge(ref, on=["experiment", "A", "B", "layer"])
gap = (m.cross_cos.round(3) - m.mean_cos_corresponding_moves).abs().max()
print(f"\nCHECK vs analogy_rsa.csv mean_cos_corresponding_moves: {len(m)} rows, max |diff| = {gap:.4f}")

pd.set_option("display.width", 230)
cols = ["cross_cos", "cross_null_mean", "cross_null_p95", "cross_p", "pct_positive",
        "within_A", "within_A_abs", "within_B", "within_B_abs", "raw_A", "raw_B"]
print("\n=== three metaphors ===")
print(df[df.experiment != "frames_v1"].set_index(["experiment", "layer"])[cols].to_string())
fr = df[df.experiment == "frames_v1"]
print("\n=== professional frames, mean over the 10 profession pairs ===")
print(fr.groupby("layer")[cols].mean().round(4).to_string())
print("\n=== frames, cross_cos per pair ===")
print(fr.pivot_table(index=["A", "B"], columns="layer", values="cross_cos").round(3).to_string())
print(f"\nwrote {OUT}/analogy_direction_stats.csv")
