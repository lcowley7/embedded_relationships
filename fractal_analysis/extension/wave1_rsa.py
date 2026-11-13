"""wave1_rsa.py -- the paper's section-8 analysis, run on the wave-1 families.

Positions and moves, both reported (plan v1), against the label-permutation null that
reshuffles the correspondence T -- exactly make_move_rsa_tables.py / make_temp_rsa_appendix.py,
just with T supplied by wave1_families instead of by a water word's nominal temperature.

  PM_A(i,j) = 1 - cos(a_i, a_j)                  within family A only
  PM_B(i,j) = 1 - cos(b_T(i), b_T(j))            within family B only
  positions rho = spearman(pdist(A), pdist(B))

  moves: m_{i->j} = a_j - a_i for every i<j in canonical order; the counterpart in B is
  b_T(j) - b_T(i). Move RSA correlates the two move-by-move distance matrices. The null
  permutes T and REBUILDS B's moves over the same A pairs, so it is centred well above 0
  (two moves sharing an endpoint are similar whatever the labelling) -- read rho against
  the null, never against zero.

Also runs the generation-artefact check the workshop notebook could only do at smoke scale:
if cloud A's co-mention matrix rank-correlates with cloud B's UNDER T, then part of any
agreement was written into the training text rather than found in the model.

Writes tables/ CSVs + a printed summary.
Run: PY fractal_analysis/extension/wave1_rsa.py
"""
import json, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, re
from pathlib import Path
from itertools import combinations
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

REPO = Path(".")
EXT = REPO / "fractal_analysis/extension"
sys.path.insert(0, str(EXT)); import wave1_families as W
sys.path.insert(0, str(REPO)); import lt_model as LT   # LT_MODEL switch
OUT = LT.out_dir(EXT / "tables")
LAYERS = LT.layers([5, 10, 15, 20]); NPERM = 5000; SEED = 0
unit = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)


def cloud(ds, cl, L):
    """Unit-normalised vectors for one cloud, in canonical T order."""
    d = LT.vec_dir(ds, L)
    w = torch.load(d / "DiffMean_weight.pt", weights_only=False).float().numpy()
    ci = pd.read_parquet(REPO / f"data/axbench/{ds}/concept_index.parquet")
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


PAIRED = [("size_imp_v1", "size", "importance"),
          ("war_arg_v1", "war", "argument"),
          ("food_ideas_v1", "food", "ideas")]

rows = []
for ds, a, b in PAIRED:
    for L in LAYERS:
        A, B = cloud(ds, a, L), cloud(ds, b, L)
        for kind in ("positions", "moves"):
            rows.append(dict(experiment=ds, A=a, B=b, layer=L, kind=kind, **rsa(A, B, kind)))
        rows[-1]["mean_cos_corresponding_moves"] = direction(A, B)

# frames: every frame pair, indexed by role. 6 roles -> 6! = 720 relabelings, so p floors
# at 1/720 = 0.0014 and no single pair can be very significant on its own.
FR = W.F1_FRAMES
for L in LAYERS:
    X = {f: cloud("frames_v1", f, L) for f in FR}
    for f1, f2 in combinations(FR, 2):
        for kind in ("positions", "moves"):
            rows.append(dict(experiment="frames_v1", A=f1, B=f2, layer=L, kind=kind,
                             **rsa(X[f1], X[f2], kind)))
        rows[-1]["mean_cos_corresponding_moves"] = direction(X[f1], X[f2])

df = pd.DataFrame(rows)
df.to_csv(OUT / "wave1_rsa.csv", index=False)

pd.set_option("display.width", 220)
print("=" * 100)
print("POSITIONS + MOVES RSA, against a null that permutes the correspondence T "
      f"({NPERM} draws, p floor {1/(NPERM+1):.1e})")
print("=" * 100)
for ds, a, b in PAIRED:
    s = df[(df.experiment == ds)]
    print(f"\n--- {ds}   {a} <-> {b} ---")
    print(s.pivot_table(index="layer", columns="kind",
                        values=["rho", "null_p95", "p"]).round(4).to_string())

print("\n--- frames_v1: 10 frame pairs x role index (n=6 roles, p floor 1/720=0.0014) ---")
fr = df[df.experiment == "frames_v1"]
print(fr.pivot_table(index=["A", "B"], columns=["kind", "layer"], values="rho").round(2).to_string())
print("\nframes pooled mean rho by layer/kind:")
print(fr.pivot_table(index="layer", columns="kind", values=["rho", "null_mean"]).round(3).to_string())

print("\n=== direction: do corresponding moves POINT the same way? "
      "(paper: water vs temp = +0.08) ===")
print(df[df.mean_cos_corresponding_moves.notna()]
      .pivot_table(index="experiment", columns="layer",
                   values="mean_cos_corresponding_moves").round(3).to_string())

# ---- generation-artefact check at full scale ------------------------------------------ #
print("\n" + "=" * 100)
print("CO-MENTION ARTEFACT CHECK at n=72 (the workshop notebook could only do 3 prompts)")
print("=" * 100)
def comention(ds, cl):
    pos = {}
    for l in open(REPO / f"data/raw/{ds}/outputs.jsonl"):
        r = json.loads(l)
        if r["polarity"] == "positive" and r["vector_id"].startswith(cl + "__"):
            pos.setdefault(r["vector_id"].split("__", 1)[1], []).append(r["completion"])
    ids = W.MOVE_ORDER[cl]
    rec = {r["id"]: r for r in W.CLOUDS[cl]}
    pat = {i: re.compile(r"\b(" + "|".join(re.escape(t) for t in W.concept_terms(rec[i])) +
                         r")\b", re.I) for i in ids}
    C = np.zeros((len(ids), len(ids)))
    for i, x in enumerate(ids):
        for j, y in enumerate(ids):
            if i != j: C[i, j] = np.mean([bool(pat[y].search(t)) for t in pos[x]])
    return C

art = []
for ds, a, b in PAIRED + [("frames_v1", f1, f2) for f1, f2 in combinations(FR, 2)]:
    CA, CB = comention(ds, a), comention(ds, b)
    off = ~np.eye(len(CA), dtype=bool)
    r = spearmanr(CA[off], CB[off]).correlation
    art.append(dict(experiment=ds, A=a, B=b, comention_rho=round(float(r), 3),
                    A_mean=round(float(CA[off].mean()), 3),
                    B_mean=round(float(CB[off].mean()), 3)))
ar = pd.DataFrame(art); ar.to_csv(OUT / "wave1_comention.csv", index=False)
print(ar.to_string(index=False))
print("\nHigh rho => the two clouds were written with the same internal co-mention wiring,")
print("so part of the RSA above would be a generation artefact. Near 0 is what we want.")
print(f"\nwrote {OUT}/wave1_rsa.csv and {OUT}/wave1_comention.csv")
