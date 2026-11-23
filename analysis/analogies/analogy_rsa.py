"""analogy_rsa.py -- the analogy analysis (Section 5): positions and moves RSA for the four analogy datasets.

Positions and moves, both reported, against the label-permutation null that reshuffles the
correspondence T -- the same statistics as the temperature analysis, with T supplied by
relgeom/analogies.py instead of by a water word's nominal temperature.

  PM_A(i,j) = 1 - cos(a_i, a_j)                  within family A only
  PM_B(i,j) = 1 - cos(b_T(i), b_T(j))            within family B only
  positions rho = spearman(pdist(A), pdist(B))

  moves: m_{i->j} = a_j - a_i for every i<j in canonical order; the counterpart in B is
  b_T(j) - b_T(i). Move RSA correlates the two move-by-move distance matrices. The null
  permutes T and REBUILDS B's moves over the same A pairs, so it is centred well above 0
  (two moves sharing an endpoint are similar whatever the labelling) -- read rho against
  the null, never against zero.

Also runs the generation-artefact check (co-mention): if cloud A's co-mention matrix rank-correlates with cloud B's UNDER T, then part of any
agreement was written into the training text rather than found in the model.

Paper: Figure 3, Tables 10/13 (with RELGEOM_MODEL=<tag>).
Writes outputs/tables/[<tag>/]analogy_rsa.csv and analogy_comention.csv, and prints a summary.
"""
import json, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, re
from pathlib import Path
from itertools import combinations
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root
from relgeom import analogies as W, model_switch
from relgeom.analogy_geometry import cloud, moves, rsa, direction, unit, NPERM, SEED
from relgeom.paths import RAW, TABLES
OUT = model_switch.out_dir(TABLES)
LAYERS = model_switch.layers([5, 10, 15, 20])


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
df.to_csv(OUT / "analogy_rsa.csv", index=False)

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
print("CO-MENTION ARTEFACT CHECK (n=72 prompts)")
print("=" * 100)
def comention(ds, cl):
    pos = {}
    for l in open(RAW / ds / "outputs.jsonl"):
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
ar = pd.DataFrame(art); ar.to_csv(OUT / "analogy_comention.csv", index=False)
print(ar.to_string(index=False))
print("\nHigh rho => the two clouds were written with the same internal co-mention wiring,")
print("so part of the RSA above would be a generation artefact. Near 0 is what we want.")
print(f"\nwrote {OUT}/analogy_rsa.csv and {OUT}/analogy_comention.csv")
