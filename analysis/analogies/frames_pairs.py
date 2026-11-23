"""frames_pairs.py -- LaTeX rows for (1) the per-profession-pair table at layer 10 (paper Table 9) and
(2) the between-cloud raw-cosine column of Table 8. Reads the vectors and
outputs/tables/analogy_{rsa,direction_stats}.csv (run analogy_rsa.py and direction_stats.py first)."""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pathlib
from itertools import combinations
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))   # repository root
from relgeom import analogies as W
from relgeom.analogy_geometry import cloud
from relgeom.paths import TABLES
ds_ = pd.read_csv(TABLES/"analogy_direction_stats.csv"); rsa = pd.read_csv(TABLES/"analogy_rsa.csv")
NAME = {"doctor": "medical", "lawyer": "legal", "dentist": "dentistry", "agent": "real estate", "teacher": "teaching"}
def raw(A, B):
    C = A @ B.T; off = ~np.eye(len(A), dtype=bool)
    return C.diagonal().mean(), C[off].mean()
# (1) per pair, layer 10
L = 10; X = {f: cloud("frames_v1", f, L) for f in W.F1_FRAMES}
lines = []
for a, b in combinations(W.F1_FRAMES, 2):
    d = ds_[(ds_.experiment == "frames_v1") & (ds_.A == a) & (ds_.B == b) & (ds_.layer == L)].iloc[0]
    r = rsa[(rsa.experiment == "frames_v1") & (rsa.A == a) & (rsa.B == b) & (rsa.layer == L)].set_index("kind")
    same, diff = raw(X[a], X[b])
    lines.append(f"{NAME[a]}--{NAME[b]} & ${same:+.2f}$ & ${diff:+.2f}$ & & ${d.cross_cos:+.2f}$ & ${d.cross_null_p95:+.2f}$ & "
                 f"& ${r.loc['positions','rho']:+.2f}$ & ${r.loc['positions','p']:.3f}$ & ${r.loc['moves','rho']:+.2f}$ & ${r.loc['moves','p']:.3f}$ \\\\")
print("\n".join(lines))
# (2) between-cloud raw cos per family/layer (matched pair and all other pairs)
print()
for L in [5, 10, 15, 20]:
    out = []
    for ds, a, b in [("size_imp_v1", "size", "importance"), ("war_arg_v1", "war", "argument"), ("food_ideas_v1", "food", "ideas")]:
        out.append((ds, *raw(cloud(ds, a, L), cloud(ds, b, L))))
    X = {f: cloud("frames_v1", f, L) for f in W.F1_FRAMES}
    v = np.array([raw(X[p], X[q]) for p, q in combinations(X, 2)]).mean(0)
    out.append(("frames", *v))
    print(L, "  ".join(f"{n}: matched {m:+.2f} other {o:+.2f}" for n, m, o in out))
