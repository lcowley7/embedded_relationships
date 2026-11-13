"""make_node_rsa_tables_sweep.py — the NODE version of the temperature RSA, swept over all
layers {0,1,2,3,4,5,10,15,20} and both units (C = temp_c1, F = temp_f1).

Points = the 18 water words (nodes), NOT moves. NO differences, NO ladders, NO local/global.
  RDM_water[i,j] = 1 - cos(W_i, W_j)      W = direct h2o_v1 water-word steering vectors (unit, no deflation)
  RDM_temp[i,j]  = 1 - cos(T(t_i), T(t_j))  T = temp manifold vector at each word's real temp (unit, no deflation)
Node RSA = Spearman(upper-tri RDM_water, upper-tri RDM_temp); shuffled null = permute the 18 labels.
For F, each word's nominal degC is converted to degF (c*9/5+32) before the nearest-row lookup.

Writes:  fractal_analysis/tables/node_rsa_by_layer_unit.csv
"""
import os, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
HERE = "./fractal_analysis"
if HERE not in sys.path: sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE)); import lt_model as LT   # LT_MODEL switch
import analysis_harness as H, alignment_harness as A
DB = H.DATA_BASE; OUT = str(LT.out_dir(HERE + "/tables"))
LAYERS = LT.layers([0, 1, 2, 3, 4, 5, 10, 15, 20]); NPERM = 5000; SEED = 0

def temp_manifold(unit, layer):
    ds = "temp_c1" if unit == "C" else "temp_f1"
    col = "temp_c" if unit == "C" else "temp_f"
    conv = (lambda c: c) if unit == "C" else (lambda c: c * 9.0 / 5.0 + 32.0)
    Wt = H.load_weight(ds, layer).astype(float)
    Tax = pd.read_parquet(f"{DB}/{ds}/concept_index.parquet")[col].values.astype(float)
    s = np.argsort(Tax); Xt = A.unit(Wt[s]); Tax = Tax[s]
    return lambda t_c: Xt[int(np.argmin(np.abs(Tax - conv(t_c))))]

rows = []
for layer in LAYERS:
    h = H.h2o_on_temp_axis(layer)                       # 18 water words + nominal degC temps
    temps = h["temp"].astype(float); Xw = A.unit(h["X"].astype(float)); n = len(temps)
    for unit in ("C", "F"):
        trow = temp_manifold(unit, layer)
        Xtemp = np.vstack([trow(t) for t in temps])     # 18 x D temperature-manifold vectors
        rw = pdist(Xw, "cosine"); rt = pdist(Xtemp, "cosine")
        sp = spearmanr(rw, rt).correlation; pe = pearsonr(rw, rt)[0]
        RW = squareform(rw); rng = np.random.default_rng(SEED); null = np.empty(NPERM)
        for k in range(NPERM):
            p = rng.permutation(n); null[k] = spearmanr(squareform(RW[np.ix_(p, p)]), rt).correlation
        rows.append(dict(layer=layer, unit=unit, N_words=n, n_pairs=n * (n - 1) // 2,
                         spearman=round(sp, 4), pearson=round(pe, 4),
                         null_mean=round(float(null.mean()), 4), null_sd=round(float(null.std()), 4),
                         null_p95=round(float(np.quantile(null, .95)), 4),
                         z=round(float((sp - null.mean()) / (null.std() + 1e-12)), 2),
                         perm_p=round(float((1 + np.sum(null >= sp)) / (NPERM + 1)), 5)))
    print(f"layer {layer:2d} done")

df = pd.DataFrame(rows); df.to_csv(f"{OUT}/node_rsa_by_layer_unit.csv", index=False)
print(f"\nwrote {OUT}/node_rsa_by_layer_unit.csv  (NPERM={NPERM}, seed={SEED})\n")
print("=== node RSA (Spearman) by layer x unit ===")
print(df.pivot_table(index="layer", columns="unit", values="spearman").to_string())
print("\n=== with z-scores ===")
print(df.pivot_table(index="layer", columns="unit", values="z").to_string())
