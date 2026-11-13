"""move_stats.py SPEC -- make_move_direction_stats.py (same statistics, seed, null and output
columns) with the move set taken from ladders.SPECS[SPEC] instead of the ladder files.
Runs every layer; writes out/SPEC/move_direction_stats_L{L}.csv and _by_layer.csv.

With LT_MODEL=<tag> (lt_model.py) the vectors are that model's, LAYERS (given in
Gemma's indexing) are mapped to the same relative depth, and output goes to out/SPEC/<tag>/.
Unset, behaviour and output are exactly as before. OUT_ROOT overrides out/ (for checks)."""
import os, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
import ladders as LD
import lt_model as LT
import analysis_harness as H, alignment_harness as A
DB = H.DATA_BASE; NPERM = 5000
SPEC = sys.argv[1]
LAYERS = LT.layers([int(x) for x in os.environ.get("LAYERS", "0,1,2,3,4,5,10,15,20").split(",")])
OUT = f"{os.environ.get('OUT_ROOT', os.path.dirname(os.path.abspath(__file__)) + '/out')}/{SPEC}"
if LT.TAG is not None:
    OUT += f"/{LT.TAG}" if LT.SPAN == "axb" else f"/{LT.TAG}_content"
os.makedirs(OUT, exist_ok=True)
EDGES, TIES = LD.edges(SPEC)


def temp_manifold(unit, layer):
    ds = "temp_c1" if unit == "C" else "temp_f1"
    col = "temp_c" if unit == "C" else "temp_f"
    conv = (lambda c: c) if unit == "C" else (lambda c: c * 9.0 / 5.0 + 32.0)
    Wt = H.load_weight(ds, layer).astype(float)
    Tax = pd.read_parquet(f"{DB}/{ds}/concept_index.parquet")[col].values.astype(float)
    s = np.argsort(Tax); Xt = A.unit(Wt[s]); Tax = Tax[s]
    return lambda t_c: Xt[int(np.argmin(np.abs(Tax - conv(t_c))))]


def unitrows(M): return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-30)


def mean_pairwise_cos(M):
    U = unitrows(M); C = U @ U.T; iu = np.triu_indices(len(U), 1)
    return float(C[iu].mean()), float(np.abs(C[iu]).mean())


allrows = []
for LAYER in LAYERS:
    cl = H.load_cloud("h2o_v1", LAYER); H2O = {str(g): i for i, g in enumerate(cl["labels"])}
    H2O.setdefault("water", H2O.get("puddle")); Xw = A.unit(cl["X"].astype(float))
    words = sorted({e["src"] for e in EDGES} | {e["tgt"] for e in EDGES})
    widx = {w: i for i, w in enumerate(words)}; nw = len(words)
    Wsub = np.stack([Xw[H2O[w]] for w in words])
    si = np.array([widx[e["src"]] for e in EDGES]); ti = np.array([widx[e["tgt"]] for e in EDGES])
    rows = []
    for unit in ("C", "F"):
        trow = temp_manifold(unit, LAYER)
        V = np.stack([trow(e["tT"]) - trow(e["sT"]) for e in EDGES]); dV = pdist(V, "cosine")
        assert np.isfinite(dV).all(), "a numeric move is zero (two temperatures map to one level)"
        stat = lambda perm: spearmanr(pdist(Wsub[perm][ti] - Wsub[perm][si], "cosine"), dV).correlation
        rho = stat(np.arange(nw))
        rng = np.random.default_rng(0)
        null = np.array([stat(rng.permutation(nw)) for _ in range(NPERM)])
        p = float((1 + np.sum(null >= rho)) / (NPERM + 1))
        G = Wsub[ti] - Wsub[si]; cross = (unitrows(G) * unitrows(V)).sum(1)
        w_mean, w_abs = mean_pairwise_cos(G); t_mean, t_abs = mean_pairwise_cos(V)
        rows.append(dict(layer=LAYER, unit=unit, n_words=nw, n_moves=len(EDGES),
            rho=round(float(rho), 4), null_mean=round(float(null.mean()), 4),
            null_sd=round(float(null.std()), 4), null_p95=round(float(np.quantile(null, .95)), 4),
            p_word=round(p, 5), cross_cos_mean=round(float(cross.mean()), 4),
            cross_cos_sd=round(float(cross.std()), 4), cross_cos_absmean=round(float(np.abs(cross).mean()), 4),
            within_water_cos=round(w_mean, 4), within_water_abscos=round(w_abs, 4),
            within_temp_cos=round(t_mean, 4), within_temp_abscos=round(t_abs, 4)))
        r = rows[-1]
        print(f"{SPEC} L{LAYER:2d} {unit}: rho {r['rho']:+.4f} null {r['null_mean']:+.4f} p95 {r['null_p95']:+.4f} "
              f"p {r['p_word']:.4f} | cross {r['cross_cos_mean']:+.4f}", flush=True)
    df = pd.DataFrame(rows); df["raw_word_cos"] = round(mean_pairwise_cos(Wsub)[0], 4)
    df.to_csv(f"{OUT}/move_direction_stats_L{LAYER}.csv", index=False); allrows.append(df)
pd.concat(allrows).to_csv(f"{OUT}/move_direction_stats_by_layer.csv", index=False)
print("done", SPEC)
