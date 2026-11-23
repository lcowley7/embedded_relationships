"""moves_rsa.py -- the temperature MOVE analysis: water-word moves vs numeric-temperature moves.

Moves are the 49 colder->warmer pairs on the three ladders of relgeom/ladders.py (18 water words).
For each layer and unit (C = temp_c1, F = temp_f1): the move RSA against a word-permutation null
(5000 draws, seed 0), and the direction statistics (corresponding-move cosine, within-domain cosine).

Paper: Table 2 (layer 10), Tables 6-7 (all layers), Figure 15; Tables 10/13 with RELGEOM_MODEL=<tag>.
Writes outputs/tables/[<tag>/]move_direction_stats_L{L}.csv and move_direction_stats_by_layer.csv.
LAYERS=10 (Gemma indexing) restricts the run; with RELGEOM_MODEL=<tag> layers map to the same relative depth."""
import os, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # repository root
from relgeom import ladders as LD, model_switch, harness as H
from relgeom.paths import TABLES
A = H
DB = H.DATA_BASE; NPERM = 5000
SPEC = "proposed"
LAYERS = model_switch.layers([int(x) for x in os.environ.get("LAYERS", "0,1,2,3,4,5,10,15,20").split(",")])
OUT = str(model_switch.out_dir(TABLES))
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
