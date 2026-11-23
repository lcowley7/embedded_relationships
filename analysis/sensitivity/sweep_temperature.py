"""sweep_temperature.py -- the temperature analysis as a function of the drop fraction f, at every trained
layer (paper Figure 16, Table 11; needs build_scores.py first).

Items are the 18 water words. Each enters the tables paired with the numeric vector at its
nominal temperature, so it is scored by the MINIMUM of the pair:
    s_i = min( Overall(word_i), Overall(numeric level nearest to t_i) ).
At each f we drop the floor(f*18) lowest-scoring words. Table 3 (nodes) is recomputed on the
surviving words; Table 4 (moves) keeps only moves whose BOTH endpoints survive, and its
word-permutation null runs over the surviving words.

Note this thins the WATER WORDS, not the numeric ladder: the nearest-row lookup that supplies
the numeric half of each pair is unchanged.

With RELGEOM_MODEL set the vectors, judge scores, layers and outputs are that model's
(relgeom/model_switch.py); the ladders and H2O_TEMP table are shared.
Writes outputs/tables/[<tag>/]temp_sweep.csv.
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # repository root
from relgeom import harness as H, model_switch, ladders as LD   # noqa
from relgeom.paths import TABLES as TABLES_ROOT
A = H
DB = H.DATA_BASE
NPERM, SEED = 5000, 0
MIN_MOVES = 5   # below this the move RDM is degenerate (e.g. 3 moves -> 3 pairs); not reported
FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
UNITS = {"C": ("temp_c1", "temp_c", lambda c: c),
         "F": ("temp_f1", "temp_f", lambda c: c * 9.0 / 5.0 + 32.0)}
canon = lambda w: "puddle" if w == "water" else w      # the "water" rung has no probe of its own

SPEC = "proposed"
TABLES = str(model_switch.out_dir(TABLES_ROOT))          # other models never overwrite Gemma's outputs
sc = pd.read_csv(f"{TABLES}/judge_scores.csv")
# a layer is sweepable only if both halves of every pair were steered there
LAYERS = [L for L in model_switch.layers([0, 1, 2, 3, 4, 5, 10, 15, 20])
          if {"h2o_v1", "temp_c1", "temp_f1"} <= set(sc[sc.layer == L].family)]
assert LAYERS, "no layer has water-word and numeric judge scores"
print(f"{model_switch.label()}: sweeping layers {LAYERS}", flush=True)


def temp_axis(unit, layer):
    ds, col, conv = UNITS[unit]
    W = H.load_weight(ds, layer).astype(float)
    Tax = pd.read_parquet(f"{DB}/{ds}/concept_index.parquet")[col].values.astype(float)
    o = np.argsort(Tax); X, Tax = A.unit(W[o]), Tax[o]
    ns = sc[(sc.family == ds) & (sc.layer == layer)]
    nscore = dict(zip(ns.coord.astype(float), ns.overall))
    def look(t_c):
        i = int(np.argmin(np.abs(Tax - conv(t_c))))
        return X[i], float(Tax[i]), nscore.get(float(Tax[i]), np.nan)
    return look


def edges():
    return LD.edges(SPEC)[0]


EDGES = edges()
rows = []
for layer in LAYERS:
    ws = sc[(sc.family == "h2o_v1") & (sc.layer == layer)]
    word_score = dict(zip(ws.label, ws.overall))
    cl = H.load_cloud("h2o_v1", layer)
    LAB = [str(g) for g in cl["labels"]]
    Xw_all = A.unit(cl["X"].astype(float))
    WIDX = {w: i for i, w in enumerate(LAB)}

    for unit in ("C", "F"):
        look = temp_axis(unit, layer)
        pair = {w: min(word_score[w], look(H.H2O_TEMP[w])[2]) for w in LAB}
        order = sorted(LAB, key=lambda w: (pair[w], WIDX[w]))
        for f in FRACS:
            k = int(np.floor(f * len(LAB)))
            dropped = set(order[:k]); keep = [w for w in LAB if w not in dropped]

            # ---- Table 3: node RSA over surviving words ----
            Xw = np.vstack([Xw_all[WIDX[w]] for w in keep])
            Xt = np.vstack([look(H.H2O_TEMP[w])[0] for w in keep])
            rw, rt = pdist(Xw, "cosine"), pdist(Xt, "cosine")
            node = float(spearmanr(rw, rt).correlation)
            RW = squareform(rw); rng = np.random.default_rng(SEED)
            nn = np.array([spearmanr(squareform(RW[np.ix_(p, p)]), rt).correlation
                           for p in (rng.permutation(len(keep)) for _ in range(NPERM))])
            node_p = float((1 + np.sum(nn >= node)) / (NPERM + 1))

            # ---- Table 4: moves with both endpoints surviving ----
            es = [e for e in EDGES if canon(e["src"]) in keep and canon(e["tgt"]) in keep]
            mw = sorted({canon(e["src"]) for e in es} | {canon(e["tgt"]) for e in es})
            mv = mp = mnull = ncos = np.nan
            if len(es) >= 3 and len(mw) >= 4:
                mi = {w: i for i, w in enumerate(mw)}
                Ws = np.vstack([Xw_all[WIDX[w]] for w in mw])
                si = np.array([mi[canon(e["src"])] for e in es])
                ti = np.array([mi[canon(e["tgt"])] for e in es])
                V = np.vstack([look(e["tT"])[0] - look(e["sT"])[0] for e in es])
                dV = pdist(V, "cosine")
                stat = lambda p: spearmanr(pdist(Ws[p][ti] - Ws[p][si], "cosine"), dV).correlation
                mv = float(stat(np.arange(len(mw))))
                rng = np.random.default_rng(SEED)
                nl = np.array([stat(rng.permutation(len(mw))) for _ in range(NPERM)])
                mp = float((1 + np.sum(nl >= mv)) / (NPERM + 1))
                mnull = float(np.quantile(nl, .95))
                G = Ws[ti] - Ws[si]
                gu = G / np.linalg.norm(G, axis=1, keepdims=True)
                vu = V / np.linalg.norm(V, axis=1, keepdims=True)
                ncos = float((gu * vu).sum(1).mean())

            rows.append(dict(layer=layer, unit=unit, f=f, n_words=len(keep),
                             node_rsa=round(node, 4), node_p=round(node_p, 5),
                             node_null_p95=round(float(np.quantile(nn, .95)), 4),
                             n_moves=len(es), n_move_words=len(mw),
                             move_rsa=round(mv, 4), move_p=round(mp, 5),
                             move_null_p95=round(mnull, 4), mean_cos=round(ncos, 4),
                             dropped=",".join(sorted(dropped))))
            print(f"  L{layer:2d} {unit} f={f:.1f}  words {len(keep):2d}  node {node:+.3f} "
                  f"(p {node_p:.4f})  moves {len(es):2d} rsa {mv:+.3f} (p {mp:.4f})", flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{TABLES}/temp_sweep.csv", index=False)

df["move_reportable"] = df.n_moves >= MIN_MOVES
df.to_csv(f"{TABLES}/temp_sweep.csv", index=False)

print(f"\nwrote {TABLES}/temp_sweep.csv")
