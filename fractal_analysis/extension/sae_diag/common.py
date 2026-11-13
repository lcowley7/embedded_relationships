"""common.py -- loaders + the paper's RSA protocols, so SAE-space vectors can be dropped into
exactly the tests that produced the main-text numbers.

  water <-> temperature MOVES : make_move_direction_stats.py (33 h2o ladder edges, de-duplicated,
                                oriented cold->hot, equal-temp pairs dropped; WORD-level null)
  analogy MOVES               : extension/wave1_rsa.py (all i<j chords; T-permutation null)
  positions                   : spearman(pdist(A), pdist(B)), label-permutation null
"""
import os, sys, json
from itertools import combinations
import numpy as np, pandas as pd, torch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

REPO = "."
DB = f"{REPO}/data/axbench"
FA = f"{REPO}/fractal_analysis"
EXT = f"{FA}/extension"
HERE = f"{EXT}/sae_diag"
INP = f"{REPO}/inputs"
sys.path.insert(0, EXT)
import wave1_families as W  # noqa: E402

SNAP = (os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")) + ("/hub/models--google--gemma-scope-2b-pt-res/"
        "snapshots/fd571b47c1c64851e9b1989792367b9babb4af63"))
SAE_PATH = {10: f"{SNAP}/layer_10/width_16k/average_l0_77/params.npz",
            20: f"{SNAP}/layer_20/width_16k/average_l0_71/params.npz"}

# paper's water-word table (analysis_harness.H2O_TEMP), degrees C
H2O_TEMP = {"ice": -10, "snow": -5, "hail": -5, "frost": -3, "sleet": 0, "slush": 1,
            "icy_water": 1, "cold_water": 8, "rain": 10, "dew": 10, "fog": 10, "puddle": 12,
            "cool_water": 15, "roomtemp_water": 21, "warm_water": 35, "hot_water": 55,
            "boiling_water": 100, "steam": 105}
ANALOGY = [("size_imp_v1", "size", "importance"), ("war_arg_v1", "war", "argument"),
           ("food_ideas_v1", "food", "ideas")]
NPERM = 5000
unit = lambda X: X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-30)


def load_sae(L):
    p = np.load(SAE_PATH[L])
    return {k: p[k].astype(np.float32) for k in p.files}


def explanations(L):
    d = json.load(open(f"{REPO}/ax_bench/axbench/axbench/data/gemma-2-2b_{L}-gemmascope-res-16k.json"))
    return {int(r["index"]): r["description"].strip() for r in d}


def diffmean(ds, L):
    """stored DiffMean steering vectors: {vector_id: vec}"""
    w = torch.load(f"{DB}/{ds}/diffmean_l{L}/train/DiffMean_weight.pt", weights_only=False).float().numpy()
    ci = pd.read_parquet(f"{DB}/{ds}/concept_index.parquet")
    return {v: w[i] for v, i in zip(ci.vector_id, ci.concept_id)}


# ---------------------------------------------------------------- concept sets ----------------
def water_ids():
    ci = pd.read_parquet(f"{DB}/h2o_v1/concept_index.parquet")
    return {str(g): v for g, v in zip(ci.concept_group, ci.vector_id)}       # word -> vector_id


def temp_ids(unit_):
    """sorted (temperature, shard, vector_id) for the °C or °F ladder (shards a..)."""
    shards = "abcd" if unit_ == "C" else "abcdef"
    col = "temp_c" if unit_ == "C" else "temp_f"
    full = pd.read_parquet(f"{DB}/temp_{unit_.lower()}1/concept_index.parquet")
    slug2t = dict(zip(full.slug, full[col]))
    rows = []
    for s in shards:
        ds = f"temp_{unit_.lower()}1{s}"
        ci = pd.read_parquet(f"{DB}/{ds}/concept_index.parquet")
        rows += [(float(slug2t[v.split("__")[-1]]), ds, v) for v in ci.vector_id]
    return sorted(rows)


def nearest_temp(rows, t_c, unit_):
    t = t_c if unit_ == "C" else t_c * 9 / 5 + 32
    return rows[int(np.argmin([abs(r[0] - t) for r in rows]))]


# ---------------------------------------------------------------- water/temp move RSA ---------
# CURRENT paper move set (2026-10 revision): ladders_18 SPECS['proposed'] -- 18 words, 49 oriented
# cold->hot moves, equal-temperature pairs (dew/fog, hail/snow) dropped. "water" = the puddle probe;
# the label "water" is kept so the sorted word order (hence permutation draws) matches
# method_audit_2026-09-16/ladders_18/move_stats.py exactly.
# (Superseded: the 14-word / 33-move set of make_move_direction_stats.py, used in the first pass.)
sys.path.insert(0, f"{REPO}/method_audit_2026-09-16/ladders_18")
import ladders as LD  # noqa: E402
_E, _TIES = LD.edges("proposed")
EDGES = [(e["src"], e["tgt"]) for e in _E]
CANON = LD.CANON
assert len(EDGES) == 49


def water_temp_move_rsa(Xw, Tvec, nperm=NPERM, seed=0):
    """Xw: {word: vec} (h2o_v1 names, 'puddle' for the water probe); Tvec: fn(temp_c) -> vec of the
    matched temperature vector. Same statistic/null as ladders_18/move_stats.py (rows unit-normalised)."""
    words = sorted({w for e in EDGES for w in e}); wi = {w: i for i, w in enumerate(words)}
    Wm = unit(np.stack([Xw[CANON(w)] for w in words]))
    si = np.array([wi[a] for a, b in EDGES]); ti = np.array([wi[b] for a, b in EDGES])
    tc = lambda w: H2O_TEMP[CANON(w)]
    V = np.stack([unit(Tvec(tc(b))) - unit(Tvec(tc(a))) for a, b in EDGES]); dV = pdist(V, "cosine")
    stat = lambda p: spearmanr(pdist(Wm[p][ti] - Wm[p][si], "cosine"), dV).correlation
    rho = stat(np.arange(len(words))); rng = np.random.default_rng(seed)
    null = np.array([stat(rng.permutation(len(words))) for _ in range(nperm)])
    G = Wm[ti] - Wm[si]
    return dict(n_items=len(words), n_moves=len(EDGES), rho=rho, null_mean=null.mean(), null_sd=null.std(),
                null_p95=np.quantile(null, .95), z=(rho - null.mean()) / null.std(),
                p=(1 + (null >= rho).sum()) / (nperm + 1),
                cross_cos=float((unit(G) * unit(V)).sum(1).mean()))


def positions_rsa(A, B, nperm=NPERM, seed=0):
    A, B = unit(A), unit(B); da = pdist(A, "cosine"); n = len(A); rng = np.random.default_rng(seed)
    f = lambda p: spearmanr(da, pdist(B[p], "cosine")).correlation
    rho = f(np.arange(n)); null = np.array([f(rng.permutation(n)) for _ in range(nperm)])
    return dict(n_items=n, rho=rho, null_mean=null.mean(), null_sd=null.std(), null_p95=np.quantile(null, .95),
                z=(rho - null.mean()) / null.std(), p=(1 + (null >= rho).sum()) / (nperm + 1))


def _moves(X):
    return np.array([X[j] - X[i] for i, j in combinations(range(len(X)), 2)])


def analogy_move_rsa(A, B, nperm=NPERM, seed=0):
    """wave1_rsa.py 'moves' (A, B rows in canonical T order)."""
    A, B = unit(A), unit(B); n = len(A); rng = np.random.default_rng(seed)
    da = pdist(_moves(A), "cosine")
    f = lambda p: spearmanr(da, pdist(_moves(B[p]), "cosine")).correlation
    rho = f(np.arange(n)); null = np.array([f(rng.permutation(n)) for _ in range(nperm)])
    ma, mb = unit(_moves(A)), unit(_moves(B))
    return dict(n_items=n, n_moves=len(da) and len(_moves(A)), rho=rho, null_mean=null.mean(), null_sd=null.std(),
                null_p95=np.quantile(null, .95), z=(rho - null.mean()) / null.std(),
                p=(1 + (null >= rho).sum()) / (nperm + 1), cross_cos=float((ma * mb).sum(1).mean()))


def analogy_ids(ds, a, b):
    T = W.EXPERIMENTS[ds]["T"]
    return [f"{a}__{x}" for x, _ in T], [f"{b}__{y}" for _, y in T]
