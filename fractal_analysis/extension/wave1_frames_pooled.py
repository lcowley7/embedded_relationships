"""wave1_frames_pooled.py -- the professional frames as ONE test (appendix), with robustness checks.

wave1_rsa.py tests each of the C(5,2) = 10 profession pairs separately. Here the statistic is the mean
Spearman rho over those 10 pairs, each pair computed exactly as in wave1_rsa.py (positions: cosine distances
among a profession's 6 role vectors; moves: the 15 role-to-role differences in F1_ROLES order, compared by
cosine distance). Null: in each of 5,000 draws, every profession's 6 role labels are permuted independently
and all pair correlations recomputed; p = (1 + #{null >= observed}) / 5001.

Robustness rows:
  - drop the doctor-dentist pair (highest co-mention, appendix flags it)      -> 9 pairs
  - drop the dentist cloud entirely (weakest-steering frame vectors)          -> 6 pairs
  - positions with the co-mention control of wave1_frames_deep.py: each profession's RDM is
    residualised (rank OLS) on its OWN co-mention matrix and the residuals re-ranked (so the observed
    per-pair statistic equals wave1_frames_deep.py's rho_ctl). Unlike that script's per-pair null, the
    per-profession permutation here moves the RDM AND its co-mention matrix together (co-mention is a
    property of the concept), so the control is rebuilt in every draw.
"""
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
EXT = Path("./fractal_analysis/extension")
sys.path.insert(0, str(EXT)); import wave1_families as W
ns = {"__file__": str(EXT / "wave1_rsa.py")}; exec((EXT / "wave1_rsa.py").read_text().split("PAIRED = [")[0], ns)
cloud, moves = ns["cloud"], ns["moves"]
dp = {"__file__": str(EXT / "wave1_frames_deep.py")}
exec((EXT / "wave1_frames_deep.py").read_text().split("def moves(")[0], dp)       # imports, RO, comention, resid
comention, resid, RO = dp["comention"], dp["resid"], dp["RO"]
assert list(RO) == list(W.F1_ROLES)
FR, LAYERS, NDRAW = W.F1_FRAMES, [5, 10, 15, 20], 5000
iu6, iu15 = np.triu_indices(6, 1), np.triu_indices(15, 1)
cosd = lambda M, iu: (1 - (M / np.linalg.norm(M, axis=1, keepdims=True)) @ (M / np.linalg.norm(M, axis=1, keepdims=True)).T)[iu]
def z(x): r = np.asarray(x, float); r = r - r.mean(); return r / r.std()
def condensed_perm(vec, p, n=6):
    S = np.zeros((n, n)); S[np.triu_indices(n, 1)] = vec; S = S + S.T; return S[np.ix_(p, p)][np.triu_indices(n, 1)]

CM = {f: comention(f) for f in FR}
fi = {f: i for i, f in enumerate(FR)}
SETS = {"all 10 pairs": list(itertools.combinations(FR, 2)),
        "without doctor–dentist (9)": [pr for pr in itertools.combinations(FR, 2) if set(pr) != {"doctor", "dentist"}],
        "without dentist (6)": list(itertools.combinations([f for f in FR if f != "dentist"], 2))}
rng = np.random.default_rng(0); rows = []
for L in LAYERS:
    X = {f: cloud("frames_v1", f, L) for f in FR}
    def sigs(perms, kind):
        out = {}
        for f in FR:
            p = perms[f]
            if kind == "positions": out[f] = z(rankdata(cosd(X[f][p], iu6)))
            elif kind == "moves": out[f] = z(rankdata(cosd(moves(X[f], p), iu15)))
            else: out[f] = z(rankdata(resid(cosd(X[f][p], iu6), condensed_perm(CM[f], p))))   # positions | co-mention (Spearman of residuals, as wave1_frames_deep.py)
        return out
    for kind in ("positions", "moves", "positions | co-mention"):
        ident = sigs({f: np.arange(6) for f in FR}, kind)
        draws = [sigs({f: rng.permutation(6) for f in FR}, kind) for _ in range(NDRAW)]
        for sname, pairs in SETS.items():
            stat = lambda S: np.mean([np.mean(S[a] * S[b]) for a, b in pairs])
            obs = stat(ident); null = np.array([stat(S) for S in draws])
            rows.append(dict(layer=L, kind=kind, set=sname, mean_rho=obs, null_mean=null.mean(),
                             null_p95=np.percentile(null, 95), p=(1 + np.sum(null >= obs)) / (NDRAW + 1)))
            r = rows[-1]
            print(f"L{L:>2} {kind:<23} {sname:<27} mean rho {obs:.3f}  null {r['null_mean']:.3f} (95th {r['null_p95']:.3f})  p {r['p']:.4f}", flush=True)
pd.DataFrame(rows).to_csv(EXT / "tables/frames_pooled.csv", index=False)
