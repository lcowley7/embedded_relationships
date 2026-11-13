"""wave1_frames_deep.py -- the frames experiment, examined properly.

Four things the headline RSA table does not answer:

  1. F-a WITH THE CO-MENTION CONTROL. Every role's positives reach for their own
     PRACTITIONER, in every frame, so the two clouds' co-mention matrices correlate at
     rho ~ 0.5 under T. Residualise each cloud's RDM on its OWN co-mention matrix (OLS on
     ranks) and re-correlate: what survives is the part of the agreement that was not
     written into the training text. This is build_temp_partial_control.py's move, with the
     co-mention matrix in place of the pure-number reference.

  2. F-b, THE TRANSPOSED RESLICE (plan v1 section 5). Same 30 vectors, cut the other way:
     a cloud per ROLE across the five professions, indexed by profession. Asks whether
     profession-similarity structure transfers from the practitioner cloud to the recipient
     cloud to the venue cloud -- the design closest in spirit to capitals<->governments.

  3. THE PARALLELOGRAM. Is patient - doctor ~ client - lawyer? For each role pair, take the
     five profession-wise offsets and measure how parallel they are. This is the word2vec
     analogy structure, measured on causal probes at depth rather than static embeddings,
     and it is a DIRECTION question -- RSA cannot see it.

  4. Graded frame similarity, and sensitivity to the weak dentist vectors.

n is small (5 professions, 6 roles), so every null here is EXACT -- all 120 or 720
relabelings enumerated, not sampled. p floors are 1/120 = 0.0083 and 1/720 = 0.0014.

Run: PY fractal_analysis/extension/wave1_frames_deep.py
"""
import json, sys, re, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from pathlib import Path
from itertools import combinations, permutations
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, rankdata

REPO = Path(".")
EXT = REPO / "fractal_analysis/extension"
sys.path.insert(0, str(EXT)); import wave1_families as W
OUT = EXT / "tables"; OUT.mkdir(exist_ok=True)
DS = "frames_v1"; LAYERS = [5, 10, 15, 20]
FR, RO = W.F1_FRAMES, W.F1_ROLES
unit = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)

# ---- vectors, indexed [frame][role] --------------------------------------------------- #
VEC = {}
for L in LAYERS:
    d = REPO / f"data/axbench/{DS}/diffmean_l{L}/train"
    w = torch.load(d / "DiffMean_weight.pt", weights_only=False).float().numpy()
    ci = pd.read_parquet(REPO / f"data/axbench/{DS}/concept_index.parquet")
    idx = {v: i for i, v in zip(ci.concept_id, ci.vector_id)}
    rid = {(r["frame"], r["role"]): r["id"] for f in FR for r in W.CLOUDS[f]}
    VEC[L] = {f: unit(np.array([w[idx[f"{f}__{rid[(f, ro)]}"]] for ro in RO])) for f in FR}

# ---- judge Overall per vector, for the sensitivity sweep ------------------------------ #
EFF = {}
ci = pd.read_parquet(REPO / f"data/axbench/{DS}/concept_index.parquet")
v_of = dict(zip(ci.concept_id, ci.vector_id))
for L in LAYERS:
    for l in open(REPO / f"data/axbench/{DS}/diffmean_l{L}/eval/results.jsonl"):
        r = json.loads(l)
        EFF[(L, v_of[r["concept_id"]])] = max(
            r["results"]["LMJudgeEvaluator"]["DiffMean"]["lm_judge_rating"])

# ---- co-mention, per frame, over the real n=72 positives ------------------------------ #
def comention(cl):
    pos = {}
    for l in open(REPO / f"data/raw/{DS}/outputs.jsonl"):
        r = json.loads(l)
        if r["polarity"] == "positive" and r["vector_id"].startswith(cl + "__"):
            pos.setdefault(r["vector_id"].split("__", 1)[1], []).append(r["completion"])
    rid = {r["role"]: r for r in W.CLOUDS[cl]}
    pat = {ro: re.compile(r"\b(" + "|".join(re.escape(t) for t in W.concept_terms(rid[ro]))
                          + r")\b", re.I) for ro in RO}
    C = np.zeros((len(RO), len(RO)))
    for i, a in enumerate(RO):
        for j, b in enumerate(RO):
            if i != j: C[i, j] = np.mean([bool(pat[b].search(t)) for t in pos[rid[a]["id"]]])
    C = (C + C.T) / 2                      # symmetric "how linked are these two roles in text"
    return C[np.triu_indices(len(RO), 1)]  # as a condensed vector, matching pdist order
CO = {f: comention(f) for f in FR}

def resid(y, x):
    """Residual of rank(y) after OLS on rank(x). Correlating two residuals is the partial."""
    ry, rx = rankdata(y), rankdata(x)
    if rx.std() == 0: return ry - ry.mean()
    b = np.polyfit(rx, ry, 1)
    return ry - (b[0] * rx + b[1])

def moves(X, perm=None):
    Y = X if perm is None else X[perm]
    return np.array([Y[j] - Y[i] for i, j in combinations(range(len(X)), 2)])

def exact_null(da, fb, n):
    return np.array([spearmanr(da, fb(np.array(p))).correlation for p in permutations(range(n))])

def test(A, B, kind, cA=None, cB=None):
    """rho + EXACT permutation p. If cA/cB given, both sides are residualised on their own
    co-mention structure first, so the statistic is the agreement not attributable to it."""
    n = len(A)
    if kind == "positions":
        da, gb = pdist(A, "cosine"), lambda p: pdist(B[p], "cosine")
    else:
        da, gb = pdist(moves(A), "cosine"), lambda p: pdist(moves(B, p), "cosine")
    if cA is not None:
        da = resid(da, cA); gb = (lambda g: (lambda p: resid(g(p), cB)))(gb)
    obs = spearmanr(da, gb(np.arange(n))).correlation
    null = exact_null(da, gb, n)
    return dict(rho=round(float(obs), 3), null_mean=round(float(null.mean()), 3),
                null_p95=round(float(np.percentile(null, 95)), 3),
                p=round(float((null >= obs).mean()), 4))

# ===== 1. F-a with and without the co-mention control ================================== #
rows = []
for L in LAYERS:
    for f1, f2 in combinations(FR, 2):
        A, B = VEC[L][f1], VEC[L][f2]
        for kind in ("positions", "moves"):
            raw = test(A, B, kind)
            ctl = test(A, B, kind, CO[f1], CO[f2]) if kind == "positions" else None
            rows.append(dict(layer=L, A=f1, B=f2, kind=kind, **{"rho": raw["rho"],
                        "p": raw["p"], "null_p95": raw["null_p95"],
                        "rho_ctl": ctl["rho"] if ctl else np.nan,
                        "p_ctl": ctl["p"] if ctl else np.nan}))
fa = pd.DataFrame(rows); fa.to_csv(OUT / "frames_fa_controlled.csv", index=False)

pd.set_option("display.width", 240)
print("=" * 96)
print("1. F-a (frame as cloud, indexed by ROLE, n=6 -> exact null over 720 relabelings)")
print("=" * 96)
p = fa[fa.kind == "positions"]
print("\npositions -- raw rho | rho after residualising each cloud on its OWN co-mention matrix")
print(p.pivot_table(index=["A", "B"], columns="layer", values=["rho", "rho_ctl"]).round(2).to_string())
print(f"\npairs with p<0.05, positions:  raw {(p.p<0.05).sum()}/40   controlled {(p.p_ctl<0.05).sum()}/40")
m = fa[fa.kind == "moves"]
print(f"pairs with p<0.05, moves:      raw {(m.p<0.05).sum()}/40")
print("\nmean over the 10 pairs, by layer:")
print(fa.pivot_table(index="layer", columns="kind", values=["rho", "rho_ctl"]).round(3).to_string())

# ===== 2. F-b: role as cloud, indexed by PROFESSION ==================================== #
print("\n" + "=" * 96)
print("2. F-b (role as cloud, indexed by PROFESSION, n=5 -> exact null over 120; p floor 0.0083)")
print("=" * 96)
rows = []
for L in LAYERS:
    RC = {ro: unit(np.array([VEC[L][f][i] for f in FR])) for i, ro in enumerate(RO)}
    for r1, r2 in combinations(RO, 2):
        t = test(RC[r1], RC[r2], "positions")
        rows.append(dict(layer=L, roleA=r1, roleB=r2, **t))
fb = pd.DataFrame(rows); fb.to_csv(OUT / "frames_fb_reslice.csv", index=False)
print("\nDoes profession-similarity structure transfer from one role cloud to another?")
print(fb.pivot_table(index=["roleA", "roleB"], columns="layer", values="rho").round(2).to_string())
print(f"\nrole pairs with p<0.05: {(fb.p<0.05).sum()}/{len(fb)}   "
      f"(mean rho {fb.rho.mean():.2f}, mean null {fb.null_mean.mean():.2f})")
print("by layer:"); print(fb.assign(s=fb.p < 0.05).groupby("layer").agg(
    n_sig=("s", "sum"), mean_rho=("rho", "mean")).round(3).to_string())

# ===== 3. the parallelogram ============================================================ #
print("\n" + "=" * 96)
print("3. PARALLELOGRAM -- is (patient - doctor) ~ (client - lawyer)?")
print("=" * 96)
rows = []
for L in LAYERS:
    for i, j in combinations(range(len(RO)), 2):
        off = unit(np.array([VEC[L][f][j] - VEC[L][f][i] for f in FR]))
        obs = float(np.mean([off[a] @ off[b] for a, b in combinations(range(len(FR)), 2)]))
        # null: the offset for each profession is rebuilt from a DIFFERENT profession's
        # role-j vector, so the role transition is no longer profession-internal.
        null = []
        for pm in permutations(range(len(FR))):
            if all(a == b for a, b in zip(pm, range(len(FR)))): continue
            o = unit(np.array([VEC[L][FR[pm[k]]][j] - VEC[L][FR[k]][i] for k in range(len(FR))]))
            null.append(np.mean([o[a] @ o[b] for a, b in combinations(range(len(FR)), 2)]))
        null = np.array(null)
        rows.append(dict(layer=L, roleA=RO[i], roleB=RO[j], mean_cos=round(obs, 3),
                         null_mean=round(float(null.mean()), 3),
                         p=round(float((null >= obs).mean()), 4)))
pg = pd.DataFrame(rows); pg.to_csv(OUT / "frames_parallelogram.csv", index=False)
print("\nmean pairwise cosine among the 5 profession-wise offsets for each role transition:")
print(pg.pivot_table(index=["roleA", "roleB"], columns="layer", values="mean_cos").round(2).to_string())
print(f"\nrole transitions with p<0.05: {(pg.p<0.05).sum()}/{len(pg)}")
print("strongest at L10:")
print(pg[pg.layer == 10].nlargest(5, "mean_cos")[["roleA","roleB","mean_cos","null_mean","p"]].to_string(index=False))

# ===== 4. graded similarity + sensitivity ============================================== #
print("\n" + "=" * 96)
print("4. Does alignment track how similar the two frames are? + weak-vector sensitivity")
print("=" * 96)
GROUP = {"doctor": "health", "dentist": "health", "lawyer": "transactional",
         "agent": "transactional", "teacher": "education"}
p10 = fa[(fa.kind == "positions") & (fa.layer == 10)].copy()
p10["same_group"] = [GROUP[a] == GROUP[b] for a, b in zip(p10.A, p10.B)]
print("\n(a-priori grouping fixed before looking: health {doctor,dentist}, "
      "transactional {lawyer,agent}, education {teacher})")
print(p10.sort_values("rho", ascending=False)[["A","B","same_group","rho","p","rho_ctl","p_ctl"]].to_string(index=False))
print(f"\nmean rho  same-group {p10[p10.same_group].rho.mean():.2f}  vs  "
      f"cross-group {p10[~p10.same_group].rho.mean():.2f}")

print("\nweakest frames vectors at L10 (judge Overall):")
w10 = sorted(((v, s) for (L, v), s in EFF.items() if L == 10), key=lambda x: x[1])[:6]
for v, s in w10: print(f"   {v:28s} {s:.2f}")
print("\nF-a positions at L10 with the dentist frame dropped (its cloud is the weak one):")
sub = p10[(p10.A != "dentist") & (p10.B != "dentist")]
print(f"   {len(sub)} remaining pairs: mean rho {sub.rho.mean():.2f}, "
      f"{(sub.p<0.05).sum()} with p<0.05, mean controlled rho {sub.rho_ctl.mean():.2f}")
print(f"\nwrote 4 CSVs to {OUT}/")
