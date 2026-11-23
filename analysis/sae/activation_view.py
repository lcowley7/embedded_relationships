"""activation_view.py -- DiffMean in SAE feature space (Appendix, Figure 18).

From data/sae_acts/acts_<ds>.npz (pipeline/sae/extract_sae_acts.py), for each concept c at layer L:
    d_c = mean_tokens SAE(x | positives of c) - mean_tokens SAE(x | that concept's negatives)   [16384]
i.e. exactly the DiffMean contrast, computed in the GemmaScope code instead of the residual.
The residual version of the same contrast is checked against the stored DiffMean_weight (cos).

Analyses (water <-> °C, water <-> °F, and the three analogy pairs):
  overlap  top-k feature overlap (Jaccard) for corresponding vs non-corresponding cross-domain pairs
  general  features in the top-20 of >=25% of concepts in BOTH domains (+ explanations)
  graded   features whose d_c tracks nominal temperature in water AND in the ladder (joint count vs
           a permutation null over the water words' temperatures)
  rsa      the paper's position/move RSA computed on d_c, (i) all features, (ii) only the features
           active in both domains ("shared"), (iii) after deleting every shared feature, so the two
           domains' codes have disjoint support ("exclusive"). Shared(k) = in the top-k of at least
           one concept in EACH domain; k = 20, 50, 200.
  matched  per corresponding pair, the top-20 features the two concepts share (Figure 18a)
Paper: Figure 18 (layer 10). Writes outputs/tables/act_*.csv.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # repository root
from relgeom.sae import *
from relgeom.paths import SAE_ACTS, TABLES, ensure

OUT = str(ensure(TABLES)); DATA = str(SAE_ACTS)
LAYERS = [10, 20]
rng = np.random.default_rng(0)
NP = int(sys.argv[1]) if len(sys.argv) > 1 else NPERM


def load(ds, L):
    z = np.load(f"{DATA}/acts_{ds}.npz", allow_pickle=True)
    vid = list(z["vector_id"]); n = z["ntok"].astype(float)[:, None]
    nneg = sum(v.startswith("__NEG") for v in vid)
    act = z[f"L{L}_sum_act"] / n; res = z[f"L{L}_sum_resid"] / n; fire = z[f"L{L}_sum_fire"] / n
    ni = z["neg_index"]
    out = {}
    for j, v in enumerate(vid[nneg:]):
        i = nneg + j; k = ni[j]
        out[v] = dict(d=act[i] - act[k], r=res[i] - res[k], fire=fire[i] - fire[k], pos=act[i])
    return out, float(z[f"L{L}_fvu"]), float(z[f"L{L}_l0"])


def zrank(S):
    R = rankdata(S, axis=0); return (R - R.mean(0)) / (R.std(0) + 1e-12)


def spear_cols(R, t):
    r = rankdata(t); r = (r - r.mean()) / r.std(); return r @ R / len(r)


def families(L):
    """name -> (A_names, A_D, B_names, B_D, match(list of (iA,iB)), tA, tB, stored_A, stored_B, recon_cos)"""
    fam, qc = {}, []
    h, fvu, l0 = load("h2o_v1", L); qc.append(("h2o_v1", fvu, l0))
    st = diffmean("h2o_v1", L); wid = water_ids()
    words = sorted(H2O_TEMP, key=lambda w: (H2O_TEMP[w], w))
    A = np.stack([h[wid[w]]["d"] for w in words]); Ar = np.stack([h[wid[w]]["r"] for w in words])
    As = np.stack([st[wid[w]] for w in words]); tA = np.array([H2O_TEMP[w] for w in words], float)
    for U in "CF":
        rows = temp_ids(U); acts, sto = {}, {}
        for ds in sorted(set(r[1] for r in rows)):
            a, fvu, l0 = load(ds, L); acts.update(a); sto.update(diffmean(ds, L)); qc.append((ds, fvu, l0))
        B = np.stack([acts[r[2]]["d"] for r in rows]); Br = np.stack([acts[r[2]]["r"] for r in rows])
        Bs = np.stack([sto[r[2]] for r in rows]); tB = np.array([r[0] for r in rows])
        names = [f"{r[0]:g}°{U}" for r in rows]
        m = [(i, names.index(f"{nearest_temp(rows, H2O_TEMP[w], U)[0]:g}°{U}")) for i, w in enumerate(words)]
        fam[f"water-temp{U}"] = dict(An=words, A=A, Bn=names, B=B, match=m, tA=tA, tB=tB, Ar=Ar, Br=Br, As=As, Bs=Bs)
    for ds, a, b in ANALOGY:
        acts, fvu, l0 = load(ds, L); qc.append((ds, fvu, l0)); sto = diffmean(ds, L); ia, ib = analogy_ids(ds, a, b)
        fam[f"{a}-{b}"] = dict(An=ia, A=np.stack([acts[i]["d"] for i in ia]), Bn=ib, B=np.stack([acts[i]["d"] for i in ib]),
                              match=[(i, i) for i in range(len(ia))],
                              Ar=np.stack([acts[i]["r"] for i in ia]), Br=np.stack([acts[i]["r"] for i in ib]),
                              As=np.stack([sto[i] for i in ia]), Bs=np.stack([sto[i] for i in ib]))
    return fam, qc


def topsets(D, k): return [set(np.argsort(-x)[:k]) for x in D]


def rsa_on(name, F, A, B):
    if name.startswith("water"):
        words = F["An"]; Xw = {w: A[i] for i, w in enumerate(words)}
        Bn = F["Bn"]; idx = {n: i for i, n in enumerate(Bn)}
        U = name[-1]; rows = temp_ids(U)
        Tf = lambda t: B[idx[f"{nearest_temp(rows, t, U)[0]:g}°{U}"]]
        return water_temp_move_rsa(Xw, Tf, nperm=NP)
    return analogy_move_rsa(A, B, nperm=NP)


def main():
    ov, gen, grad, rsa, tops, qcs, recon, mps = [], [], [], [], [], [], [], []
    for L in LAYERS:
        ex = explanations(L); fam, qc = families(L)
        qcs += [dict(layer=L, dataset=d, fvu=f, l0=l) for d, f, l in qc]
        for name, F in fam.items():
            A, B = F["A"], F["B"]
            # sanity: residual contrast vs stored DiffMean
            recon.append(dict(layer=L, family=name, cos_resid_vs_stored_A=np.mean((unit(F["Ar"]) * unit(F["As"])).sum(1)),
                              cos_resid_vs_stored_B=np.mean((unit(F["Br"]) * unit(F["Bs"])).sum(1)),
                              min_cos=min((unit(F["Ar"]) * unit(F["As"])).sum(1).min(), (unit(F["Br"]) * unit(F["Bs"])).sum(1).min())))
            m = np.zeros((len(A), len(B)), bool)
            for i, j in F["match"]: m[i, j] = True
            Cab = unit(A) @ unit(B).T
            for k in (10, 20, 50):
                TA, TB = topsets(A, k), topsets(B, k)
                J = np.array([[len(a & b) / len(a | b) for b in TB] for a in TA])
                Sh = np.array([[len(a & b) for b in TB] for a in TA])
                ov.append(dict(layer=L, family=name, k=k, jacc_matched=J[m].mean(), jacc_null=J[~m].mean(),
                               shared_matched=Sh[m].mean(), shared_null=Sh[~m].mean(),
                               frac_matched_any=(Sh[m] > 0).mean(), frac_null_any=(Sh[~m] > 0).mean(),
                               auc=np.mean([(J[i, j] > J[i][~m[i]]).mean() + .5 * (J[i, j] == J[i][~m[i]]).mean() for i, j in F["match"]]),
                               cos_code_matched=Cab[m].mean(), cos_code_null=Cab[~m].mean(),
                               cos_resid_matched=(unit(F["Ar"]) @ unit(F["Br"]).T)[m].mean()))
                if k == 20:
                    if name != "water-tempF":
                        for i, j in F["match"]:
                            s_ = sorted(int(f) for f in TA[i] & TB[j])
                            mps.append(dict(layer=L, family=name, pair=f"{F['An'][i]} | {F['Bn'][j]}",
                                            n_shared_top20=len(s_),
                                            shared="; ".join(f"{f}:{ex.get(f, '')[:45]}" for f in s_)))
                    ca = np.bincount(np.concatenate([list(t) for t in TA]), minlength=16384) / len(TA)
                    cb = np.bincount(np.concatenate([list(t) for t in TB]), minlength=16384) / len(TB)
                    for f in np.where((ca >= .25) & (cb >= .25))[0]:
                        gen.append(dict(layer=L, family=name, feature=int(f), frac_A=ca[f], frac_B=cb[f],
                                        mean_d_A=A[:, f].mean(), mean_d_B=B[:, f].mean(), explanation=ex.get(int(f), "")))
                    for i, j in F["match"]:
                        for side, n, x, part in (("A", F["An"][i], A[i], TB[j]), ("B", F["Bn"][j], B[j], TA[i])):
                            for r_, f in enumerate(np.argsort(-x)[:10]):
                                tops.append(dict(layer=L, family=name, pair=f"{F['An'][i]} | {F['Bn'][j]}", side=side,
                                                 concept=n, rank=r_ + 1, feature=int(f), d=round(float(x[f]), 3),
                                                 in_partner_top20=int(f in part), explanation=ex.get(int(f), "")))
            # graded (temperature families only)
            if "tA" in F:
                RA = zrank(A); rA = spear_cols(RA, F["tA"]); rB = spear_cols(zrank(B), F["tB"])
                alive = (np.abs(A).max(0) > 0) & (np.abs(B).max(0) > 0)
                for thr in (0.5, 0.7):
                    hit = lambda ra: int(((np.abs(ra) > thr) & (np.abs(rB) > thr) & (np.sign(ra) == np.sign(rB)) & alive).sum())
                    obs = hit(rA); nul = np.array([hit(spear_cols(RA, rng.permutation(F["tA"]))) for _ in range(1000)])
                    grad.append(dict(layer=L, family=name, kind="count", thr=thr, n_joint=obs, null_mean=nul.mean(),
                                     null_p95=np.quantile(nul, .95), p=(1 + (nul >= obs).sum()) / 1001))
                joint = np.where(alive & (np.sign(rA) == np.sign(rB)), np.minimum(np.abs(rA), np.abs(rB)), 0)
                for f in np.argsort(-joint)[:10]:
                    grad.append(dict(layer=L, family=name, kind="top", feature=int(f), rho_water=round(rA[f], 3),
                                     rho_ladder=round(rB[f], 3), max_d_water=round(A[:, f].max(), 3),
                                     max_d_ladder=round(B[:, f].max(), 3), explanation=ex.get(int(f), "")))
            # RSA in code space
            def add(label, Asub, Bsub, nfeat):
                r = rsa_on(name, F, Asub, Bsub)
                rsa.append(dict(layer=L, family=name, space=label, n_features=nfeat,
                                **{k_: round(float(v), 4) for k_, v in r.items()}))
            add("residual_stored", F["As"], F["Bs"], 2304)
            add("sae_all", A, B, 16384)
            for k in (20, 50, 200):
                S = sorted(set(np.argsort(-A, 1)[:, :k].ravel()) & set(np.argsort(-B, 1)[:, :k].ravel()))
                keep = np.setdiff1d(np.arange(16384), S)
                add(f"sae_shared_top{k}", A[:, S], B[:, S], len(S))
                add(f"sae_minus_shared_top{k}", A[:, keep], B[:, keep], len(keep))
            # strict disjoint support: delete every feature with a non-negligible contrast in BOTH
            # domains (|d| >= 5% of the typical per-concept peak |d| of that domain, in any concept)
            act_dom = lambda X: (np.abs(X) >= 0.05 * np.median(np.abs(X).max(1))).any(0)
            both_nz = np.where(act_dom(A) & act_dom(B))[0]
            ex_keep = np.setdiff1d(np.arange(16384), both_nz)
            add("sae_disjoint_support", A[:, ex_keep], B[:, ex_keep], len(ex_keep))
            print(f"L{L} {name}: done (both-nonzero feats={len(both_nz)})", flush=True)
    W_ = lambda rows, f: pd.DataFrame(rows).round(4).to_csv(f"{OUT}/{f}", index=False)
    W_(ov, "act_overlap.csv"); W_(gen, "act_general_features.csv"); W_(grad, "act_graded_features.csv")
    W_(rsa, "act_rsa.csv"); pd.DataFrame(tops).to_csv(f"{OUT}/act_top_features.csv", index=False)
    W_(qcs, "act_sae_qc.csv"); W_(recon, "act_recon_check.csv")
    pd.DataFrame(mps).to_csv(f"{OUT}/act_matched_pair_shared.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
    for f in ("act_recon_check", "act_sae_qc", "act_overlap", "act_general_features", "act_graded_features", "act_rsa"):
        d = pd.read_csv(f"{OUT}/{f}.csv")
        if f == "act_overlap": d = d[d.k == 20]
        print(f"\n== {f}\n{d.to_string(index=False)}")


if __name__ == "__main__":
    main()
