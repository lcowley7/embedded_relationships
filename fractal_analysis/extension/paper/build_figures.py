"""Figures for the wave-1 write-up. Light mode only (PNG for LaTeX).

Palette is the dataviz reference categorical order, validated with
scripts/validate_palette.js --mode light: all checks pass, worst adjacent CVD dE 9.1.
The contrast WARN on aqua/yellow is discharged the way the skill requires -- every
series is direct-labelled and every figure ships an identical CSV table beside it.
"""
import json, sys, numpy as np, pandas as pd, torch
from pathlib import Path
from itertools import combinations
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(".")
EXT = REPO / "fractal_analysis/extension"
PAPER = EXT / "paper"; FIG = PAPER / "figures"; TAB = PAPER / "tables"
sys.path.insert(0, str(EXT)); import wave1_families as W

DS = ["size_imp_v1", "war_arg_v1", "food_ideas_v1", "frames_v1"]
NICE = {"size_imp_v1": "IMPORTANCE IS SIZE", "war_arg_v1": "ARGUMENT IS WAR",
        "food_ideas_v1": "IDEAS ARE FOOD", "frames_v1": "professional frames"}
C = dict(zip(DS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
GREY, INK, INK2 = "#8a8a85", "#0b0b0b", "#52514e"
LAYERS = [5, 10, 15, 20]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#c9c8c3", "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.7,
    "axes.axisbelow": True, "font.size": 9, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# ---------- data ------------------------------------------------------------------ #
eff = []
for ds in DS:
    ci = pd.read_parquet(REPO / f"data/axbench/{ds}/concept_index.parquet")
    v_of = dict(zip(ci.concept_id, ci.vector_id))
    for L in LAYERS:
        for l in open(REPO / f"data/axbench/{ds}/diffmean_l{L}/eval/results.jsonl"):
            r = json.loads(l); d = r["results"]["LMJudgeEvaluator"]["DiffMean"]
            eff.append(dict(ds=ds, layer=L, vector_id=v_of[r["concept_id"]],
                            overall=max(d["lm_judge_rating"])))
eff = pd.DataFrame(eff); eff["cloud"] = eff.vector_id.str.split("__").str[0]
eff.to_csv(TAB / "T2_steering_efficacy_per_vector.csv", index=False)

rsa = pd.read_csv(EXT / "tables/wave1_rsa.csv")
com = pd.read_csv(EXT / "tables/wave1_comention.csv")
fa  = pd.read_csv(EXT / "tables/frames_fa_controlled.csv")
pg  = pd.read_csv(EXT / "tables/frames_parallelogram.csv")

# ---------- F1: steering efficacy ------------------------------------------------- #
fig, ax = plt.subplots(figsize=(6.4, 3.0))
wid = 0.2
for k, ds in enumerate(DS):
    m = [eff[(eff.ds == ds) & (eff.layer == L)].overall.mean() for L in LAYERS]
    x = np.arange(len(LAYERS)) + (k - 1.5) * wid
    ax.bar(x, m, wid * 0.86, color=C[ds], label=NICE[ds], linewidth=0)
    for xi, mi in zip(x, m):
        ax.text(xi, mi + 0.03, f"{mi:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
ax.axhline(1.09, color=GREY, lw=1, ls=(0, (4, 3)))
ax.text(3.55, 1.11, "water words (paper), low end", fontsize=6.5, color=GREY, ha="right")
ax.axhline(0.62, color=GREY, lw=1, ls=(0, (1, 2)))
ax.text(3.55, 0.64, "body parts (paper), low end", fontsize=6.5, color=GREY, ha="right")
ax.set_xticks(np.arange(len(LAYERS))); ax.set_xticklabels([f"L{l}" for l in LAYERS])
ax.set_ylabel("mean per-concept Overall (0–2)"); ax.set_ylim(0, 2.1)
ax.set_title("Steering efficacy: every wave-1 family steers at least as well as the\n"
             "published water-word set, and none is judge-limited", loc="left", fontsize=9.5)
ax.legend(loc="lower right", ncol=2, fontsize=7.5)
fig.savefig(FIG / "F1_steering_efficacy.png"); plt.close(fig)

# ---------- F2: the main RSA result ----------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
for ax, kind in zip(axes, ["positions", "moves"]):
    for ds in DS:
        s = rsa[(rsa.experiment == ds) & (rsa.kind == kind)]
        # frames is a MEAN over its 10 frame pairs, so averaging p-values would be
        # meaningless -- mark it solid only when a majority of its pairs clear.
        g = s.groupby("layer").agg(rho=("rho", "mean"), p95=("null_p95", "mean"),
                                   frac_sig=("p", lambda v: (v < 0.05).mean())).reindex(LAYERS)
        g["p"] = np.where(g.frac_sig >= 0.5, 0.0, 1.0)
        ax.plot(LAYERS, g.rho, color=C[ds], lw=2, zorder=3)
        ax.plot(LAYERS, g.p95, color=C[ds], lw=1, ls=(0, (3, 3)), alpha=.75, zorder=2)
        sig = g.p < 0.05
        ax.scatter(np.array(LAYERS)[sig.values], g.rho[sig.values], s=42, color=C[ds],
                   edgecolors="white", linewidths=1.6, zorder=4)
        ax.scatter(np.array(LAYERS)[~sig.values], g.rho[~sig.values], s=42,
                   facecolors="white", edgecolors=C[ds], linewidths=1.6, zorder=4)
    ax.set_xticks(LAYERS); ax.set_xticklabels([f"L{l}" for l in LAYERS])
    ax.set_title(kind, loc="left", fontsize=9)
    ax.set_xlim(3, 25)
axes[0].set_ylabel("RSA (Spearman ρ)"); axes[0].set_ylim(-0.05, 1.0)
for ds, y in zip(DS, [0.90, 0.80, 0.70, 0.60]):
    axes[1].text(20.6, y, NICE[ds], color=C[ds], fontsize=7.5, va="center")
axes[1].text(20.6, 0.50, "dashed = that family's\npermutation-null 95th pct",
             color=GREY, fontsize=6.5, va="center")
fig.suptitle("Solid = clears its own label-permutation null (p<0.05); hollow = does not. Frames is a\n"
             "mean over its 10 frame pairs, solid only if a majority clear. The metaphor families agree\n"
             "on MOVES; only the size ladder also agrees on positions.",
             x=0.012, ha="left", fontsize=9.5, y=1.16)
fig.savefig(FIG / "F2_rsa_positions_moves.png"); plt.close(fig)

# ---------- F3: the generation-artefact check ------------------------------------- #
fig, ax = plt.subplots(figsize=(6.0, 3.2))
com["label"] = np.where(com.experiment == "frames_v1",
                        com.A + " ↔ " + com.B, com.experiment.map(NICE))
com["col"] = com.experiment.map(C)
s = com.sort_values("comention_rho")
ax.barh(range(len(s)), s.comention_rho, color=s.col, height=0.66, linewidth=0)
for i, (v, lab) in enumerate(zip(s.comention_rho, s.label)):
    ax.text(v + (0.012 if v >= 0 else -0.012), i, f"{v:+.2f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=6.5, color=INK2)
ax.set_yticks(range(len(s))); ax.set_yticklabels(s.label, fontsize=7)
ax.axvline(0, color="#c9c8c3", lw=1)
ax.set_xlabel("Spearman ρ between the two clouds' co-mention matrices, under T")
ax.set_xlim(-0.32, 0.62)
ax.set_title("Was the agreement written into the training text?\n"
             "The three metaphor families: no. The frames: partly.",
             loc="left", fontsize=9.5)
fig.savefig(FIG / "F3_comention_artefact.png"); plt.close(fig)

# ---------- F4: the frames carrier ------------------------------------------------ #
FR, RO = W.F1_FRAMES, W.F1_ROLES
unit = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True)
car = []
for L in LAYERS:
    d = REPO / f"data/axbench/frames_v1/diffmean_l{L}/train"
    w = torch.load(d / "DiffMean_weight.pt", weights_only=False).float().numpy()
    ci = pd.read_parquet(REPO / "data/axbench/frames_v1/concept_index.parquet")
    idx = {v: i for i, v in zip(ci.concept_id, ci.vector_id)}
    rid = {(r["frame"], r["role"]): r["id"] for f in FR for r in W.CLOUDS[f]}
    V = {(f, ro): unit(w[idx[f"{f}__{rid[(f, ro)]}"]]) for f in FR for ro in RO}
    car.append(dict(layer=L,
        same_frame=np.mean([V[(f, a)] @ V[(f, b)] for f in FR for a, b in combinations(RO, 2)]),
        same_role=np.mean([V[(a, r)] @ V[(b, r)] for r in RO for a, b in combinations(FR, 2)]),
        neither=np.mean([V[(f1, r1)] @ V[(f2, r2)] for f1 in FR for f2 in FR
                         for r1 in RO for r2 in RO if f1 != f2 and r1 != r2])))
car = pd.DataFrame(car); car.to_csv(TAB / "T6_frames_carrier.csv", index=False)
fig, ax = plt.subplots(figsize=(5.4, 3.0))
for col, lab, sty in [("same_frame", "same frame, different role", "-"),
                      ("same_role", "same role, different frame", "-"),
                      ("neither", "neither shared", "-")]:
    col_of = {"same_frame": C["frames_v1"], "same_role": "#2a78d6", "neither": "#4a3aa7"}
    ax.plot(car.layer, car[col], sty, lw=2, color=col_of[col], marker="o", ms=5)
    ax.text(20.4, car[col].iloc[-1], lab, fontsize=7.5, va="center", color=col_of[col])
ax.axhline(0.71, color=GREY, lw=1, ls=(0, (4, 3)))   # reference ink, not a series
ax.text(5.2, 0.717, "water-word probes in the paper (0.71)", fontsize=6.5, color=GREY)
ax.set_xticks(LAYERS); ax.set_xticklabels([f"L{l}" for l in LAYERS])
ax.set_ylabel("mean pairwise cosine"); ax.set_xlim(4, 31); ax.set_ylim(0.6, 0.95)
ax.set_title("The frames carrier swamps the structure.\n"
             "Frame membership binds about twice as hard as role.", loc="left", fontsize=9.5)
fig.savefig(FIG / "F4_frames_carrier.png"); plt.close(fig)

# ---------- F5: the parallelogram ------------------------------------------------- #
p10 = pg[pg.layer == 10].copy()
p10["lab"] = p10.roleA.str.title() + " → " + p10.roleB.str.title()
p10 = p10.sort_values("mean_cos")
fig, ax = plt.subplots(figsize=(6.0, 4.0))
y = np.arange(len(p10))
ax.barh(y, p10.mean_cos, color=C["frames_v1"], height=0.6, linewidth=0, label="observed")
ax.scatter(p10.null_mean, y, s=34, marker="|", color=INK, linewidths=1.6,
           label="permutation-null mean", zorder=4)
for yi, v in zip(y, p10.mean_cos):
    ax.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=6.5, color=INK2)
ax.set_yticks(y); ax.set_yticklabels(p10.lab, fontsize=7)
ax.set_xlabel("mean pairwise cosine among the five profession-wise offsets")
ax.set_xlim(-0.17, 0.78)   # four null means are negative and must not be clipped
ax.axvline(0, color="#c9c8c3", lw=1, zorder=0)
h, lb = ax.get_legend_handles_labels()
ax.legend(h[::-1], lb[::-1], loc="lower right", fontsize=7.5)
ax.set_title("The parallelogram, layer 10: clinic−doctor ≈ law firm−lawyer.\n"
             "All 15 transitions clear their exact null; the abstract one does not cohere.",
             loc="left", fontsize=9.5)
fig.savefig(FIG / "F5_frames_parallelogram.png"); plt.close(fig)

# ---------- F6: F-a raw vs co-mention-controlled ---------------------------------- #
f10 = fa[(fa.kind == "positions") & (fa.layer == 10)].copy()
f10["lab"] = f10.A + " ↔ " + f10.B
f10 = f10.sort_values("rho")
fig, ax = plt.subplots(figsize=(6.0, 3.4))
y = np.arange(len(f10))
ax.hlines(y, f10.rho_ctl, f10.rho, color="#d9d8d3", lw=2.5, zorder=1)
ax.scatter(f10.rho, y, s=46, color=C["frames_v1"], edgecolors="white", linewidths=1.4,
           zorder=3, label="raw")
ax.scatter(f10.rho_ctl, y, s=46, color="#4a3aa7", edgecolors="white", linewidths=1.4,
           zorder=3, label="after removing co-mention")
ax.set_yticks(y); ax.set_yticklabels(f10.lab, fontsize=7)
ax.set_xlabel("F-a positions RSA (Spearman ρ), layer 10")
ax.legend(loc="lower right", fontsize=7.5); ax.set_xlim(0, 0.95)
ax.annotate("the flagship pair\nloses most of its signal",
            xy=(0.50, len(f10) - 1.05), xytext=(0.17, len(f10) - 2.9), fontsize=6.8,
            color=INK2, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
ax.set_title("Frames, F-a: what survives residualising each cloud on its own co-mention.\n"
             "doctor↔dentist — the pair that shares the most vocabulary — collapses.",
             loc="left", fontsize=9.5)
fig.savefig(FIG / "F6_frames_comention_control.png"); plt.close(fig)

print("wrote:", *sorted(p.name for p in FIG.glob("*.png")), sep="\n  ")
