"""main_rsa_figures.py -- the two main-text RSA figures.

  outputs/figures/analogy_rsa_moves.{pdf,png}           Figure 3: analogy moves by layer
  outputs/figures/models_verification_moves.{pdf,png}   Figure 4: temperature moves (C, F) + the three
                                                        metaphor moves, three models

Reads outputs/tables/analogy_rsa.csv and cross_model_summary.csv. Computes nothing new.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root
from relgeom.paths import TABLES, FIGURES, ensure
FIG = ensure(FIGURES)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})


def tidy(ax):
    ax.grid(axis="y", color=GRID, lw=.6, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


def snap(ax, lo, hi):
    ax.set_ylim(np.floor((lo - .03) * 20) / 20, np.ceil((hi + .03) * 20) / 20)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote", FIG / f"{name}.pdf/.png")


# ---------------------------------------------------------------- analogy moves, one panel --
LAYERS = [5, 10, 15, 20]
DS = ["size_imp_v1", "war_arg_v1", "food_ideas_v1", "frames_v1"]
NAME = {"size_imp_v1": "Importance is size", "war_arg_v1": "Argument is war",
        "food_ideas_v1": "Ideas are food", "frames_v1": "Professional frames"}
COL = dict(zip(DS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
rsa = pd.read_csv(TABLES / "analogy_rsa.csv")
x = np.arange(len(LAYERS))
fig, ax = plt.subplots(figsize=(3.3, 2.35))
lo, hi = np.inf, -np.inf
for k, ds in enumerate(DS):
    # frames = mean over its 10 profession pairs; filled only if most pairs clear
    g = (rsa[(rsa.experiment == ds) & (rsa.kind == "moves")]
         .groupby("layer").agg(rho=("rho", "mean"), p95=("null_p95", "mean"),
                               sig=("p", lambda v: (v < .05).mean() > .5)).reindex(LAYERS))
    c, sig, xd = COL[ds], g.sig.values.astype(bool), x + (k - 1.5) * 0.06
    lo, hi = min(lo, g.rho.min(), g.p95.min()), max(hi, g.rho.max(), g.p95.max())
    ax.plot(xd, g.p95, color=c, lw=1, ls=(0, (3, 2)), alpha=.8, zorder=3)
    ax.plot(xd, g.rho, color=c, lw=1.8, zorder=4, clip_on=False)
    ax.plot(xd[sig], g.rho.values[sig], ls="none", marker="o", ms=5, mfc=c, mec="white", mew=.9, zorder=5, clip_on=False)
    ax.plot(xd[~sig], g.rho.values[~sig], ls="none", marker="o", ms=5, mfc="white", mec=c, mew=1.4, zorder=5, clip_on=False)
ax.set_xticks(x); ax.set_xticklabels(LAYERS); ax.set_xlim(-.4, len(LAYERS) - .6)
ax.set_xlabel("Layer"); ax.set_ylabel("Move RSA (Spearman $\\rho$)"); snap(ax, lo, hi); tidy(ax)
handles = [Line2D([], [], color=COL[d], lw=1.8, marker="o", ms=4, mec=COL[d]) for d in DS]
handles += [Line2D([], [], color=MUTED, lw=1, ls=(0, (3, 2))),
            Line2D([], [], ls="none", marker="o", ms=4, mfc="white", mec=MUTED, mew=1.3)]
labels = [NAME[d] for d in DS] + ["null, 95th pct.", "$p \\geq 0.05$"]
ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False,
          handlelength=1.8, columnspacing=1.0, labelspacing=.3)
save(fig, "analogy_rsa_moves")

# ---------------------------------------------------------------- models, one row ----------
MODELS = ["Gemma-2-2B", "OLMo-3.1-32B", "Qwen3-32B"]
N_LAYERS = {"Gemma-2-2B": 26, "OLMo-3.1-32B": 64, "Qwen3-32B": 64}
MCOL = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a"]))
MARK = dict(zip(MODELS, ["o", "s", "^"]))
S = pd.read_csv(TABLES / "cross_model_summary.csv")
S["depth"] = S.layer / S.model.map(N_LAYERS)
PANELS = [("temperature", "moves_C", "(a) Temperature, °C"),
          ("temperature", "moves_F", "(b) Temperature, °F"),
          ("size_imp_v1", "moves", "(c) Importance is size"),
          ("war_arg_v1", "moves", "(d) Argument is war"),
          ("food_ideas_v1", "moves", "(e) Ideas are food")]
fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.85))
for ax, (exp, an, title) in zip(axes, PANELS):
    lo, hi = np.inf, -np.inf
    for i, m in enumerate(MODELS):
        g = S[(S.experiment == exp) & (S.analysis == an) & (S.model == m)].sort_values("depth")
        sig = (g.p < .05).values; xd = g.depth.values + (i - 1) * 0.006; c = MCOL[m]
        ax.plot(xd, g.null_p95, color=c, lw=.9, ls=(0, (3, 2)), alpha=.75, zorder=3)
        ax.plot(xd, g.rho, color=c, lw=1.4, zorder=4)
        ax.plot(xd[sig], g.rho.values[sig], ls="none", marker=MARK[m], ms=3.8, mfc=c, mec="white", mew=.6, zorder=5)
        ax.plot(xd[~sig], g.rho.values[~sig], ls="none", marker=MARK[m], ms=3.8, mfc="white", mec=c, mew=1.1, zorder=5)
        lo, hi = min(lo, g.rho.min(), g.null_p95.min()), max(hi, g.rho.max(), g.null_p95.max())
    snap(ax, lo, hi); ax.set_xlim(-.03, .81); ax.set_xticks([0, .4, .8])
    ax.set_title(title, loc="left", pad=3); ax.set_xlabel("Relative depth"); tidy(ax)
axes[0].set_ylabel("Move RSA ($\\rho$)")
handles = [Line2D([], [], color=MCOL[m], lw=1.4, marker=MARK[m], ms=4, mec="white", mew=.6) for m in MODELS]
handles += [Line2D([], [], color=MUTED, lw=.9, ls=(0, (3, 2))),
            Line2D([], [], ls="none", marker="o", ms=4, mfc="white", mec=MUTED, mew=1.1)]
fig.legend(handles, MODELS + ["null, 95th pct.", "$p \\geq 0.05$"], loc="upper center", ncol=5,
           frameon=False, bbox_to_anchor=(.5, 1.08), handlelength=2, columnspacing=1.3)
fig.subplots_adjust(left=.06, right=.995, bottom=.22, top=.83, wspace=.38)
save(fig, "models_verification_moves")
