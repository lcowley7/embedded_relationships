"""Variant of build_paper_models.py figure without the WordNet panel (2026-10-05).
Top row is temperature (positions C, moves C, moves F); bottom row the three analogy moves.
Writes figures/models_verification_nowordnet.{pdf,png}; the original figure is untouched."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

NP = Path(".")
PAPER = NP / "fractal_analysis/extension/paper"
FIG, TEX = PAPER / "figures", PAPER / "latex/generated"
TEX.mkdir(parents=True, exist_ok=True)

MODELS = ["Gemma-2-2B", "OLMo-3.1-32B", "Qwen3-32B"]
N_LAYERS = {"Gemma-2-2B": 26, "OLMo-3.1-32B": 64, "Qwen3-32B": 64}
COL = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a"]))      # validated: dataviz validate_palette.js
MARK = dict(zip(MODELS, ["o", "s", "^"]))                         # secondary encoding (green < 3:1 contrast)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})

S = pd.read_csv(NP / "bigmodel/tables/cross_model_summary.csv")
S["depth"] = S.layer / S.model.map(N_LAYERS)


def tidy(ax):
    ax.grid(axis="y", color=GRID, lw=.6, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)



# (experiment, analysis, panel title). WordNet's reference line is the best of 200 random
# spanning trees; every other panel's is the permutation null's 95th percentile.
PANELS = [("temperature", "positions_C", "(a) Temperature positions, °C"),
          ("temperature", "moves_C", "(b) Temperature moves, °C"),
          ("temperature", "moves_F", "(c) Temperature moves, °F"),
          ("size_imp_v1", "moves", "(d) Importance is size, moves"),
          ("war_arg_v1", "moves", "(e) Argument is war, moves"),
          ("food_ideas_v1", "moves", "(f) Ideas are food, moves")]

fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5))
for ax, (exp, an, title) in zip(axes.ravel(), PANELS):
    lo, hi = np.inf, -np.inf
    for i, m in enumerate(MODELS):
        g = S[(S.experiment == exp) & (S.analysis == an) & (S.model == m)].sort_values("depth")
        ref = g.tree_max if exp == "wordnet" else g.null_p95
        sig = (g.beats_all_trees.astype(bool) if exp == "wordnet" else g.p < .05).values
        x = g.depth.values + (i - 1) * 0.006          # tiny dodge so coincident markers show
        c = COL[m]
        ax.plot(x, ref, color=c, lw=1, ls=(0, (3, 2)), alpha=.75, zorder=3)
        ax.plot(x, g.rho, color=c, lw=1.6, zorder=4)
        ax.plot(x[sig], g.rho.values[sig], ls="none", marker=MARK[m], ms=5, mfc=c, mec="white",
                mew=.8, zorder=5)
        ax.plot(x[~sig], g.rho.values[~sig], ls="none", marker=MARK[m], ms=5, mfc="white", mec=c,
                mew=1.3, zorder=5)
        lo, hi = min(lo, g.rho.min(), ref.min()), max(hi, g.rho.max(), ref.max())
    ax.set_ylim(np.floor((lo - .03) * 20) / 20, np.ceil((hi + .03) * 20) / 20)
    ax.set_xlim(-.03, .81); ax.set_xticks([0, .2, .4, .6, .8])
    ax.set_title(title, loc="left", pad=5); tidy(ax)
for ax in axes[1]:
    ax.set_xlabel("Relative depth (layer / layers)")
for ax in axes[:, 0]:
    ax.set_ylabel("RSA (Spearman $\\rho$)")

handles = [Line2D([], [], color=COL[m], lw=1.6, marker=MARK[m], ms=5, mec="white", mew=.8) for m in MODELS]
handles += [Line2D([], [], color=MUTED, lw=1, ls=(0, (3, 2))),
            Line2D([], [], ls="none", marker="o", ms=5, mfc="white", mec=MUTED, mew=1.3)]
labels = MODELS + ["null (dashed)", "not significant (hollow)"]
fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(.5, 1.0),
           handlelength=2.2, columnspacing=1.4)
fig.subplots_adjust(left=.08, right=.99, bottom=.10, top=.88, wspace=.28, hspace=.42)
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"models_verification_nowordnet.{ext}", dpi=300)
plt.close(fig)
print("wrote", FIG / "models_verification_nowordnet.pdf/.png")
