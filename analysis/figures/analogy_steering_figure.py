"""analogy_steering_figure.py -- Figure 7: mean steering Overall per analogy domain and layer.

Reads outputs/tables/T2_steering_efficacy_per_vector.csv (analysis/steering/analogy_judge_scores.py).
Writes outputs/figures/analogy_steering.{pdf,png}.
"""
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root
from relgeom.paths import TABLES, FIGURES, ensure
FIG = ensure(FIGURES)
LAYERS = [5, 10, 15, 20]
DS = ["size_imp_v1", "war_arg_v1", "food_ideas_v1", "frames_v1"]
NAME = {"size_imp_v1": "Importance is size", "war_arg_v1": "Argument is war",
        "food_ideas_v1": "Ideas are food", "frames_v1": "Professional frames"}
COL = dict(zip(DS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})


def tidy(ax):
    ax.grid(axis="y", color=GRID, lw=.6, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote", FIG / f"{name}.pdf/.png")


x = np.arange(len(LAYERS))
# ---------------------------------------------------------------- steering ---------------
eff = pd.read_csv(TABLES / "T2_steering_efficacy_per_vector.csv")
fig, ax = plt.subplots(figsize=(3.4, 2.3))
w = 0.2
for k, ds in enumerate(DS):
    m = [eff[(eff.ds == ds) & (eff.layer == L)].overall.mean() for L in LAYERS]
    ax.bar(x + (k - 1.5) * w, m, w * 0.88, color=COL[ds], label=NAME[ds], lw=0, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(LAYERS); ax.set_xlabel("Layer")
ax.set_ylabel("Mean steering score (0–2)"); ax.set_ylim(0, 2); tidy(ax)
ax.legend(loc="lower center", ncol=2, frameon=False, handlelength=1.2, columnspacing=1.0,
          bbox_to_anchor=(.5, 1.0))
save(fig, "analogy_steering")
