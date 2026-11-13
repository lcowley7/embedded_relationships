"""Paper-ready versions of the two analogy figures (Section 'Analogies').

  analogy_rsa.{pdf,png}       positions + moves RSA across layers, all four families
  analogy_steering.{pdf,png}  mean steering Overall per family and layer

Styling follows make_temp_rsa_appendix.py (temp_rsa_by_layer), so the analogy figure reads
as a sibling of the temperature one. No titles or claims inside the figures -- those live in
the LaTeX captions. Reads the tables written by wave1_rsa.py and build_figures.py (T2).
Run: PY fractal_analysis/extension/paper/build_paper_figures_analogy.py
"""
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EXT = Path("./fractal_analysis/extension")
FIG = EXT / "paper/figures"
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


# ---------------------------------------------------------------- RSA --------------------
rsa = pd.read_csv(EXT / "tables/wave1_rsa.csv")
x = np.arange(len(LAYERS))
# Each panel gets its own y-range: move nulls sit at 0.3-0.8 (moves sharing an endpoint resemble
# each other), so on a shared 0-1 axis every move line is squeezed into the top band. Positions and
# moves rho are not comparable in any case, because their nulls differ.
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=False)
ranges = {}
for ax, kind, title in zip(axes, ["positions", "moves"], ["(a) Positions", "(b) Moves"]):
    ranges[kind] = [np.inf, -np.inf]
    for ds in DS:
        # frames = mean over its 10 frame pairs; a marker is filled only if most pairs clear
        g = (rsa[(rsa.experiment == ds) & (rsa.kind == kind)]
             .groupby("layer").agg(rho=("rho", "mean"), p95=("null_p95", "mean"),
                                   sig=("p", lambda v: (v < .05).mean() > .5))
             .reindex(LAYERS))
        c, sig = COL[ds], g.sig.values.astype(bool)
        ranges[kind] = [min(ranges[kind][0], g.rho.min(), g.p95.min()), max(ranges[kind][1], g.rho.max(), g.p95.max())]
        # small horizontal dodge per family so near-equal values (e.g. size and war moves at
        # layer 15) do not hide each other's markers
        xd = x + (DS.index(ds) - 1.5) * 0.06
        ax.plot(xd, g.p95, color=c, lw=1, ls=(0, (3, 2)), alpha=.8, zorder=3)
        ax.plot(xd, g.rho, color=c, lw=2, zorder=4, clip_on=False)
        ax.plot(xd[sig], g.rho.values[sig], ls="none", marker="o", ms=6, mfc=c, mec="white",
                mew=1.0, zorder=5, clip_on=False)
        ax.plot(xd[~sig], g.rho.values[~sig], ls="none", marker="o", ms=6, mfc="white", mec=c,
                mew=1.6, zorder=5, clip_on=False)
    ax.set_xticks(x); ax.set_xticklabels(LAYERS); ax.set_xlim(-.4, len(LAYERS) - .6)
    ax.set_xlabel("Layer"); ax.set_title(title, loc="left", pad=6); tidy(ax)
axes[0].set_ylim(0, 1.0); axes[0].set_ylabel("RSA (Spearman $\\rho$)")
# moves: pad the plotted range by 0.03 and snap outward to the 0.05 grid
lo, hi = ranges["moves"]
axes[1].set_ylim(np.floor((lo - .03) * 20) / 20, np.ceil((hi + .03) * 20) / 20)
axes[1].set_ylabel("RSA (Spearman $\\rho$)")

handles = [Line2D([], [], color=COL[d], lw=2, marker="o", ms=5, mec=COL[d]) for d in DS]
handles += [Line2D([], [], color=MUTED, lw=1, ls=(0, (3, 2))),
            Line2D([], [], ls="none", marker="o", ms=5, mfc="white", mec=MUTED, mew=1.4)]
labels = [NAME[d] for d in DS] + ["null, 95th pct.", "$p \\geq 0.05$"]
# fixed margins, not tight_layout: a figure-level legend makes tight_layout squeeze the
# panels against the edges and leave a gap in the middle
fig.subplots_adjust(left=.08, right=.99, bottom=.17, top=.80, wspace=.26)
fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, handlelength=1.8,
           columnspacing=1.2, bbox_to_anchor=(.53, 1.0))
save(fig, "analogy_rsa")

# ---------------------------------------------------------------- steering ---------------
eff = pd.read_csv(EXT / "paper/tables/T2_steering_efficacy_per_vector.csv")
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
