"""appendix_figures.py -- the three appendix figures.

  outputs/figures/temp_rsa_by_layer_moves.{pdf,png}   Figure 15: temperature move RSA at all nine layers
  outputs/figures/temp_sweep_moves.{pdf,png}          Figure 16: efficacy sweep, temperature moves
  outputs/figures/analogy_sweep_moves.{pdf,png}       Figure 17: efficacy sweep, analogy moves

Reads move_direction_stats_by_layer.csv, temp_sweep.csv, analogy_sweep.csv and analogy_sweep_frames.csv
from outputs/tables/. Computes nothing new.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root
from relgeom.paths import TABLES, FIGURES, ensure
FIG = ensure(FIGURES)
SPEC = SWEEP = TABLES
LAYERS = [0, 1, 2, 3, 4, 5, 10, 15, 20]
FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
MIN_MOVES = 5
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
C1, C2 = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})
star = lambda p: "" if p != p else ("***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote", FIG / f"{name}.pdf/.png")


# ---------------------------------------------------------------- layer sweep -------------
mv = pd.read_csv(SPEC / "move_direction_stats_by_layer.csv")
x = np.arange(len(LAYERS))
fig, ax = plt.subplots(figsize=(4.2, 2.6))
ax.axvspan(LAYERS.index(10) - .38, LAYERS.index(10) + .38, color="#e2e0d6", zorder=0)
for unit, col in (("C", C1), ("F", C2)):
    g = mv[mv.unit == unit].set_index("layer").reindex(LAYERS)
    sig = (g.p_word < .05).values
    ax.plot(x, g.null_p95, color=col, lw=1, ls=(0, (3, 2)), alpha=.8, zorder=3)
    ax.plot(x, g.rho, color=col, lw=2, zorder=4)
    ax.plot(x[sig], g.rho.values[sig], ls="none", marker="o", ms=6, mfc=col, mec="white", mew=1, zorder=5, label=f"°{unit}")
    ax.plot(x[~sig], g.rho.values[~sig], ls="none", marker="o", ms=6, mfc="white", mec=col, mew=1.6, zorder=5)
ax.set_xticks(x); ax.set_xticklabels(LAYERS); ax.set_xlabel("Layer of gemma-2-2b-it")
ax.set_ylabel("Move RSA (Spearman $\\rho$)"); ax.set_ylim(.3, .75)
ax.grid(axis="y", color=GRID, lw=.6, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(loc="lower right", frameon=False, ncol=2)
save(fig, "temp_rsa_by_layer_moves")

# ---------------------------------------------------------------- temperature sweep -------
df = pd.read_csv(SPEC / "temp_sweep.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.9), sharey=True)
for ax, unit in zip(axes, ("C", "F")):
    M = np.full((len(LAYERS), len(FRACS)), np.nan)
    for i, L in enumerate(LAYERS):
        for j, f in enumerate(FRACS):
            r = df[(df.layer == L) & (df.unit == unit) & (df.f == f)].iloc[0]
            if r.n_moves < MIN_MOVES:
                ax.text(j, i, "--", ha="center", va="center", fontsize=7, color="0.35"); continue
            M[i, j] = r.move_rsa
            ax.text(j, i, f"{r.move_rsa:.2f}\n{star(r.move_p)} ({int(r.n_moves)})", ha="center", va="center", fontsize=6)
    im = ax.imshow(M, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(FRACS))); ax.set_xticklabels([f"{f:.1f}" for f in FRACS])
    ax.set_xlabel("drop fraction $f$"); ax.set_title(f"Moves, °{unit}", loc="left")
axes[0].set_yticks(range(len(LAYERS))); axes[0].set_yticklabels(LAYERS); axes[0].set_ylabel("layer")
fig.colorbar(im, ax=axes, shrink=.7, label=r"Spearman $\rho$")
save(fig, "temp_sweep_moves")

# ---------------------------------------------------------------- analogy sweep -----------
an = pd.read_csv(SWEEP / "analogy_sweep.csv"); fr = pd.read_csv(SWEEP / "analogy_sweep_frames.csv")
FAM = [("size_imp_v1", "Importance is size"), ("war_arg_v1", "Argument is war"),
       ("food_ideas_v1", "Ideas are food"), ("frames_v1", "Professional frames")]
AL = sorted(an.layer.unique())
rows = [(ds, nm, L) for ds, nm in FAM for L in AL]
fig, ax = plt.subplots(figsize=(4.6, 6.2))
M = np.full((len(rows), len(FRACS)), np.nan)
for i, (ds, _, L) in enumerate(rows):
    for j, f in enumerate(FRACS):
        if ds == "frames_v1":
            r = fr[(fr.layer == L) & (fr.f == f) & (fr.kind == "moves")].iloc[0]
            v, lab = (r.mean_rho, f"{r.mean_rho:.2f}\n{int(r.n_sig)}/10") if r.reportable else (np.nan, "--")
        else:
            r = an[(an.experiment == ds) & (an.layer == L) & (an.f == f) & (an.kind == "moves")].iloc[0]
            v, lab = (r.rho, f"{r.rho:.2f}\n{star(r.p)} ({int(r.n_items)})") if r.reportable else (np.nan, "--")
        M[i, j] = v
        ax.text(j, i, lab, ha="center", va="center", fontsize=6, color="0.35" if lab == "--" else "black")
im = ax.imshow(M, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
for y in np.arange(len(AL), len(rows), len(AL)) - .5:
    ax.axhline(y, color="white", lw=3)
ax.set_xticks(range(len(FRACS))); ax.set_xticklabels([f"{f:.1f}" for f in FRACS]); ax.set_xlabel("drop fraction $f$")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{nm}, layer {L}" for _, nm, L in rows], fontsize=7)
fig.colorbar(im, ax=ax, shrink=.5, label=r"Spearman $\rho$")
save(fig, "analogy_sweep_moves")
