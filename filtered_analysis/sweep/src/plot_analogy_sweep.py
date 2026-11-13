"""plot_analogy_sweep.py -- figure for the analogy efficacy sweep, from tables/analogy_sweep*.csv.

Same form as temp_sweep.png (heat grid, RdYlBu_r on [0, 1], stars = the cell's own permutation
null) so the two read together in the appendix. Rows are family x layer; columns are the drop
fraction. Metaphor cells give rho, stars and the surviving pair count; frames cells give the
mean rho over the 10 profession pairs and how many of the 10 clear p < 0.05. Cells with fewer
than MIN_ITEMS surviving items are shown as --.
With LT_MODEL set it plots that model's sweep, at the layers the sweep actually ran.
Run: PY filtered_analysis/sweep/src/plot_analogy_sweep.py
"""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "."
sys.path.insert(0, f"{REPO}")
import lt_model as LT   # noqa: E402

BASE = f"{REPO}/filtered_analysis/sweep"
TABLES, FIGS = str(LT.out_dir(f"{BASE}/tables")), str(LT.out_dir(f"{BASE}/figs"))
FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
FAM = [("size_imp_v1", "Importance is size"), ("war_arg_v1", "Argument is war"),
       ("food_ideas_v1", "Ideas are food"), ("frames_v1", "Professional frames")]

df = pd.read_csv(f"{TABLES}/analogy_sweep.csv")
fr = pd.read_csv(f"{TABLES}/analogy_sweep_frames.csv")
LAYERS = sorted(df.layer.unique())


def star(p):
    return "" if p != p else ("***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "")


def cell(ds, L, f, kind):
    """(value, label) for one grid cell; value NaN when not reportable."""
    if ds == "frames_v1":
        r = fr[(fr.layer == L) & (fr.f == f) & (fr.kind == kind)].iloc[0]
        if not r.reportable:
            return np.nan, "--"
        return r.mean_rho, f"{r.mean_rho:.2f}\n{int(r.n_sig)}/10"
    r = df[(df.experiment == ds) & (df.layer == L) & (df.f == f) & (df.kind == kind)].iloc[0]
    if not r.reportable:
        return np.nan, "--"
    return r.rho, f"{r.rho:.2f}\n{star(r.p)} ({int(r.n_items)})"


rows = [(ds, name, L) for ds, name in FAM for L in LAYERS]
fig, axes = plt.subplots(1, 2, figsize=(8.6, 7.4), sharey=True)
for ax, kind, title in zip(axes, ["positions", "moves"], ["(a) Positions", "(b) Moves"]):
    M = np.full((len(rows), len(FRACS)), np.nan)
    for i, (ds, _, L) in enumerate(rows):
        for j, f in enumerate(FRACS):
            v, lab = cell(ds, L, f, kind)
            M[i, j] = v
            ax.text(j, i, lab, ha="center", va="center", fontsize=6.2,
                    color="0.35" if lab == "--" else "black")
    im = ax.imshow(M, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
    for y in np.arange(len(LAYERS), len(rows), len(LAYERS)) - .5:
        ax.axhline(y, color="white", lw=3)
    ax.set_xticks(range(len(FRACS))); ax.set_xticklabels([f"{f:.1f}" for f in FRACS])
    ax.set_xlabel("drop fraction $f$"); ax.set_title(title, loc="left", fontsize=10)
axes[0].set_yticks(range(len(rows)))
axes[0].set_yticklabels([f"{name}, layer {L}" for _, name, L in rows], fontsize=7.5)
fig.colorbar(im, ax=axes, shrink=.5, label=r"Spearman $\rho$")
for ext in ("png", "pdf"):
    fig.savefig(f"{FIGS}/analogy_sweep.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"wrote {FIGS}/analogy_sweep.png/.pdf")
