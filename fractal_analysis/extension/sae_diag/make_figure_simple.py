"""make_figure_simple.py -- simplified appendix figure (2026-10-07): analyses 1 and 3 only.
 (a) move RSA on the residual steering vectors, on the full SAE code, and after deleting every
     feature active in both domains. Grey bar = permutation-null mean..95th percentile.
 (b) top-20 feature overlap (Jaccard) of corresponding vs non-corresponding cross-domain pairs.
Reads tables/act_rsa.csv and tables/act_overlap.csv; writes figs/sae_simple_L{L}.{png,pdf}."""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); T = f"{HERE}/tables"
L = int(sys.argv[1]) if len(sys.argv) > 1 else 10
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "pdf.fonttype": 42})
FAM = [("water-tempC", "Water ↔ °C"), ("water-tempF", "Water ↔ °F"), ("size-importance", "Importance is size"),
       ("war-argument", "Argument is war"), ("food-ideas", "Ideas are food")]
SPACES = [("residual_stored", "steering vectors (main text)", "#CFCECE", "o"),
          ("sae_all", "SAE code, all features", "#0F4D92", "s"),
          ("sae_disjoint_support", "SAE code, shared features deleted", "#B64342", "D")]
r = pd.read_csv(f"{T}/act_rsa.csv"); r = r[r.layer == L]
o = pd.read_csv(f"{T}/act_overlap.csv"); o = o[(o.layer == L) & (o.k == 20)].set_index("family")
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.5, 1], "wspace": .18})
ax = axes[0]; off = np.linspace(-.22, .22, len(SPACES))
for i, (fam, _) in enumerate(FAM):
    for j, (sp, lab, c, m) in enumerate(SPACES):
        row = r[(r.family == fam) & (r.space == sp)].iloc[0]; y = i + off[j]
        ax.plot([row.null_mean, row.null_p95], [y, y], color="#999999", lw=2, alpha=.55, solid_capstyle="butt")
        ax.scatter(row.rho, y, color=c, marker=m, s=24, edgecolor="black", linewidth=.4, zorder=3,
                   label=lab if i == 0 else None)
ax.set_yticks(range(len(FAM))); ax.set_yticklabels([l for _, l in FAM]); ax.invert_yaxis()
ax.set_xlim(0, 1); ax.set_xlabel("Move RSA (Spearman ρ)")
ax.set_title(f"(a) Move agreement, layer {L}", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(.5, -.2), ncol=2, fontsize=7, handletextpad=.2, columnspacing=.8)
ax = axes[1]; y = np.arange(len(FAM)); keys = [f for f, _ in FAM]
ax.barh(y - .19, o.loc[keys, "jacc_matched"], .36, color="#0F4D92", label="corresponding pairs")
ax.barh(y + .19, o.loc[keys, "jacc_null"], .36, color="#CFCECE", label="non-corresponding pairs")
ax.set_yticks(y); ax.set_yticklabels([]); ax.invert_yaxis()
ax.set_xlabel("Top-20 feature overlap (Jaccard)"); ax.set_title("(b) Shared features per pair", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(.5, -.2), ncol=1, fontsize=7)
for ext in ("png", "pdf"):
    fig.savefig(f"{HERE}/figs/sae_simple_L{L}.{ext}", dpi=300, bbox_inches="tight")
print(f"{HERE}/figs/sae_simple_L{L}.png")
