"""make_figure_overlap.py -- appendix figure for the SAE feature-overlap analysis only (2026-10-07).
 (a) for each water word, the number of top-20 SAE features it shares with its matched Celsius
     temperature, coloured by which feature is shared.
 (b) mean top-20 overlap (Jaccard), matched vs unmatched water-word/temperature pairs (°C).
Layer 10. Reads tables/act_matched_pair_shared.csv, act_overlap.csv. Writes figs/sae_overlap_L10.{png,pdf}."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
HERE = os.path.dirname(os.path.abspath(__file__)); T = f"{HERE}/tables"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "pdf.fonttype": 42})
m = pd.read_csv(f"{T}/act_matched_pair_shared.csv"); m = m[(m.layer == 10) & (m.family == "water-tempC")]
o = pd.read_csv(f"{T}/act_overlap.csv"); o = o[(o.layer == 10) & (o.k == 20) & (o.family == "water-tempC")].iloc[0]
NICE = {"icy_water": "icy water", "cold_water": "cold water", "cool_water": "cool water",
        "roomtemp_water": "room-temp. water", "warm_water": "warm water", "hot_water": "hot water",
        "boiling_water": "boiling water"}
lab, n, feat = [], [], []
for _, r in m.iterrows():
    w, t = [s.strip() for s in r.pair.split("|")]
    lab.append(f"{NICE.get(w, w)} / {t}"); n.append(int(r.n_shared_top20))
    feat.append(str(r.shared).split(":")[0] if r.n_shared_top20 else "")
COL = {"12865": "#3775BA", "15502": "#E08A2E", "": "#CFCECE"}
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1), gridspec_kw={"width_ratios": [1.7, 1], "wspace": .55})
ax = axes[0]; y = np.arange(len(lab))
ax.barh(y, [max(v, .04) for v in n], color=[COL.get(f, "#777777") for f in feat], height=.7)
ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=6.8); ax.invert_yaxis()
ax.set_xlim(0, 1.3); ax.set_xticks([0, 1]); ax.set_xlabel("Top-20 features shared with matched temperature")
ax.set_title("(a) Each water word and its temperature", loc="left")
ax.legend(handles=[Patch(color=COL["12865"], label="“seasonal weather and winter apparel”"),
                   Patch(color=COL["15502"], label="“temperature variations and measurements”"),
                   Patch(color=COL[""], label="none shared")],
          loc="upper center", bbox_to_anchor=(.35, -.16), fontsize=6.5, handlelength=1, ncol=1)
ax = axes[1]
ax.bar([0, 1], [o.jacc_matched, o.jacc_null], color=["#0F4D92", "#CFCECE"], width=.6)
ax.axhline(1 / 39, color="#52514e", lw=.9, ls=(0, (3, 2)))
ax.text(1.35, 1 / 39, "1 shared\nfeature", fontsize=6.5, color="#52514e", va="center")
ax.set_xticks([0, 1]); ax.set_xticklabels(["matched\npairs", "unmatched\npairs"])
ax.set_xlim(-.5, 1.9); ax.set_ylim(0, .03); ax.set_ylabel("Mean top-20 overlap (Jaccard)")
ax.set_title("(b) All pairs", loc="left")
for ext in ("png", "pdf"):
    fig.savefig(f"{HERE}/figs/sae_overlap_L10.{ext}", dpi=300, bbox_inches="tight")
print(f"{HERE}/figs/sae_overlap_L10.png", round(o.jacc_matched, 4), round(o.jacc_null, 4))
