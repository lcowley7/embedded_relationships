"""replot_spec.py SPEC -- replot_temp.py reading/writing out/SPEC (the paper figure is this one)."""
"""replot_temp.py -- regenerate the temperature sweep figure and table from temp_sweep.csv
with a corrected degeneracy threshold. No permutations are re-run; only the gate and the
rendering change.

MIN_MOVES = 5: a move RDM over M moves has C(M,2) entries, so M=3 gives 3 and M=4 gives 6 --
Spearman on that is unconstrained (one such cell returned exactly +1.00). At M>=5 the
word-permutation null still discriminates: the M=7 cells return p from 0.006 to 0.68.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
BASE = "./method_audit_2026-09-16/ladders_18/out/" + sys.argv[1]
MIN_MOVES = 5
FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
LAYERS = [0, 1, 2, 3, 4, 5, 10, 15, 20]

df = pd.read_csv(f"{BASE}/temp_sweep.csv")
df["move_reportable"] = df.n_moves >= MIN_MOVES
df.to_csv(f"{BASE}/temp_sweep.csv", index=False)
print(f"suppressed {int((~df.move_reportable).sum())} of {len(df)} move cells (M < {MIN_MOVES})")

def star(p):
    return "" if p != p else ("***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "")

PANELS = [("node_rsa", "node_p", None, "Positions"),
          ("move_rsa", "move_p", "move_reportable", "Moves")]
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
for r, (rc, pc, gate, tname) in enumerate(PANELS):
    for c, unit in enumerate(("C", "F")):
        ax = axes[r, c]
        M = np.full((len(LAYERS), len(FRACS)), np.nan)
        for i, L in enumerate(LAYERS):
            for j, f in enumerate(FRACS):
                row = df[(df.layer == L) & (df.unit == unit) & (df.f == f)].iloc[0]
                if gate and not row[gate]:
                    continue
                M[i, j] = row[rc]
        im = ax.imshow(M, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
        for i, L in enumerate(LAYERS):
            for j, f in enumerate(FRACS):
                row = df[(df.layer == L) & (df.unit == unit) & (df.f == f)].iloc[0]
                if gate and not row[gate]:
                    ax.text(j, i, "--", ha="center", va="center", fontsize=8, color="0.35")
                    continue
                lab = f"{row[rc]:.2f}\n{star(row[pc])}"
                if gate:
                    lab = f"{row[rc]:.2f}\n{star(row[pc])} ({int(row.n_moves)})"
                ax.text(j, i, lab, ha="center", va="center", fontsize=6.2, color="black")
        ax.set_xticks(range(len(FRACS))); ax.set_xticklabels([f"{f:.1f}" for f in FRACS])
        ax.set_yticks(range(len(LAYERS))); ax.set_yticklabels([str(L) for L in LAYERS])
        ax.set_title(f"{tname}, $^\\circ${unit}", fontsize=10)
        if c == 0: ax.set_ylabel("layer")
        if r == 1: ax.set_xlabel("drop fraction $f$")
fig.colorbar(im, ax=axes, shrink=.6, label=r"Spearman $\rho$")
fig.suptitle("Temperature sweep at all nine trained layers. Stars are each cell's own "
             "permutation null (*** $p<.001$, ** $p<.01$, * $p<.05$).\nWater words remaining: "
             "18, 17, 15, 13, 11, 9; surviving move count in parentheses. Move cells with "
             f"fewer than {MIN_MOVES} moves are shown as --.", fontsize=9)
fig.savefig(f"{BASE}/temp_sweep.png", dpi=140, bbox_inches="tight")
plt.close(fig)

def tex_star(p):
    return "" if p != p else (r"$^{***}$" if p < .001 else r"$^{**}$" if p < .01
                              else r"$^{*}$" if p < .05 else "")
L = [r"\begin{table}[t]", r"\setlength{\tabcolsep}{4pt}", r"\small", r"\centering",
     r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
     r"Layer & Statistic & \multicolumn{6}{c}{drop fraction $f$} \\", r"\cmidrule(l){3-8}",
     " & & " + " & ".join(f"${f:.1f}$" for f in FRACS) + r" \\", r"\midrule",
     r"\multicolumn{2}{@{}l}{water words remaining} & 18 & 17 & 15 & 13 & 11 & 9 \\", r"\midrule"]
for li, lay in enumerate(LAYERS):
    first = True
    for unit in ("C", "F"):
        for kind, rc, pc, gate in (("nodes", "node_rsa", "node_p", None),
                                   ("moves", "move_rsa", "move_p", "move_reportable")):
            s = df[(df.layer == lay) & (df.unit == unit)].set_index("f")
            cells = []
            for f in FRACS:
                row = s.loc[f]
                cells.append("--" if (gate and not row[gate])
                             else f"{row[rc]:+.2f}{tex_star(row[pc])}")
            L.append((rf"{lay} & " if first else " & ") + rf"$^\circ${unit} {kind}"
                     + " & " + " & ".join(cells) + r" \\")
            first = False
    if li < len(LAYERS) - 1:
        L.append(r"\addlinespace")
L += [r"\bottomrule", r"\end{tabular}",
      r"\caption{Temperature geometry under the efficacy sweep, at all nine trained layers."
      r" Items are the 18 water words, each scored by the minimum of its own Overall and that"
      r" of the numeric level it is paired with; the sweep thins the water words, not the"
      r" temperature ladder. Stars mark each cell's own permutation null ($^{***}p<.001$,"
      r" $^{**}p<.01$, $^{*}p<.05$; 5000 draws, $p_{\min}=2\times10^{-4}$)."
      r" The number of \emph{moves} that survive is not a function of $f$ alone: the eight"
      r" adjectival words each carry six moves, the six noun-ladder words three, and"
      r" \texttt{dew}, \texttt{fog}, \texttt{frost} and \texttt{hail} belong to no progression"
      r" and carry none, so which words are dropped matters as much as how many. Cells with"
      r" fewer than 5 surviving moves are degenerate and shown as --.}",
      r"\label{tab:temp_sweep}", r"\end{table}"]
open(f"{BASE}/temp_sweep.tex", "w").write("\n".join(L) + "\n")
print("wrote figs/temp_sweep.png and tables/temp_sweep.tex")
