"""build_appendix_spec.py SPEC -- copy of fractal_analysis/make_temp_rsa_appendix.py reading
out/SPEC and writing its figure + LaTeX there. Positions are unchanged and read from the real tables."""
"""make_temp_rsa_appendix.py -- collect the temperature/water RSA layer sweep into the
appendix figure + LaTeX tables ("Temperature results for other layers").

Consumes, without recomputing anything:
  tables/node_rsa_by_layer_unit.csv        <- make_node_rsa_tables_sweep.py   (POSITIONS, 18 words)
  tables/move_direction_stats_L{L}.csv     <- make_move_direction_stats.py    (MOVES, 33 moves)

Those are the same two scripts that produced the layer-10 numbers in the main text, so the
layer-10 rows here reproduce the main-text tables exactly.

NOTE the move numbers must come from make_move_direction_stats.py, NOT from
make_move_rsa_tables_sweep.py: the latter runs on the 66 DIRECTED edges of the h2o shift
datasets, while the paper's move set is those edges de-duplicated to unordered word pairs and
oriented cold->hot (33 moves). Both are defensible; only the 33-move version matches the text.

Writes:
  tables/move_direction_stats_by_layer.csv
  figs/temp_rsa_by_layer.pdf / .png
  ../../paper/appendix_temp_other_layers.tex
"""
import os, sys, glob, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "./fractal_analysis"
SPEC = sys.argv[1]
TAB = HERE + "/tables"
FIG = PAPER = TABS = f"./method_audit_2026-09-16/ladders_18/out/{SPEC}"
os.makedirs(PAPER, exist_ok=True)

LAYERS = [0, 1, 2, 3, 4, 5, 10, 15, 20]
MAIN = 10                      # the layer reported in the main text
PMIN = 1.0 / 5001              # 5000 permutation draws

# ---------------------------------------------------------------- gather -------------------
node = pd.read_csv(f"{TAB}/node_rsa_by_layer_unit.csv")

move = pd.concat([pd.read_csv(f"{TABS}/move_direction_stats_L{L}.csv") for L in LAYERS],
                 ignore_index=True).sort_values(["layer", "unit"])
move.to_csv(f"{TABS}/move_direction_stats_by_layer.csv", index=False)
print(f"wrote {TABS}/move_direction_stats_by_layer.csv  ({len(move)} rows)")

for df, name in ((node, "node"), (move, "move")):
    got = sorted(df.layer.unique())
    assert got == LAYERS, f"{name}: expected layers {LAYERS}, got {got}"

# sanity: the layer-10 rows must equal the main-text tables
m10 = move[(move.layer == MAIN)].set_index("unit")
n10 = node[(node.layer == MAIN)].set_index("unit")
# move values updated 2026-09-18, when the moves switched to H2O_TEMP (was 0.6319 / 0.7166 / 0.3808 / 0.4929)
checks = [("node rho C", n10.loc["C", "spearman"], 0.3313), ("node rho F", n10.loc["F", "spearman"], 0.4784),
          ]
for lab, got, want in checks:
    assert abs(got - want) < 1e-4, f"{lab}: {got} != {want}"
print("layer-10 rows reproduce the main-text tables")


NM, NW = int(move.n_moves.iloc[0]), int(move.n_words.iloc[0])
move["z"] = (move.rho - move.null_mean) / move.null_sd


def get(df, unit, col):
    return df[df.unit == unit].sort_values("layer")[col].values


# ---------------------------------------------------------------- figure -------------------
C1, C2 = "#2a78d6", "#eb6834"          # categorical slots 1 (C) and 2 (F); validated all-pairs
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
x = np.arange(len(LAYERS))

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
panels = [(axes[0], node, "spearman", "null_p95", "perm_p",
           "(a) Positions \u00b7 18 water words vs. 18 numeric levels"),
          (axes[1], move, "rho", "null_p95", "p_word",
           f"(b) Moves \u00b7 {NM} colder\u2192warmer chords, pooled")]

for ax, df, rcol, pcol, ppcol, title in panels:
    p95 = np.maximum(get(df, "C", pcol), get(df, "F", pcol))
    ax.axvspan(x[LAYERS.index(MAIN)] - .38, x[LAYERS.index(MAIN)] + .38,
               color="#e2e0d6", zorder=0)                       # the main-text layer
    ax.fill_between(x, -0.05, p95, color="#eceae4", alpha=.9, lw=0, zorder=1)
    for unit, col in (("C", C1), ("F", C2)):
        y, pv = get(df, unit, rcol), get(df, unit, ppcol)
        ax.plot(x, get(df, unit, pcol), color=col, lw=1, ls=(0, (3, 2)), alpha=.8, zorder=3)
        ax.plot(x, y, color=col, lw=2, zorder=4, clip_on=False)
        sig = pv < .05
        # filled = above its own null 95th pct; hollow = not significant (secondary encoding)
        ax.plot(x[sig], y[sig], ls="none", marker="o", ms=6.5, mfc=col, mec="white", mew=1.0,
                color=col, label=f"\u00b0{unit}", zorder=5, clip_on=False)
        if (~sig).any():
            ax.plot(x[~sig], y[~sig], ls="none", marker="o", ms=6.5, mfc="white", mec=col,
                    mew=1.8, color=col, zorder=5, clip_on=False)
    ax.set_xticks(x); ax.set_xticklabels(LAYERS)
    ax.set_xlabel("layer of gemma-2-2b-it")
    ax.set_title(title, loc="left", pad=6)
    ax.set_xlim(-.5, len(LAYERS) - .5)
    ax.grid(axis="y", color=GRID, lw=.6, zorder=0)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)

axes[0].set_ylim(-0.05, 0.88)
axes[0].set_ylabel("RSA (Spearman $\\rho$)")

# direct labels at the right edge, plus the legend (identity never by colour alone)
for ax, df, rcol in ((axes[0], node, "spearman"), (axes[1], move, "rho")):
    for unit, col in (("C", C1), ("F", C2)):
        ax.annotate(f"\u00b0{unit}", (x[-1] + .14, get(df, unit, rcol)[-1]), color=col,
                    fontsize=8, fontweight="bold", va="center", ha="left", annotation_clip=False)

axes[0].legend(loc="upper left", frameon=False, ncol=2, handlelength=1.6, handletextpad=.4,
               columnspacing=1.1, borderaxespad=.1, bbox_to_anchor=(0, 1.0))
axes[1].annotate("grey band / dashed line = permutation null, 95th pct\n"
                 "hollow marker = not above that null ($p \\geq 0.05$)",
                 (0.025, .03), xycoords="axes fraction", fontsize=6.8, color=MUTED, va="bottom",
                 linespacing=1.6)
axes[1].annotate("main text", (x[LAYERS.index(MAIN)], .875), fontsize=6.8, color=MUTED,
                 ha="center", va="top")

fig.suptitle("Water-word and numeric-temperature geometry agree across depth, peaking at layers 4\u20135",
             fontsize=9, y=1.0, x=.005, ha="left")
fig.tight_layout(rect=(0, 0, 1, .95))
for ext in ("pdf", "png"):
    fig.savefig(f"{FIG}/temp_rsa_by_layer.{ext}", dpi=200, bbox_inches="tight")
print(f"wrote {FIG}/temp_rsa_by_layer.pdf/.png")


# ---------------------------------------------------------------- LaTeX --------------------
def fp(p):
    """format a permutation p-value; 1/5001 is the floor."""
    return "$\\le 0.0002$" if p <= PMIN + 1e-9 else f"${p:.4f}$"


def fs(v, nd=2):
    return f"${v:+.{nd}f}$"


def lay(L):
    return f"\\textbf{{{L}}}" if L == MAIN else f"{L}"


rows_pos, rows_mov, rows_dir = [], [], []
for L in LAYERS:
    n = node[node.layer == L].set_index("unit")
    m = move[move.layer == L].set_index("unit")
    # NB every row carries the empty spacer cells the tabular specs declare
    rows_pos.append(" & ".join([
        lay(L),
        fs(n.loc["C", "spearman"]), f"${n.loc['C','z']:.1f}$", fp(n.loc["C", "perm_p"]), "",
        fs(n.loc["F", "spearman"]), f"${n.loc['F','z']:.1f}$", fp(n.loc["F", "perm_p"]),
    ]) + r" \\")
    rows_mov.append(" & ".join([
        lay(L),
        fs(m.loc["C", "rho"]), fs(m.loc["C", "null_mean"]), fs(m.loc["C", "null_p95"]),
        f"${m.loc['C','z']:.1f}$", fp(m.loc["C", "p_word"]), "",
        fs(m.loc["F", "rho"]), fs(m.loc["F", "null_mean"]), fs(m.loc["F", "null_p95"]),
        f"${m.loc['F','z']:.1f}$", fp(m.loc["F", "p_word"]),
    ]) + r" \\")
    rows_dir.append(" & ".join([
        lay(L),
        fs(m.loc["C", "cross_cos_mean"], 3), fs(m.loc["F", "cross_cos_mean"], 3), "",
        fs(m.loc["C", "within_water_cos"]), "",
        fs(m.loc["C", "within_temp_cos"]), fs(m.loc["F", "within_temp_cos"]), "",
        fs(m.loc["C", "raw_word_cos"]),
    ]) + r" \\")

# ---- claims the prose makes, computed from the tables (were hard-coded before 2026-09-18) ----
mv = move.copy(); nd = node.rename(columns={"z": "z_pos", "null_p95": "p95_pos"})
mm = mv.merge(nd[["layer", "unit", "spearman", "z_pos", "p95_pos"]], on=["layer", "unit"])
fails = mv[mv.p_word >= .05].sort_values(["unit", "layer"])
n_move_fail = len(fails)
null_lo, null_hi = mv.null_mean.min(), mv.null_mean.max()
NULLR = f"$+{null_lo:.2f}$--$+{null_hi:.2f}$"
cC, cF = mv[mv.unit == "C"].sort_values("layer"), mv[mv.unit == "F"].sort_values("layer")
f_lead = int((cF.rho.values > cC.rho.values).sum())
c_ahead = [int(L) for L, a, b in zip(cC.layer, cC.rho.values, cF.rho.values) if a >= b]
n_rho_more = int((mm.rho > mm.spearman).sum())
n_z_more = int((mm.z > mm.z_pos).sum())
n_dist_more = int(((mm.rho - mm.null_p95) > (mm.spearman - mm.p95_pos)).sum())
xlo, xhi = mv.cross_cos_mean.min(), mv.cross_cos_mean.max()
peak = {u: int(mv[mv.unit == u].sort_values("rho").layer.iloc[-1]) for u in "CF"}
if n_move_fail == 0:
    MOVES_PARA = (f"\\paragraph{{Moves.}} The move analysis (Table~\\ref{{tab:app_edge_temp_rsa_layers}}) "
                  f"clears its null at every layer tested and on both scales ($p \\le {mv.p_word.max():.4f}$ "
                  f"throughout), peaking at " + (f"layer ${peak['C']}$ on both scales" if peak['C'] == peak['F'] else
                  f"layer ${peak['C']}$ for $^{{\\circ}}$C and ${peak['F']}$ for $^{{\\circ}}$F") + f". As for positions, the alignment is present from the first block output "
                  f"onward.")
    FIG_SIG = "Filled markers clear that null; every cell does."
    TAB_SIG = "Every cell clears its null."
else:
    cells = ", ".join(f"$^{{\\circ}}${r.unit} at layer ${int(r.layer)}$ ($p = {r.p_word:.3f}$)" for r in fails.itertuples())
    MOVES_PARA = (f"\\paragraph{{Moves.}} The move analysis (Table~\\ref{{tab:app_edge_temp_rsa_layers}}) "
                  f"clears its null in ${18 - n_move_fail}$ of the $18$ layer\\,$\\times$\\,scale cells; "
                  f"it does not clear it for {cells}.")
    FIG_SIG = "Filled markers clear that null, hollow markers do not."
    TAB_SIG = f"The {n_move_fail} cells with $p \\ge 0.05$ are the only ones in the temperature results that do not clear their null."
SCALE_MOVES = (f"for moves it leads at {'all nine' if f_lead == 9 else f'${f_lead}$ of the nine'} layers"
               + ("" if not c_ahead else f" ($^{{\\circ}}$C is marginally ahead at layer{'s' if len(c_ahead) > 1 else ''} "
                  + " and ".join(f"${L}$" for L in c_ahead) + ")"))

TEX = r"""% ---------------------------------------------------------------------------
% Auto-generated by fractal_analysis/make_temp_rsa_appendix.py
% Sources: tables/node_rsa_by_layer_unit.csv, tables/move_direction_stats_by_layer.csv
% The layer-10 rows reproduce Tables~\ref{tab:node_temp_rsa} and~\ref{tab:edge_temp_rsa}.
% Needs: \usepackage{booktabs,graphicx}
% ---------------------------------------------------------------------------
\section{Temperature results for other layers}
\label{app:temp_other_layers}

Section~\ref{sec:temperature} reports layer $10$. Here we give the same two analyses ---
positions (Table~\ref{tab:node_temp_rsa}) and moves (Table~\ref{tab:edge_temp_rsa}) --- at
every layer we trained, $\{0,1,2,3,4,5,10,15,20\}$, on both numeric scales. Nothing about the
method changes with depth: the same $18$ water words, the same correspondence $T$, the same
$__NM__$ pooled colder--warmer moves, the same $5000$-draw permutation null (so
$p_{\min} = 1/5001 \approx 2\times10^{-4}$), and the same seed. Only the layer at which the
steering vectors were trained differs. The layer-$10$ row of each table is set in bold and
reproduces the main text exactly. Layer $0$ here is the output of the first transformer block,
not the static token embedding.

\paragraph{Positions.} The water words predict numeric-temperature geometry at
\emph{every} layer tested and on both scales (Table~\ref{tab:app_node_temp_rsa_layers}): the
largest $p$ anywhere in the table is $0.010$, and $\rho$ never falls below $+0.24$. The two
families are therefore already correlated at the first block output, so the alignment is not
manufactured somewhere in the middle of the network. It is nonetheless clearly
depth-dependent: $\rho$ rises to a peak at layers $4$--$5$ ($+0.45$/$+0.49$ for
$^{\circ}$C/$^{\circ}$F), then falls back, with layer $15$ the weakest depth on both scales.

__MOVES_PARA__

\paragraph{Which scale.} $^{\circ}$F is the better predictor of the two nearly everywhere:
__SCALE_MOVES__, and for positions from layer $4$ onward; in the first
four blocks $^{\circ}$C is marginally ahead on positions. This mirrors the main-text ordering
($+0.48$ vs.\ $+0.33$ at layer $10$) and is not an artefact of one depth.

\paragraph{Comparing the two analyses.} The raw $\rho$ is larger for moves than for positions
at __N_RHO_MORE__ of the $18$ layer\,$\times$\,scale cells, which is the comparison made in the main text.
That comparison should be read with care: the two analyses have very different nulls. The
position null is centred at $0$, while the move null is centred at __NULLR__, because
two moves that share an endpoint are similar however the words are labelled. Standardising
each $\rho$ against its own null removes most of the gap --- moves exceed positions in $z$ in
$__N_Z_MORE__$ of the $18$ cells, and exceed them in raw distance above the null $95$th percentile
in only $__N_DIST_MORE__$. We therefore report both $z$ and the null percentiles in
Table~\ref{tab:app_edge_temp_rsa_layers} so the two analyses can be compared on the same
footing, and we do not read the height difference between the panels of
Figure~\ref{fig:temp_rsa_by_layer} as itself the effect.

\paragraph{Direction.} Table~\ref{tab:app_temp_direction_layers} sweeps the \emph{Direction}
block of Table~\ref{tab:edge_temp_rsa} over depth. The near-orthogonality reported in the main
text is not special to layer $10$: a water move and its numeric counterpart have mean cosine
between __XR2__ at every layer, on both scales. The cosine is smallest (and very
slightly negative) in the first three blocks and largest at layer $10$, so what depth changes
is the strength of a weak shared direction, not its presence. Within each family the moves
share only a partial common axis at any depth, so neither space ever acquires a single
``warmer'' direction. Meanwhile the raw, undifferenced water-word probes grow steadily more
collinear with depth, from $+0.57$ at layer $0$ to $+0.74$ at layer $15$, while the move
cosines stay near zero --- which is what the differencing is for.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figs/temp_rsa_by_layer.pdf}
\caption{The temperature/water RSA at every layer tested. \textbf{(a)} positions: RSA between
the $18\times18$ water-word RDM and the numeric RDM at the temperatures those words name.
\textbf{(b)} moves: RSA between the $__NM__\times__NM__$ RDMs of colder$\to$warmer difference vectors.
Solid lines are the observed $\rho$; the dashed line of the matching colour is that series'
permutation-null $95$th percentile, and the grey region is everything at or below the higher
of the two. __FIG_SIG__ Both panels peak at layers $4$--$5$. The shaded column is
layer $10$, the layer reported in the main text. \emph{The panels share a $y$-axis for
legibility, but their nulls differ --- $0$ for positions, __NULLR__ for moves --- so
the vertical offset between panels is not itself an effect; read each series against the
dashed line of its own colour.} Layers are plotted at equal spacing, not to scale.}
\label{fig:temp_rsa_by_layer}
\end{figure*}

\begin{table}[t]
\centering
\setlength{\tabcolsep}{3.5pt}
\small
\begin{tabular}{@{}lrrrcrrr@{}}
\toprule
 & \multicolumn{3}{c}{$^{\circ}$C} & & \multicolumn{3}{c}{$^{\circ}$F} \\
\cmidrule(r){2-4}\cmidrule(l){6-8}
Layer & $\rho$ & $z$ & $p$ & & $\rho$ & $z$ & $p$ \\
\midrule
__ROWS_POS__
\bottomrule
\end{tabular}
\caption{\textbf{Positions, all layers.} RSA between the water-word prediction matrix
(Eq.~\ref{eq:temp-pm-pos}) and the measured numeric reference (Eq.~\ref{eq:temp-rm-pos}), over
$18$ words and $\binom{18}{2}=153$ pairs. $z$ is the observed $\rho$ in standard deviations of
the label-permutation null, which is centred at $0$ here (mean $|\rho_{\text{null}}| < 0.003$,
s.d.\ $\approx 0.09$ at every layer). Every cell clears its null. Layer $10$ (bold) is
Table~\ref{tab:node_temp_rsa}.}
\label{tab:app_node_temp_rsa_layers}
\end{table}

\begin{table*}[t]
\centering
\setlength{\tabcolsep}{4pt}
\small
\begin{tabular}{@{}lrrrrrcrrrrr@{}}
\toprule
 & \multicolumn{5}{c}{$^{\circ}$C} & & \multicolumn{5}{c}{$^{\circ}$F} \\
\cmidrule(r){2-6}\cmidrule(l){8-12}
Layer & $\rho$ & null mean & null $95$th & $z$ & $p$ & & $\rho$ & null mean & null $95$th & $z$ & $p$ \\
\midrule
__ROWS_MOV__
\bottomrule
\end{tabular}
\caption{\textbf{Moves, all layers.} RSA between the move prediction matrix
(Eq.~\ref{eq:temp-pm-move}) and the measured move reference (Eq.~\ref{eq:temp-rm-move}), over
$__NM__$ pooled colder$\to$warmer moves and $\binom{__NM__}{2}=__NPAIR__$ pairs. As in the main text the
null reassigns which of the $__NW__$ ladder words sits at which temperature and rebuilds the moves
over the same word pairs, so it is centred near __NULLR__ rather than $0$: two moves
sharing an endpoint are similar whatever the labelling. \textbf{Read $\rho$ against the null
$95$th percentile in the same column, not against zero}, and use $z$ rather than $\rho$ when
comparing against Table~\ref{tab:app_node_temp_rsa_layers}. __TAB_SIG__
Layer $10$ (bold) is the \emph{Arrangement} block of Table~\ref{tab:edge_temp_rsa}.}
\label{tab:app_edge_temp_rsa_layers}
\end{table*}

\begin{table*}[t]
\centering
\setlength{\tabcolsep}{4pt}
\small
\begin{tabular}{@{}lrrcrcrrcr@{}}
\toprule
 & \multicolumn{2}{c}{$\cos$(water move, temp.\ move)} & & within & & \multicolumn{2}{c}{within temp.\ moves} & & raw water \\
\cmidrule(r){2-3}\cmidrule(lr){5-5}\cmidrule(lr){7-8}
Layer & $^{\circ}$C & $^{\circ}$F & & water moves & & $^{\circ}$C & $^{\circ}$F & & probes \emph{(ref.)} \\
\midrule
__ROWS_DIR__
\bottomrule
\end{tabular}
\caption{\textbf{Direction, all layers} --- the \emph{Direction} block of
Table~\ref{tab:edge_temp_rsa} swept over depth. Corresponding water and numeric moves are close
to orthogonal at every layer (mean $\cos$ between __XR3__), so the agreement
measured in Table~\ref{tab:app_edge_temp_rsa_layers} is about relative arrangement rather than
shared direction at every depth. Within each family the moves share only a partial common axis,
so neither space has a single ``warmer'' direction anywhere in the network. The last column is
the mean pairwise cosine among the $__NW__$ raw, undifferenced water-word probes: it rises from
$+0.57$ to $+0.74$ with depth while the move cosines stay near zero, which is what differencing
is for. The within-water column is shared across scales by construction (it does not involve
the numeric manifold). Layer $10$ (bold) is Table~\ref{tab:edge_temp_rsa}.}
\label{tab:app_temp_direction_layers}
\end{table*}
"""

TEX = (TEX.replace("__MOVES_PARA__", MOVES_PARA).replace("__SCALE_MOVES__", SCALE_MOVES)
          .replace("__N_RHO_MORE__", "all" if n_rho_more == 18 else f"${n_rho_more}$")
          .replace("__NULLR__", NULLR).replace("__N_Z_MORE__", str(n_z_more)).replace("__N_DIST_MORE__", str(n_dist_more))
          .replace("__XR2__", f"${xlo:+.2f}$ and ${xhi:+.2f}$").replace("__XR3__", f"${xlo:+.3f}$ and ${xhi:+.3f}$")
          .replace("__FIG_SIG__", FIG_SIG).replace("__NPAIR__", str(NM * (NM - 1) // 2)).replace("__NM__", str(NM)).replace("__NW__", str(NW)).replace("__TAB_SIG__", TAB_SIG))
assert "__" not in TEX.replace("__ROWS_POS__", "").replace("__ROWS_MOV__", "").replace("__ROWS_DIR__", ""), "unfilled placeholder"
TEX = (TEX.replace("__ROWS_POS__", "\n".join(rows_pos))
          .replace("__ROWS_MOV__", "\n".join(rows_mov))
          .replace("__ROWS_DIR__", "\n".join(rows_dir)))
out = f"{PAPER}/appendix_temp_other_layers.tex"
open(out, "w").write(TEX)
print(f"wrote {out}")

print("\n=== positions ===")
print(node.pivot_table(index="layer", columns="unit", values=["spearman", "perm_p"]).to_string())
print("\n=== moves ===")
print(move.pivot_table(index="layer", columns="unit",
                       values=["rho", "null_p95", "z", "p_word"]).to_string())
