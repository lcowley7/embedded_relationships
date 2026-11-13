"""sweep_spec.py SPEC -- copy of filtered_analysis/sweep/src/sweep_temp.py with the move set from ladders.SPECS[SPEC].

Original docstring: sweep_temp.py -- Tables 3 and 4 as a function of the drop fraction f, at every trained layer.

Items are the 18 water words. Each enters the tables paired with the numeric vector at its
nominal temperature, so it is scored by the MINIMUM of the pair:
    s_i = min( Overall(word_i), Overall(numeric level nearest to t_i) ).
At each f we drop the floor(f*18) lowest-scoring words. Table 3 (nodes) is recomputed on the
surviving words; Table 4 (moves) keeps only moves whose BOTH endpoints survive, and its
word-permutation null runs over the surviving words.

Note this thins the WATER WORDS, not the numeric ladder: the nearest-row lookup that supplies
the numeric half of each pair is unchanged.

With LT_MODEL set the vectors (through analysis_harness), judge scores, layers and outputs are
that model's (lt_model.py); the ladder specs and H2O_TEMP table are shared.
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "."
sys.path.insert(0, f"{REPO}/fractal_analysis")
sys.path.insert(0, f"{REPO}")
import analysis_harness as H, alignment_harness as A   # noqa
import lt_model as LT                                  # noqa
DB = H.DATA_BASE
BASE = f"{REPO}/filtered_analysis/sweep"
INP = f"{REPO}/inputs"
NPERM, SEED = 5000, 0
MIN_MOVES = 5   # below this the move RDM is degenerate (e.g. 3 moves -> 3 pairs); not reported
FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
UNITS = {"C": ("temp_c1", "temp_c", lambda c: c),
         "F": ("temp_f1", "temp_f", lambda c: c * 9.0 / 5.0 + 32.0)}
canon = lambda w: "puddle" if w == "water" else w      # the "water" rung has no probe of its own

SPEC = sys.argv[1]
sys.path.insert(0, "./method_audit_2026-09-16/ladders_18")
import ladders as LD  # noqa
TABLES = FIGS = f"./method_audit_2026-09-16/ladders_18/out/{SPEC}" \
    + ("" if LT.TAG is None else f"/{LT.TAG}")          # other models never overwrite Gemma's outputs
os.makedirs(TABLES, exist_ok=True)
sc = pd.read_csv(f"{LT.out_dir(f'{BASE}/manifests')}/judge_scores.csv")
# a layer is sweepable only if both halves of every pair were steered there
LAYERS = [L for L in LT.layers([0, 1, 2, 3, 4, 5, 10, 15, 20])
          if {"h2o_v1", "temp_c1", "temp_f1"} <= set(sc[sc.layer == L].family)]
assert LAYERS, "no layer has water-word and numeric judge scores"
print(f"{LT.label()}: sweeping layers {LAYERS}", flush=True)
# figure titles: unchanged wording for the paper's Gemma run, model + layer count otherwise
WHICH = "all nine trained layers" if LT.TAG is None else f"{LT.label()}, {len(LAYERS)} layers"


def temp_axis(unit, layer):
    ds, col, conv = UNITS[unit]
    W = H.load_weight(ds, layer).astype(float)
    Tax = pd.read_parquet(f"{DB}/{ds}/concept_index.parquet")[col].values.astype(float)
    o = np.argsort(Tax); X, Tax = A.unit(W[o]), Tax[o]
    ns = sc[(sc.family == ds) & (sc.layer == layer)]
    nscore = dict(zip(ns.coord.astype(float), ns.overall))
    def look(t_c):
        i = int(np.argmin(np.abs(Tax - conv(t_c))))
        return X[i], float(Tax[i]), nscore.get(float(Tax[i]), np.nan)
    return look


def edges():
    return LD.edges(SPEC)[0]


EDGES = edges()
rows = []
for layer in LAYERS:
    ws = sc[(sc.family == "h2o_v1") & (sc.layer == layer)]
    word_score = dict(zip(ws.label, ws.overall))
    cl = H.load_cloud("h2o_v1", layer)
    LAB = [str(g) for g in cl["labels"]]
    Xw_all = A.unit(cl["X"].astype(float))
    WIDX = {w: i for i, w in enumerate(LAB)}

    for unit in ("C", "F"):
        look = temp_axis(unit, layer)
        pair = {w: min(word_score[w], look(H.H2O_TEMP[w])[2]) for w in LAB}
        order = sorted(LAB, key=lambda w: (pair[w], WIDX[w]))
        for f in FRACS:
            k = int(np.floor(f * len(LAB)))
            dropped = set(order[:k]); keep = [w for w in LAB if w not in dropped]

            # ---- Table 3: node RSA over surviving words ----
            Xw = np.vstack([Xw_all[WIDX[w]] for w in keep])
            Xt = np.vstack([look(H.H2O_TEMP[w])[0] for w in keep])
            rw, rt = pdist(Xw, "cosine"), pdist(Xt, "cosine")
            node = float(spearmanr(rw, rt).correlation)
            RW = squareform(rw); rng = np.random.default_rng(SEED)
            nn = np.array([spearmanr(squareform(RW[np.ix_(p, p)]), rt).correlation
                           for p in (rng.permutation(len(keep)) for _ in range(NPERM))])
            node_p = float((1 + np.sum(nn >= node)) / (NPERM + 1))

            # ---- Table 4: moves with both endpoints surviving ----
            es = [e for e in EDGES if canon(e["src"]) in keep and canon(e["tgt"]) in keep]
            mw = sorted({canon(e["src"]) for e in es} | {canon(e["tgt"]) for e in es})
            mv = mp = mnull = ncos = np.nan
            if len(es) >= 3 and len(mw) >= 4:
                mi = {w: i for i, w in enumerate(mw)}
                Ws = np.vstack([Xw_all[WIDX[w]] for w in mw])
                si = np.array([mi[canon(e["src"])] for e in es])
                ti = np.array([mi[canon(e["tgt"])] for e in es])
                V = np.vstack([look(e["tT"])[0] - look(e["sT"])[0] for e in es])
                dV = pdist(V, "cosine")
                stat = lambda p: spearmanr(pdist(Ws[p][ti] - Ws[p][si], "cosine"), dV).correlation
                mv = float(stat(np.arange(len(mw))))
                rng = np.random.default_rng(SEED)
                nl = np.array([stat(rng.permutation(len(mw))) for _ in range(NPERM)])
                mp = float((1 + np.sum(nl >= mv)) / (NPERM + 1))
                mnull = float(np.quantile(nl, .95))
                G = Ws[ti] - Ws[si]
                gu = G / np.linalg.norm(G, axis=1, keepdims=True)
                vu = V / np.linalg.norm(V, axis=1, keepdims=True)
                ncos = float((gu * vu).sum(1).mean())

            rows.append(dict(layer=layer, unit=unit, f=f, n_words=len(keep),
                             node_rsa=round(node, 4), node_p=round(node_p, 5),
                             node_null_p95=round(float(np.quantile(nn, .95)), 4),
                             n_moves=len(es), n_move_words=len(mw),
                             move_rsa=round(mv, 4), move_p=round(mp, 5),
                             move_null_p95=round(mnull, 4), mean_cos=round(ncos, 4),
                             dropped=",".join(sorted(dropped))))
            print(f"  L{layer:2d} {unit} f={f:.1f}  words {len(keep):2d}  node {node:+.3f} "
                  f"(p {node_p:.4f})  moves {len(es):2d} rsa {mv:+.3f} (p {mp:.4f})", flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{TABLES}/temp_sweep.csv", index=False)

df["move_reportable"] = df.n_moves >= MIN_MOVES
df.to_csv(f"{TABLES}/temp_sweep.csv", index=False)

LAB_L = [str(L) for L in LAYERS]
PANELS = [("node_rsa", "node_p", None, "Positions"),
          ("move_rsa", "move_p", "move_reportable", "Moves")]


def star(p):
    return "" if p != p else ("***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "")


# ---------- primary figure: heat grid (legible at 9 layers x 6 drop fractions) ----------
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
for r, (rc, pc, gate, tname) in enumerate(PANELS):
    for c, unit in enumerate(("C", "F")):
        ax = axes[r, c]
        M = np.full((len(LAYERS), len(FRACS)), np.nan)
        for i, L in enumerate(LAYERS):
            for j, f in enumerate(FRACS):
                row = df[(df.layer == L) & (df.unit == unit) & (df.f == f)]
                if not len(row):
                    continue
                row = row.iloc[0]
                if gate and not row[gate]:
                    continue
                M[i, j] = row[rc]
        im = ax.imshow(M, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
        for i, L in enumerate(LAYERS):
            for j, f in enumerate(FRACS):
                row = df[(df.layer == L) & (df.unit == unit) & (df.f == f)]
                if not len(row):
                    continue
                row = row.iloc[0]
                if gate and not row[gate]:
                    ax.text(j, i, "--", ha="center", va="center", fontsize=8, color="0.35")
                    continue
                ax.text(j, i, f"{row[rc]:.2f}\n{star(row[pc])}", ha="center", va="center",
                        fontsize=6.6, color="black")
        ax.set_xticks(range(len(FRACS))); ax.set_xticklabels([f"{f:.1f}" for f in FRACS])
        ax.set_yticks(range(len(LAYERS))); ax.set_yticklabels(LAB_L)
        ax.set_title(f"{tname}, $^\\circ${unit}", fontsize=10)
        if c == 0:
            ax.set_ylabel("layer")
        if r == 1:
            ax.set_xlabel("drop fraction $f$")
fig.colorbar(im, ax=axes, shrink=.6, label=r"Spearman $\rho$")
fig.suptitle(f"Temperature sweep at {WHICH}. Stars are each cell's own "
             "permutation null (*** $p<.001$, ** $p<.01$, * $p<.05$).\n"
             f"Water words remaining: 18, 17, 15, 13, 11, 9. Move cells with fewer than "
             f"{MIN_MOVES} surviving moves are degenerate and shown as --.", fontsize=9)
fig.savefig(f"{FIGS}/temp_sweep.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# ---------- alternative: line version, same data ----------
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
cm = plt.cm.viridis(np.linspace(.05, .92, len(LAYERS)))
for c, unit in enumerate(("C", "F")):
    for col, L in zip(cm, LAYERS):
        d = df[(df.unit == unit) & (df.layer == L)]
        axes[0, c].plot(d.f, d.node_rsa, "-o", color=col, ms=3.5, lw=1.4, label=f"L{L}")
        dm = d[d.move_reportable]
        axes[1, c].plot(dm.f, dm.move_rsa, "-o", color=col, ms=3.5, lw=1.4, label=f"L{L}")
    axes[0, c].set_title(f"Positions, $^\\circ${unit}", fontsize=10)
    axes[1, c].set_title(f"Moves, $^\\circ${unit}", fontsize=10)
    axes[1, c].set_xlabel("drop fraction $f$ (least effective removed)")
for a in axes.ravel():
    a.grid(alpha=.3)
axes[0, 0].set_ylabel(r"Spearman $\rho$"); axes[1, 0].set_ylabel(r"Spearman $\rho$")
axes[0, 1].legend(fontsize=6.5, ncol=2, loc="best")
fig.suptitle(f"Temperature sweep, {WHICH} (line form of the same data). "
             f"Move points with fewer than {MIN_MOVES} surviving moves are omitted.", fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIGS}/temp_sweep_lines.png", dpi=140)
plt.close(fig)
print(f"\nwrote {TABLES}/temp_sweep.csv, {FIGS}/temp_sweep.png, temp_sweep_lines.png")
