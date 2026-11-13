"""Steering-performance plots for temp_c1 / temp_f1 — proof the DiffMean vectors work,
and how that varies BY LAYER (companion to the by-layer ball plots).

Source: <ds>/diffmean_l<L>/eval/results.jsonl  (gemini-3.1-flash-lite judge, 3 rubrics,
harmonic-mean-with-zero-gate; per-concept Overall = max over factors of lm_judge_rating in [0,2]).

Run: PY=python
     $PY fractal_analysis/make_temp_steering_eval.py
"""
import sys, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "./fractal_analysis")
import analysis_harness as H
FIG = "./fractal_analysis/figs"
# All nine layers the temperature manifolds were trained and evaluated at -- previously this
# hardcoded [5, 10, 15, 20] and silently dropped the six shallow layers, which do have eval data.
LAYERS = [0, 1, 2, 3, 4, 5, 10, 15, 20]
DB = H.DATA_BASE


def load(ds, L, tcol=None):
    R = [json.loads(l) for l in open(f"{DB}/{ds}/diffmean_l{L}/eval/results.jsonl")]
    fac = np.array(R[0]["results"]["LMJudgeEvaluator"]["DiffMean"]["factor"], float)
    rat = np.array([r["results"]["LMJudgeEvaluator"]["DiffMean"]["lm_judge_rating"] for r in R], float)
    T = np.array([r[tcol] for r in R], float) if tcol and tcol in R[0] else None
    best = np.nanmax(rat, axis=1)
    return fac, rat, T, best


# The three temperature manifolds: the two numeric ladders plus the water words, which are the
# physically-grounded temperature representation the cross-family comparison rests on.
DS = [("temp_c1", "temp_c", "Celsius (131 levels)", "tab:blue"),
      ("temp_f1", "temp_f", "Fahrenheit (251 levels)", "tab:green"),
      ("h2o_v1",  None,     "water words (18)", "tab:purple")]
fig, ax = plt.subplots(1, 3, figsize=(19, 5))

# Panel A: dose-response per layer (temp_c1) — mean rating vs factor
cmap = plt.cm.viridis(np.linspace(0.10, 0.92, len(LAYERS)))
for ci, L in enumerate(LAYERS):
    fac, rat, T, best = load("temp_c1", L, "temp_c")
    ax[0].plot(fac, np.nanmean(rat, 0), "-o", ms=3.5, lw=1.4, color=cmap[ci], label=f"L{L}")
ax[0].set_xlabel("steering factor"); ax[0].set_ylabel("mean steering score (lm_judge_rating, 0–2)")
ax[0].set_title("Celsius: dose-response, all nine layers\n(rises with factor; shallow and mid layers steer best)", fontsize=10)
ax[0].legend(title="layer", fontsize=7, ncol=2); ax[0].grid(alpha=.3)

# Panel B: mean best-factor score by layer, for all three temperature manifolds
w = 0.27; x = np.arange(len(LAYERS))
for k, (ds, tcol, name, col) in enumerate(DS):
    means = []
    for L in LAYERS:
        _, _, _, best = load(ds, L, tcol); means.append(best.mean())
    ax[1].bar(x + (k - 1) * w, means, w, color=col, edgecolor="k", lw=.3, label=name)
    for xi, m in zip(x + (k - 1) * w, means):
        ax[1].text(xi, m + .01, f"{m:.2f}", ha="center", fontsize=5.5, rotation=90)
ax[1].set_xticks(x); ax[1].set_xticklabels([f"L{L}" for L in LAYERS], fontsize=8)
ax[1].set_xlabel("layer"); ax[1].set_ylabel("mean per-concept Overall score (max over factors, 0–2)")
ax[1].set_title("Mean steering score by layer and manifold\n(water words steer far better than either numeric ladder)", fontsize=10)
ax[1].set_ylim(0, 2.0); ax[1].legend(fontsize=7.5, loc="upper left"); ax[1].grid(axis="y", alpha=.3)

# Panel C: per-temperature steering score at L10 (does it connect to the geometry?)
fac, rat, T, best = load("temp_c1", 10, "temp_c")
si = np.argsort(T)
ax[2].axhline(np.median(best), color="0.5", ls="--", lw=1, label=f"median {np.median(best):.2f}")
for xx in range(-20, 111, 10): ax[2].axvline(xx, color="tab:red", ls=":", lw=.6, alpha=.25)
ax[2].scatter(T[si], best[si], c=best[si], cmap="viridis", s=26, edgecolor="k", lw=.2)
ax[2].set_xlabel("temperature (°C)  (red combs = round °C)"); ax[2].set_ylabel("per-temperature Overall score (0–2)")
ax[2].set_title("Celsius L10: steering score vs temperature", fontsize=10)
ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

fig.suptitle("Do the temperature DiffMean vectors actually steer? — gemini-3.1-flash-lite judge, "
             "3 rubrics (concept/instruction/fluency), harmonic-mean gate", fontsize=12, y=1.02)
plt.tight_layout()
out = f"{FIG}/temp_steering_eval.png"
plt.savefig(out, dpi=115, bbox_inches="tight"); print(f"saved -> {out}")

# ---- printed table for the writeup ----
print("\nmean best-factor Overall score (0–2), with % of concepts scoring > 0.5:")
hdr = "  " + f"{'manifold':24s}" + "".join(f"{'L'+str(L):>14s}" for L in LAYERS)
print(hdr); print("  " + "-" * (len(hdr) - 2))
for ds, tcol, name, _ in DS:
    cells = []
    for L in LAYERS:
        _, _, _, best = load(ds, L, tcol)
        cells.append(f"{best.mean():.2f} ({100*np.mean(best>0.5):.0f}%)")
    print("  " + f"{name:24s}" + "".join(f"{c:>14s}" for c in cells))
