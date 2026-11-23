"""Figure 6: judge-free steering check. Does a temperature within ±3 of the target actually
appear in the steered output?

Success per (vector × held-out prompt): at some non-zero factor a parsed temperature lands
within TOL of the target AND beats the factor-0 baseline error. Per-temperature success =
mean over that vector's held-out prompts. Reported per layer, for Celsius (±3°C) and
Fahrenheit (±3°F). Two parses: PRIMARY = numbers with a temperature cue (°C / degrees /
celsius); LOOSE = any integer in the text (these are thermometer-reading prompts).

Data: <ds>{chunks}/diffmean_l<L>/inference/steering_data.parquet  (DiffMean_steered_generation).
Writes outputs/figures/temp_steering_parse.{png,pdf}.
"""
import re, glob, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # repository root
from relgeom.paths import AXBENCH, FIGURES, ensure
from relgeom.data import need
need("axbench/*/diffmean_l*/inference/steering_data.parquet")
FIG = str(ensure(FIGURES))
DB = str(AXBENCH)
LAYERS = [0, 1, 2, 3, 4, 5, 10, 15, 20]
TOL = 3
CUE = {"C": re.compile(r'(-?\d+)\s*(?:°\s*c|°c|degrees?\s*c(?:elsius)?|degrees?|celsius)', re.I),
       "F": re.compile(r'(-?\d+)\s*(?:°\s*f|°f|degrees?\s*f(?:ahrenheit)?|degrees?|fahrenheit)', re.I)}
ANY = re.compile(r'-?\d+')


def slug_temp(vid):
    s = vid.split("__")[-1]
    return -int(s[3:]) if s.startswith("neg") else int(s)


def parsed(text, unit, loose):
    text = str(text)
    hits = ANY.findall(text) if loose else CUE[unit].findall(text)
    out = []
    for h in hits:
        try: out.append(int(h))
        except ValueError: pass
    return out


def eval_ds(prefix, unit, layer, loose):
    """returns dict temp -> success_frac over held-out prompts."""
    per_temp = {}
    for parq in glob.glob(f"{DB}/{prefix}*/diffmean_l{layer}/inference/steering_data.parquet"):
        chunk = parq.split("/diffmean")[0]
        ci = pd.read_parquet(f"{chunk}/concept_index.parquet")
        cid2t = {int(r.concept_id): slug_temp(r.vector_id) for r in ci.itertuples()}
        d = pd.read_parquet(parq)
        for (cid, iid), g in d.groupby(["concept_id", "input_id"]):
            T = cid2t.get(int(cid))
            if T is None: continue
            g = g.sort_values("factor")
            fac = g["factor"].values
            emit = [parsed(x, unit, loose) for x in g["DiffMean_steered_generation"].values]
            errs = [min([abs(e - T) for e in em], default=None) for em in emit]
            base = errs[0] if fac[0] == 0.0 else None
            nz = [e for f, e in zip(fac, errs) if f > 0 and e is not None]
            best = min(nz) if nz else None
            passed = best is not None and best <= TOL and (base is None or base > best)
            per_temp.setdefault(T, []).append(passed)
    return {t: float(np.mean(v)) for t, v in per_temp.items()}


DS = [("temp_c1", "C", "Celsius", "tab:blue"), ("temp_f1", "F", "Fahrenheit", "tab:green")]
print(f"parse-based steering success (target ±{TOL} appears in output, beats baseline)")
tables = {}
for prefix, unit, name, _ in DS:
    print(f"\n{name}:")
    tables[prefix] = {}
    for L in LAYERS:
        strict = eval_ds(prefix, unit, L, loose=False)
        loose = eval_ds(prefix, unit, L, loose=True)
        tables[prefix][L] = (strict, loose)
        ms = np.mean(list(strict.values())) if strict else float("nan")
        ml = np.mean(list(loose.values())) if loose else float("nan")
        print(f"  L{L}: temp-cued success {100*ms:4.0f}%   |  any-number success {100*ml:4.0f}%   (n_temps={len(strict)})")

# ---- figure: (A) mean success by layer C vs F (both parses); (B) success vs temp at L10 (C) ----
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(LAYERS)); w = 0.2
for k, (prefix, unit, name, col) in enumerate(DS):
    strictm = [100*np.mean(list(tables[prefix][L][0].values())) for L in LAYERS]
    loosem = [100*np.mean(list(tables[prefix][L][1].values())) for L in LAYERS]
    ax[0].bar(x + (2*k-1.5)*w, strictm, w, color=col, edgecolor="k", lw=.3, label=f"{name} temp-cued")
    ax[0].bar(x + (2*k-0.5)*w, loosem, w, color=col, alpha=0.45, edgecolor="k", lw=.3, label=f"{name} any-number")
ax[0].set_xticks(x); ax[0].set_xticklabels([f"L{L}" for L in LAYERS])
ax[0].set_ylabel(f"steering success (%)  [target ±{TOL} appears, beats baseline]")
ax[0].set_title(f"Parse-based steering success by layer (±{TOL}°)", fontsize=11)
ax[0].legend(fontsize=7); ax[0].grid(axis="y", alpha=.3)

st10 = tables["temp_c1"][10][0]
T = np.array(sorted(st10)); S = np.array([st10[t] for t in T]) * 100
for xx in range(-20, 111, 10): ax[1].axvline(xx, color="tab:red", ls=":", lw=.6, alpha=.25)
ax[1].axhline(np.mean(S), color="0.5", ls="--", lw=1, label=f"mean {np.mean(S):.0f}%")
ax[1].scatter(T, S, c=S, cmap="viridis", s=28, edgecolor="k", lw=.2)
ax[1].set_xlabel("temperature (°C)  (red combs = round °C)"); ax[1].set_ylabel(f"success (%) — target ±{TOL}°C appears")
ax[1].set_title("Celsius L10: parse-based steering success vs temperature", fontsize=11)
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
fig.suptitle(f"Does the target temperature (±{TOL}°) actually appear in the steered output? "
             f"— judge-free parse", fontsize=12, y=1.02)
plt.tight_layout()
out = f"{FIG}/temp_steering_parse.png"
plt.savefig(out, dpi=115, bbox_inches="tight"); print(f"\nsaved -> {out}")
