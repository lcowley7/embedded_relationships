"""compare_model_sweeps.py -- the efficacy sweep side by side for Gemma-2-2B, OLMo-3.1-32B and
Qwen3-32B, at depth-matched layers (Gemma L -> round(L * 64 / 26): 5/10/15 -> 12/25/37).

Reads each model's own sweep tables (tables/, tables/<tag>/) and reduces every analysis to
"holds to f": the largest drop fraction f such that the result is significant at EVERY f' <= f
(p < 0.05 and reportable), '-' if it already fails at f = 0. Writes tables/cross_model_sweep.csv.

Run: PY filtered_analysis/sweep/src/compare_model_sweeps.py
"""
import os
import numpy as np, pandas as pd

BASE = "./filtered_analysis/sweep"
MODELS = [("Gemma-2-2B", ""), ("OLMo-3.1-32B", "olmo31_32b"), ("Qwen3-32B", "qwen3_32b")]
DEPTH = {5: "5", 10: "10", 15: "15", 20: "20", 12: "5", 25: "10", 37: "15", 49: "20"}  # -> Gemma layer
FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
# Temperature follows the paper's 18-word / 49-move ladders: the sweep of
# method_audit_2026-09-16/ladders_18/sweep_spec.py (moves reportable at >= 5 surviving moves),
# not tables[/<tag>]/temp_sweep.csv (the older 14-word / 33-move ladders).
TEMP_SWEEP = "./method_audit_2026-09-16/ladders_18/out/proposed"


def holds_to(ok_by_f):
    """ok_by_f: {f: bool}. Largest f with every f' <= f True; None if f = 0 fails."""
    last = None
    for f in FRACS:
        if not ok_by_f.get(f, False):
            break
        last = f
    return last


def tab(tag, name):
    """One sweep table, restricted to layers with a depth-matched twin (drops Gemma's 0-4)."""
    p = f"{BASE}/tables/{tag + '/' if tag else ''}{name}"
    d = pd.read_csv(p)
    return d[d.layer.isin(list(DEPTH))]


rows = []
for model, tag in MODELS:
    # ---- WordNet: joint graph beats every sampled tree AND gain CI excludes 0 ----
    b = tab(tag, "bodypart_sweep.csv")
    for L, d in b.groupby("layer"):
        d = d.set_index("f")
        ok = {f: bool(d.loc[f, "rsa_full"] > d.loc[f, "tree_max"] and d.loc[f, "gain_lo"] > 0) for f in d.index}
        rows.append(dict(model=model, family="WordNet", analysis="joint graph > all trees, gain CI > 0",
                         layer=int(L), depth=DEPTH[int(L)], holds_to=holds_to(ok),
                         stat_f0=d.loc[0.0, "gain"], stat_f5=d.loc[0.5, "gain"], stat="gain"))
    # ---- Temperature: nodes and moves, each unit (the paper's 18-word ladders, sweep_spec.py) ----
    t = pd.read_csv(f"{TEMP_SWEEP}/{tag + '/' if tag else ''}temp_sweep.csv")
    t = t[t.layer.isin(list(DEPTH))]
    for (L, u), d in t.groupby(["layer", "unit"]):
        d = d.set_index("f")
        for kind, rc, pc, gate in [("positions", "node_rsa", "node_p", None),
                                   ("moves", "move_rsa", "move_p", "move_reportable")]:
            ok = {f: bool(d.loc[f, pc] < .05 and (gate is None or d.loc[f, gate])) for f in d.index}
            rows.append(dict(model=model, family=f"temperature °{u}", analysis=kind, layer=int(L),
                             depth=DEPTH[int(L)], holds_to=holds_to(ok),
                             stat_f0=d.loc[0.0, rc], stat_f5=d.loc[0.5, rc], stat="rho"))
    # ---- Analogies (wave 1 metaphors) ----
    a = tab(tag, "analogy_sweep.csv")
    for (ds, L, kind), d in a[a.experiment != "frames_v1"].groupby(["experiment", "layer", "kind"]):
        d = d.set_index("f")
        ok = {f: bool(d.loc[f, "p"] < .05 and d.loc[f, "reportable"]) for f in d.index}
        rows.append(dict(model=model, family=ds.split("_")[0] + "/" + ds.split("_")[1],
                         analysis=kind, layer=int(L), depth=DEPTH[int(L)], holds_to=holds_to(ok),
                         stat_f0=d.loc[0.0, "rho"], stat_f5=d.loc[0.5, "rho"], stat="rho"))
    # ---- Frames: how many of 10 profession pairs clear, f <= 0.3 only ----
    fr = tab(tag, "analogy_sweep_frames.csv")
    for (L, kind), d in fr.groupby(["layer", "kind"]):
        d = d.set_index("f")
        rows.append(dict(model=model, family="frames", analysis=kind, layer=int(L), depth=DEPTH[int(L)],
                         holds_to=np.nan, stat="n_sig of 10, f=0/.1/.2/.3",
                         stat_f0=int(d.loc[0.0, "n_sig"]), stat_f5=np.nan,
                         frames_nsig="/".join(str(int(d.loc[f, "n_sig"])) for f in FRACS[:4])))

df = pd.DataFrame(rows)
df.to_csv(f"{BASE}/tables/cross_model_sweep.csv", index=False)

fmt = lambda v: "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1f}"
main = df[df.family != "frames"].copy()
main["cell"] = main.holds_to.map(fmt)
piv = main.pivot_table(index=["family", "analysis"], columns=["depth", "model"], values="cell",
                       aggfunc="first")
order = [(dp, m) for dp in ["5", "10", "15", "20"] for m, _ in MODELS if (dp, m) in piv.columns]
print("Largest drop fraction f up to which each result stays significant at every step "
      "('-' = not significant at f = 0), by Gemma-equivalent depth\n")
print(piv[order].fillna("").to_string())
fr = df[df.family == "frames"].pivot_table(index="analysis", columns=["depth", "model"],
                                           values="frames_nsig", aggfunc="first")
print("\nFrames: profession pairs (of 10) clearing p<.05 at f = 0 / .1 / .2 / .3\n")
print(fr[[c for c in order if c in fr.columns]].fillna("").to_string())
print(f"\nwrote {BASE}/tables/cross_model_sweep.csv")
