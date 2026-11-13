"""compare_models.py -- the paper's analyses side by side: Gemma-2-2B (the paper), OLMo-3.1-32B, Qwen3-32B.

Layers are matched by relative depth: Gemma layer L -> round(L * n_layers / 26) (lt_model.layer).
Reads only tables the analyses already wrote -- Gemma's canonical ones and the <tag>/ subfolders
produced by run_analyses.sh -- and computes nothing. Models whose tables are missing are skipped.

Writes bigmodel/tables/cross_model_summary.csv (long format, one row per
experiment x analysis x layer x model) and prints compact tables (* = p < 0.05).
Run: PY bigmodel/compare_models.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

NP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NP / "bigmodel"))
from models import MODELS   # noqa: E402

FA = NP / "fractal_analysis"
# Temperature MOVES follow the paper's 18-word ladders (phase / precipitation / adjectival, 49 moves;
# method_audit_2026-09-16/ladders_18/ladders.py, SPECS["proposed"]), written by that folder's
# move_stats.py -- not the 14-word, 33-move tables in fractal_analysis/tables/. Positions are unaffected.
TEMP_MOVES = NP / "method_audit_2026-09-16/ladders_18/out/proposed"
TAGS = [None, "olmo31_32b", "qwen3_32b"]
NAME = {None: "Gemma-2-2B", "olmo31_32b": "OLMo-3.1-32B", "qwen3_32b": "Qwen3-32B"}
TEMP_L, MAIN_L, SCAN_L = [0, 1, 2, 3, 4, 5, 10, 15, 20], [5, 10, 15, 20], [0, 5, 10, 15, 20]


def mapL(tag, L):
    return L if tag is None else round(L * MODELS[tag]["n_layers"] / 26)


def sub(base, tag):
    return Path(base) if tag is None else Path(base) / tag


rows = []
add = lambda **k: rows.append(k)

for t in TAGS:
    m = NAME[t]
    # ---- temperature: positions and moves
    f = sub(FA / "tables", t) / "node_rsa_by_layer_unit.csv"
    if f.exists():
        d = pd.read_csv(f)
        for gl in TEMP_L:
            for u in ("C", "F"):
                r = d[(d.layer == mapL(t, gl)) & (d.unit == u)].iloc[0]
                add(experiment="temperature", analysis=f"positions_{u}", gemma_layer=gl, model=m,
                    layer=mapL(t, gl), rho=r.spearman, p=r.perm_p, null_p95=r.null_p95)
    for gl in TEMP_L:
        f = sub(TEMP_MOVES, t) / f"move_direction_stats_L{mapL(t, gl)}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        pc = next(c for c in ("p", "perm_p", "p_word") if c in d.columns)
        for u in ("C", "F"):
            r = d[d.unit == u].iloc[0]
            add(experiment="temperature", analysis=f"moves_{u}", gemma_layer=gl, model=m,
                layer=mapL(t, gl), rho=r.rho, p=r[pc], null_p95=r.null_p95)
    # ---- analogies: wave 1 and SCAN
    for base, fn, gls in ((FA / "extension/tables", "wave1_rsa.csv", MAIN_L),
                          (FA / "extension/wave3_scan/tables", "scan_rsa.csv", SCAN_L)):
        f = sub(base, t) / fn
        if not f.exists():
            continue
        d = pd.read_csv(f)
        for gl in gls:
            L = mapL(t, gl)
            for (exp, kind), g in d[d.layer == L].groupby(["experiment", "kind"]):
                if exp == "frames_v1":
                    add(experiment=exp, analysis=kind, gemma_layer=gl, model=m, layer=L,
                        rho=g.rho.mean(), null_p95=g.null_p95.mean(),
                        n_sig=int((g.p < .05).sum()), n_pairs=len(g))
                else:
                    r = g.iloc[0]
                    add(experiment=exp, analysis=kind, gemma_layer=gl, model=m, layer=L,
                        rho=r.rho, p=r.p, null_p95=r.null_p95)
    # ---- WordNet: full joint graph vs IS-A vs random spanning trees
    f = sub(NP / "wordnet_steering", t) / "layer_panels.json"
    if f.exists():
        j = json.load(open(f))
        for gl in MAIN_L:
            r = j[str(mapL(t, gl))]
            add(experiment="wordnet", analysis="full_graph", gemma_layer=gl, model=m, layer=mapL(t, gl),
                rho=r["full"], rho_isa=r["hier"], tree_max=r["tree_max"],
                beats_all_trees=bool(r["full"] > r["tree_max"]))

df = pd.DataFrame(rows)
out = NP / "bigmodel/tables"
out.mkdir(exist_ok=True)
df.to_csv(out / "cross_model_summary.csv", index=False)


def cell(r):
    if r.experiment == "wordnet":
        return f"{r.rho:.3f} ({r.rho - r.rho_isa:+.3f}{' all' if r.beats_all_trees else ''})"
    if r.experiment == "frames_v1":
        return f"{r.rho:.2f} ({int(r.n_sig)}/{int(r.n_pairs)})"
    return f"{r.rho:+.2f}{'*' if r.p < .05 else ' '}"


df["cell"] = df.apply(cell, axis=1)
models = [NAME[t] for t in TAGS if NAME[t] in set(df.model)]
pd.set_option("display.width", 220)
print(f"models: {models}   (* p<.05; WordNet: rho (gain over IS-A, 'all' = beats every sampled tree);"
      f" frames: mean rho (#sig/#pairs))")
for (exp, an), g in df.groupby(["experiment", "analysis"], sort=False):
    t = g.pivot_table(index="gemma_layer", columns="model", values="cell", aggfunc="first")[
        [m for m in models if m in set(g.model)]]
    print(f"\n== {exp} / {an}  (rows: Gemma layer; each model at the same relative depth)")
    print(t.to_string())
print(f"\nwrote {out / 'cross_model_summary.csv'} ({len(df)} rows)")
