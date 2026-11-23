"""cross_model_rsa.py -- the paper's analyses side by side: Gemma-2-2B (the paper), OLMo-3.1-32B, Qwen3-32B.

Layers are matched by relative depth: Gemma layer L -> round(L * n_layers / 26) (model_switch.layer).
Reads only tables the analyses already wrote -- Gemma's in outputs/tables/ and the <tag>/ subfolders
written when each analysis is run with RELGEOM_MODEL=<tag> -- and computes nothing. Models whose tables are
missing are skipped. Paper: Figure 4, Tables 10 and 13.

Writes outputs/tables/cross_model_summary.csv (long format, one row per
experiment x analysis x layer x model) and prints compact tables (* = p < 0.05).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root
from relgeom.models import MODELS   # noqa: E402
from relgeom.paths import TABLES   # noqa: E402

# Temperature MOVES follow the paper's 18-word ladders (phase / precipitation / adjectival, 49 moves;
# relgeom/ladders.py), written by analysis/temperature/moves_rsa.py.
TAGS = [None, "olmo31_32b", "qwen3_32b"]
NAME = {None: "Gemma-2-2B", "olmo31_32b": "OLMo-3.1-32B", "qwen3_32b": "Qwen3-32B"}
TEMP_L, MAIN_L = [0, 1, 2, 3, 4, 5, 10, 15, 20], [5, 10, 15, 20]


def mapL(tag, L):
    return L if tag is None else round(L * MODELS[tag]["n_layers"] / 26)


def sub(base, tag):
    return Path(base) if tag is None else Path(base) / tag


rows = []
add = lambda **k: rows.append(k)

for t in TAGS:
    m = NAME[t]
    # ---- temperature: positions and moves
    f = sub(TABLES, t) / "node_rsa_by_layer_unit.csv"
    if f.exists():
        d = pd.read_csv(f)
        for gl in TEMP_L:
            for u in ("C", "F"):
                r = d[(d.layer == mapL(t, gl)) & (d.unit == u)].iloc[0]
                add(experiment="temperature", analysis=f"positions_{u}", gemma_layer=gl, model=m,
                    layer=mapL(t, gl), rho=r.spearman, p=r.perm_p, null_p95=r.null_p95)
    for gl in TEMP_L:
        f = sub(TABLES, t) / f"move_direction_stats_L{mapL(t, gl)}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        pc = next(c for c in ("p", "perm_p", "p_word") if c in d.columns)
        for u in ("C", "F"):
            r = d[d.unit == u].iloc[0]
            add(experiment="temperature", analysis=f"moves_{u}", gemma_layer=gl, model=m,
                layer=mapL(t, gl), rho=r.rho, p=r[pc], null_p95=r.null_p95)
    # ---- analogies
    for base, fn, gls in ((TABLES, "analogy_rsa.csv", MAIN_L),):
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

df = pd.DataFrame(rows)
out = TABLES
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "cross_model_summary.csv", index=False)


def cell(r):
    if r.experiment == "frames_v1":
        return f"{r.rho:.2f} ({int(r.n_sig)}/{int(r.n_pairs)})"
    return f"{r.rho:+.2f}{'*' if r.p < .05 else ' '}"


df["cell"] = df.apply(cell, axis=1)
models = [NAME[t] for t in TAGS if NAME[t] in set(df.model)]
pd.set_option("display.width", 220)
print(f"models: {models}   (* p<.05; frames: mean rho (#sig/#pairs))")
for (exp, an), g in df.groupby(["experiment", "analysis"], sort=False):
    t = g.pivot_table(index="gemma_layer", columns="model", values="cell", aggfunc="first")[
        [m for m in models if m in set(g.model)]]
    print(f"\n== {exp} / {an}  (rows: Gemma layer; each model at the same relative depth)")
    print(t.to_string())
print(f"\nwrote {out / 'cross_model_summary.csv'} ({len(df)} rows)")
