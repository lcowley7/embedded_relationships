"""compare_steering.py -- steering efficacy side by side: Gemma-2-2B (the paper), OLMo-3.1-32B, Qwen3-32B.

The 32B models were steered on a reduced grid (steer.py via bigmodel/run_steer.sh: 9 of Gemma's 15
factors, the first 6 of its 12 prompts), so Gemma is RE-SCORED on exactly that subset from its
per-row judge scores; all three columns are then the same statistic on the same grid:

    Overall = max over factors of the zero-gated harmonic mean of the three rubric means
              (the judge's lm_judge_rating; 0-2 scale), averaged over vectors.

"never steer" counts vectors whose Overall is 0 at every factor. Layers are matched by relative
depth (Gemma 5/10/15 -> 12/25/37). A (model, layer) column is only filled for a family when every
dataset of that family was evaluated there.

Writes bigmodel/tables/steering_comparison.csv and prints the tables.
Run: PY bigmodel/compare_steering.py
"""
import json
from pathlib import Path

import pandas as pd

NP = Path(__file__).resolve().parents[1]
FACT = [0.0, 0.4, 1.0, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0]     # run_steer.sh GRID, a subset of Gemma's 15
N_PROMPTS = 6
FAMILY = {"bodypart_v1": "WordNet", "h2o_v1": "temperature",
          **{f"temp_c1{s}": "temperature" for s in "abcd"},
          **{f"temp_f1{s}": "temperature" for s in "abcdef"},
          **{d: "wave 1" for d in ["size_imp_v1", "war_arg_v1", "food_ideas_v1", "frames_v1"]},
          **{d: "SCAN" for d in ["scan_arg_war_v1", "scan_career_hill_v1", "scan_idea_food_v1",
                                 "scan_ideas_seeds_v1", "scan_mind_comp_v1", "scan_nation_ship_v1"]}}
COLS = [("Gemma-2-2B", None, 5), ("OLMo-3.1-32B", "olmo31_32b", 12), ("Qwen3-32B", "qwen3_32b", 12),
        ("Gemma-2-2B", None, 10), ("OLMo-3.1-32B", "olmo31_32b", 25), ("Qwen3-32B", "qwen3_32b", 25),
        ("Gemma-2-2B", None, 15), ("OLMo-3.1-32B", "olmo31_32b", 37)]


def hm0(v):
    """AxBench's zero-gated harmonic mean."""
    return 0.0 if (len(v) == 0 or min(v) <= 0) else len(v) / sum(1.0 / x for x in v)


def gemma(ds, L):
    """{concept_id: Overall} re-scored on the 32B grid (9 factors x first 6 prompts)."""
    p = NP / f"data/axbench/{ds}/diffmean_l{L}/eval/per_row_scores.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    d = d[d.factor.round(4).isin([round(f, 4) for f in FACT])]
    first = d.groupby("concept_id").input_id.unique().apply(lambda a: sorted(a)[:N_PROMPTS])
    d = d[[i in first[c] for c, i in zip(d.concept_id, d.input_id)]]
    per = d.groupby(["concept_id", "factor", "rubric"]).score.mean().unstack("rubric")
    return per.apply(lambda r: hm0(list(r.values)), axis=1).groupby("concept_id").max().to_dict()


def model(tag, ds, L):
    p = NP / f"models/{tag}/axbench/{ds}/diffmean_l{L}/eval/results.jsonl"
    if not p.exists():
        return None
    out = {}
    for line in open(p):
        r = json.loads(line)
        out[int(r["concept_id"])] = max(r["results"]["LMJudgeEvaluator"]["DiffMean"]["lm_judge_rating"])
    return out


rows = []
for name, tag, L in COLS:
    for ds, fam in FAMILY.items():
        s = gemma(ds, L) if tag is None else model(tag, ds, L)
        rows.append(dict(model=name, layer=L, depth=L if tag is None else round(L * 26 / 64),
                         family=fam, dataset=ds, evaluated=s is not None,
                         n=len(s) if s else 0, total=sum(s.values()) if s else 0.0,
                         never=sum(v == 0 for v in s.values()) if s else 0))
df = pd.DataFrame(rows)
df.to_csv(NP / "bigmodel/tables/steering_per_dataset.csv", index=False)

g = df.groupby(["family", "model", "layer", "depth"]).agg(
    complete=("evaluated", "all"), n=("n", "sum"), total=("total", "sum"), never=("never", "sum")).reset_index()
g["mean_overall"] = (g.total / g.n).round(3)
g.loc[~g.complete, ["mean_overall", "never"]] = [float("nan"), pd.NA]
g.to_csv(NP / "bigmodel/tables/steering_comparison.csv", index=False)

key = lambda r: f"{r.model.split('-')[0]} L{r.layer}"
g["col"] = g.apply(key, axis=1)
order = [key(pd.Series(dict(model=m, layer=L))) for m, _, L in COLS]
for val, title in [("mean_overall", "Mean judge Overall (0-2), identical 9 factors x 6 prompts"),
                   ("never", "Vectors that never steer (Overall 0 at every factor)")]:
    t = g.pivot(index="family", columns="col", values=val)[order]
    if val == "never":
        t = t.astype("Int64").astype(str).replace("<NA>", "")
    t.insert(0, "vectors", g[g.model == "Gemma-2-2B"].groupby("family").n.max())
    print(f"\n{title}  (blank = family not fully evaluated at that layer)\n")
    print(t.to_string(na_rep=""))
print(f"\nwrote {NP / 'bigmodel/tables/steering_comparison.csv'} and steering_per_dataset.csv")
