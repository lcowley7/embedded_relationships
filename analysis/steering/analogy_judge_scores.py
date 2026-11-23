"""analogy_judge_scores.py -- per-vector steering efficacy of the 94 analogy vectors (input to Figure 7).

Overall = max over the 15 steering factors of the judge's lm_judge_rating (harmonic mean with zero
gate of the three rubrics; 0-2), per vector and layer.

Writes outputs/tables/T2_steering_efficacy_per_vector.csv.
"""
import json, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root
from relgeom.paths import AXBENCH, TABLES, ensure

DS = ["size_imp_v1", "war_arg_v1", "food_ideas_v1", "frames_v1"]
LAYERS = [5, 10, 15, 20]

eff = []
for ds in DS:
    ci = pd.read_parquet(AXBENCH / ds / "concept_index.parquet")
    v_of = dict(zip(ci.concept_id, ci.vector_id))
    for L in LAYERS:
        for l in open(AXBENCH / ds / f"diffmean_l{L}/eval/results.jsonl"):
            r = json.loads(l); d = r["results"]["LMJudgeEvaluator"]["DiffMean"]
            eff.append(dict(ds=ds, layer=L, vector_id=v_of[r["concept_id"]],
                            overall=max(d["lm_judge_rating"])))
eff = pd.DataFrame(eff); eff["cloud"] = eff.vector_id.str.split("__").str[0]
eff.to_csv(ensure(TABLES) / "T2_steering_efficacy_per_vector.csv", index=False)
print(eff.groupby(["ds", "layer"]).overall.mean().unstack().round(2).to_string())
