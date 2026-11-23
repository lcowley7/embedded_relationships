"""build_scores.py -- one judge-Overall score per vector per layer, for every family (input to the sweeps).

With RELGEOM_MODEL set (model_switch.py) the scores come from that model's own steering
evaluation, at its own depth-matched layers, and land in outputs/tables/<tag>/. Layers the model was
not steered at simply contribute no rows, so the sweeps can see which layers they can run.

Overall = max over the 15 steering factors of lm_judge_rating (harmonic mean with zero gate of
the three rubrics), on 0-2. Exactly the statistic plotted in Appendix A.

Writes outputs/tables/[<tag>/]judge_scores.csv with columns:
    family, layer, concept_id, label, coord, overall
where `coord` is the family's natural coordinate (integer value, temperature, or empty).
"""
import glob, json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # repository root
from relgeom import model_switch   # noqa: E402
from relgeom.paths import AXBENCH, TABLES

DB = str(AXBENCH)          # concept indices: always canonical
OUT = str(model_switch.out_dir(TABLES))
# The analogies were trained at four depths; the temperature families also at layers 0-4.
LAYERS = model_switch.layers([5, 10, 15, 20])
TEMP_LAYERS = model_switch.layers([0, 1, 2, 3, 4, 5, 10, 15, 20])

def overalls(ds, layer):
    """{concept_id: Overall}"""
    p = model_switch.eval_path(ds, layer)
    if not os.path.exists(p):
        return {}
    out = {}
    for line in open(p):
        r = json.loads(line)
        rat = np.array(r["results"]["LMJudgeEvaluator"]["DiffMean"]["lm_judge_rating"], float)
        out[int(r["concept_id"])] = float(np.nanmax(rat))
    return out


def slug_temp(vid):
    s = vid.split("__")[-1]
    return -int(s[3:]) if s.startswith("neg") else int(s)


rows = []
for L in LAYERS:
    # ---- analogies: one family per experiment, keyed by vector_id --
    for ds in ("size_imp_v1", "war_arg_v1", "food_ideas_v1", "frames_v1"):
        ci = pd.read_parquet(f"{DB}/{ds}/concept_index.parquet")
        o = overalls(ds, L)
        for _, r in ci.iterrows():
            cid = int(r.concept_id)
            if cid in o:
                rows.append(dict(family=ds, layer=L, concept_id=cid,
                                 label=str(r.vector_id), coord=np.nan, overall=o[cid]))


for L in TEMP_LAYERS:
    # ---- water words ------------------------------------------------------
    ci = pd.read_parquet(f"{DB}/h2o_v1/concept_index.parquet")
    o = overalls("h2o_v1", L)
    for _, r in ci.iterrows():
        cid = int(r.concept_id)
        if cid in o:
            rows.append(dict(family="h2o_v1", layer=L, concept_id=cid,
                             label=str(r.vector_id).split("__")[-1], coord=np.nan,
                             overall=o[cid]))

    # ---- numeric temperature ladders (shards -> one family, keyed by temp) --
    for fam in ("temp_c1", "temp_f1"):
        for d in sorted(glob.glob(f"{DB}/{fam}?")):
            shard = os.path.basename(d)
            ci = pd.read_parquet(f"{d}/concept_index.parquet")
            o = overalls(shard, L)
            for _, r in ci.iterrows():
                cid = int(r.concept_id)
                if cid in o:
                    t = slug_temp(r.vector_id)
                    rows.append(dict(family=fam, layer=L, concept_id=cid,
                                     label=f"{fam}_{t}", coord=t, overall=o[cid]))

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/judge_scores.csv", index=False)
print(df.groupby(["family", "layer"]).overall.agg(
    n="size", min="min", median="median", mean="mean", max="max",
    zeros=lambda s: int((s == 0).sum())).to_string())
print(f"\nwrote {OUT}/judge_scores.csv  ({len(df)} rows)")
