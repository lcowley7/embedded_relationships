"""Generate the temperature NUMERIC proforma datasets (steer ladders) for the AxBench pipeline.

Proforma / number-swap (paired on the SAME instrument prompt, v = (B|A) - (A|A)):
  steer@t   prompt=instrument question   neg=absent (no figure)   pos="...{t}°C/°F..."
One shared instrument-prompt pool (Gemini, generated once) is reused across every dataset so
carriers are IDENTICAL across all levels -> consecutive trained vectors difference cleanly
(v_{T+1}-v_T = pure one-degree direction), which the curvature plot needs.

Datasets emitted:
  temp_c5   °C 5° steps  -20..110            (27)
  temp_f5   °F = c5 grid converted (twins)   (27)
  temp_c1a..d  °C 1° steps -20..110, 4 chunks (131 total)  -- the heavy/deferred sweep
Templates drawn from temperature_v1.ipynb (source of truth).

Usage:
  python generate_temp_numeric_v1.py --dry              # plan + vector specs, no API
  python generate_temp_numeric_v1.py --only temp_c5     # one dataset (smoke)
  python generate_temp_numeric_v1.py                    # all 6
"""
import json, sys, asyncio, argparse, math
from pathlib import Path

REPO = Path(".")
HT = REPO / "hypothesis_test"
sys.path.insert(0, str(REPO))

_nb = json.loads((HT / "temperature_v1.ipynb").read_text())
def _cell(marker):
    for c in _nb["cells"]:
        if c["cell_type"] == "code" and marker in "".join(c["source"]):
            return "".join(c["source"])
    raise RuntimeError(f"temperature_v1.ipynb has no cell containing {marker!r}")
ns = {}
for _m in ["import pipeline as P", "async def gemini", "C5 =", "TEMP_PROMPTGEN ="]:
    exec(_cell(_m), ns)
ns["MAX_CALLS"] = 100_000

P            = ns["P"]
gem          = ns["gemini"]
C5, F5       = ns["C5"], ns["F5"]
C1_CHUNKS    = ns["C1_CHUNKS"]
tslug        = ns["tslug"]
render_promptgen = ns["render_promptgen"]
body_c, body_f   = ns["body_c"], ns["body_f"]
render_absent    = ns["render_absent"]
filter_prompts   = ns["filter_prompts"]

# (dataset, unit, levels)
DATASETS = [("temp_c5", "C", C5), ("temp_f5", "F", F5)] + \
           [(f"temp_c1{c}", "C", chunk) for c, chunk in zip("abcd", C1_CHUNKS)]

def spec_for(ds, unit, t):
    sym = "°C" if unit == "C" else "°F"
    slug = tslug(t)
    tgt = {"id": slug, "description": f"{t}{sym}", "value": t, "unit": unit}
    if unit == "F":
        tgt["c_twin"] = round((t - 32) * 5 / 9)
    return dict(vector_id=f"{ds}__steer__{slug}", concept_id=slug, perspective="steer",
                concept=f"the response states a temperature of about {t}{sym}",
                domain="temperature_reading", magnitude_axis=f"temperature_{unit.lower()}",
                source_magnitude={"id": "absent", "description": "no temperature stated"},
                target_magnitude=tgt)

def body(unit, i, t):
    return body_c(i, t) if unit == "C" else body_f(i, t)

async def gen_prompt_pool(n_total):
    pool, per = [], 24
    for _ in range(8):
        if len(pool) >= n_total: break
        raws = await gem([render_promptgen(per)], temperature=1.0, max_output_tokens=1600, concurrency=4)
        for r in raws:
            pool.extend(P.parse_numbered(r, per))
        pool = filter_prompts(pool, 10_000)   # dedup across batches
    return pool

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=72)
    ap.add_argument("--n-heldout", type=int, default=30)
    ap.add_argument("--only", default=None, help="one dataset name (smoke)")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    datasets = [d for d in DATASETS if (args.only is None or d[0] == args.only)]
    if not datasets:
        sys.exit(f"--only {args.only}: not in {[d[0] for d in DATASETS]}")

    # vector specs (always written)
    for ds, unit, levels in datasets:
        specs = [spec_for(ds, unit, t) for t in levels]
        vp = REPO / "inputs" / f"{ds}_vectors.jsonl"
        with open(vp, "w") as f:
            for s in specs: f.write(json.dumps(s) + "\n")
        print(f"[{ds}] {len(specs)} vectors ({unit}, {levels[0]}..{levels[-1]}) -> {vp.name}")
    total = sum(len(lv) for _, _, lv in datasets)
    print(f"total vectors: {total} | proforma -> only the prompt pool costs API")
    if args.dry:
        print("--dry: no API."); return

    pool = await gen_prompt_pool(args.n_prompts + args.n_heldout + 10)
    train_p, held_p = pool[:args.n_prompts], pool[args.n_prompts:args.n_prompts + args.n_heldout]
    print(f"prompt pool: {len(pool)} unique -> {len(train_p)} train, {len(held_p)} heldout (shared across all datasets)")

    for ds, unit, levels in datasets:
        raw_dir = REPO / "data" / "raw" / ds
        raw_dir.mkdir(parents=True, exist_ok=True)
        with open(raw_dir / "outputs.jsonl", "w") as f:
            for t in levels:
                vid = f"{ds}__steer__{tslug(t)}"
                for i, p in enumerate(train_p):
                    f.write(json.dumps(dict(vector_id=vid, prompt=p, completion=body(unit, i, t), polarity="positive")) + "\n")
                    f.write(json.dumps(dict(vector_id=vid, prompt=p, completion=render_absent(i), polarity="negative")) + "\n")
        with open(raw_dir / "heldout_prompts.jsonl", "w") as f:
            for t in levels:
                vid = f"{ds}__steer__{tslug(t)}"
                for idx, p in enumerate(held_p):
                    f.write(json.dumps(dict(vector_id=vid, heldout_prompt_idx=idx, prompt=p)) + "\n")
        (raw_dir / "meta.json").write_text(json.dumps(dict(
            dataset=ds, unit=unit, n_vectors=len(levels), n_prompts_default=len(train_p),
            n_heldout=len(held_p), levels=levels, model=ns.get("GEN_MODEL")), indent=2))
        n = len(levels) * len(train_p)
        print(f"[{ds}] wrote {2*n} train rows ({n}/{n}) + {len(levels)*len(held_p)} heldout -> {raw_dir}")
    print("\nnext: convert each, e.g.")
    for ds, _, _ in datasets:
        print(f"  python convert_to_axbench.py --vectors inputs/{ds}_vectors.jsonl "
              f"--raw-dir data/raw/{ds} --out-dir data/axbench/{ds}")

if __name__ == "__main__":
    asyncio.run(main())
