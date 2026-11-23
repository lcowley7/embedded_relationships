"""Generate the Celsius temperature datasets temp_c1a..d (131 levels, -20..110 °C) for the AxBench pipeline.

Proforma / number-swap (paired on the SAME instrument prompt, v = (B|A) - (A|A)):
  steer@t   prompt=instrument question   neg=absent (no figure)   pos="...{t}°C/°F..."
One shared instrument-prompt pool (Gemini, generated once) is reused across every dataset so
carriers are IDENTICAL across all levels -> consecutive trained vectors difference cleanly
(v_{T+1}-v_T = pure one-degree direction), which the curvature plot needs.

Datasets emitted: temp_c1a..d, °C 1° steps -20..110 in 4 chunks (131 vectors; merged into temp_c1 after
training by pipeline/axbench/recombine_temp_c1.py). Templates: temperature_templates.py.
The Fahrenheit ladder (generate_temp_fahrenheit.py) reuses this prompt pool with no API calls.

Writes data/inputs/<ds>_vectors.jsonl (also under --dry) and data/raw/<ds>/.
Usage:
  python pipeline/generate/generate_temp_celsius.py --dry              # plan + vector specs, no API
  python pipeline/generate/generate_temp_celsius.py --only temp_c1a    # one dataset (smoke)
  python pipeline/generate/generate_temp_celsius.py                    # all 4
"""
import json, sys, asyncio, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root
from relgeom.paths import INPUTS, RAW
from gemini_client import gemini as gem, GEN_MODEL
from gen_utils import parse_numbered
from temperature_templates import (C1_CHUNKS, tslug, render_promptgen, body_c, body_f,
                                   render_absent, filter_prompts)

# (dataset, unit, levels)
DATASETS = [(f"temp_c1{c}", "C", chunk) for c, chunk in zip("abcd", C1_CHUNKS)]

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
            pool.extend(parse_numbered(r, per))
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
        vp = INPUTS / f"{ds}_vectors.jsonl"
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
        raw_dir = RAW / ds
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
            n_heldout=len(held_p), levels=levels, model=GEN_MODEL), indent=2))
        n = len(levels) * len(train_p)
        print(f"[{ds}] wrote {2*n} train rows ({n}/{n}) + {len(levels)*len(held_p)} heldout -> {raw_dir}")
    print("\nnext: convert each, e.g.")
    for ds, _, _ in datasets:
        print(f"  python pipeline/axbench/convert_to_axbench.py --vectors data/inputs/{ds}_vectors.jsonl "
              f"--raw-dir data/raw/{ds} --out-dir data/axbench/{ds}")

if __name__ == "__main__":
    asyncio.run(main())
