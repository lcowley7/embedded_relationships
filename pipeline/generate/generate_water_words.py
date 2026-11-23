"""Generate the h2o_v1 dataset (18 substance-identity vectors) — frequency-style STEER.

v = (member | prompt) - (absent | prompt): pos and neg are two continuations of the SAME neutral
scene-request. The absent neg is generated ONCE per prompt and shared across all 18 members
(paired by prompt), so every member vector = mean(pos_m) - mean(neg) over identical prompts and the
only systematic difference is the member. NO pooled/independent pos-neg distributions.

The 18 members tile a (temperature x substance) plane: a frozen cluster (snow/ice/hail/frost/sleet/
slush, all <0°C), liquids (rain/dew/fog/liquid water), gas (steam), and an adjective-water row
(icy..boiling water) — so the geometry can show temperature-clustering (H1) with identity riding
orthogonally (H2). Templates: temperature_templates.py. NB the temp_c field written into each vector
spec is generation-time metadata only; the analyses use relgeom.harness.H2O_TEMP (paper Table 1).

Writes data/inputs/h2o_v1_vectors.jsonl (also under --dry) and data/raw/h2o_v1/.
Usage:
  python pipeline/generate/generate_water_words.py --dry                # plan + vector specs, no API
  python pipeline/generate/generate_water_words.py --n-prompts 4        # smoke (small)
  python pipeline/generate/generate_water_words.py                      # full (n=72; ~1.4k calls)
"""
import json, sys, asyncio, argparse
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root
from relgeom.paths import INPUTS, RAW
from gemini_client import gemini as gem, GEN_MODEL
from gen_utils import parse_numbered
from temperature_templates import (H2O, filter_prompts, render_h2o_promptgen, render_h2o_cont,
                                   render_h2o_absent)

DS = "h2o_v1"

def spec_for(m):
    return dict(vector_id=f"h2o__{m['slug']}", concept_id=m["slug"], perspective="steer",
                concept=f"the response describes a scene that prominently features {m['desc']}",
                domain="h2o_form", magnitude_axis="h2o_identity",
                source_magnitude={"id": "absent", "description": "a neutral scene with no water in any form"},
                target_magnitude={"id": m["slug"], "description": m["label"], "temp_c": m["temp_c"], "group": m["group"]})

async def gen_prompt_pool(n_total):
    pool, per = [], 24
    for _ in range(8):
        if len(pool) >= n_total: break
        raws = await gem([render_h2o_promptgen(per)], temperature=1.0, max_output_tokens=1600, concurrency=4)
        for r in raws:
            pool.extend(parse_numbered(r, per))
        pool = filter_prompts(pool, 10_000)
    return pool

async def complete(reqs, label):
    print(f"  [cont:{label}] {len(reqs)} continuations", flush=True)
    return await gem(reqs, temperature=0.7, max_output_tokens=400, concurrency=12)

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=72)
    ap.add_argument("--n-heldout", type=int, default=30)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    N = args.n_prompts

    specs = [spec_for(m) for m in H2O]
    vp = INPUTS / f"{DS}_vectors.jsonl"
    with open(vp, "w") as f:
        for s in specs: f.write(json.dumps(s) + "\n")
    print(f"[{DS}] {len(specs)} vectors | groups {dict(Counter(m['group'] for m in H2O))} -> {vp.name}")
    print(f"  members: {', '.join(m['slug'] for m in H2O)}")
    print(f"  projected continuation calls ~{N + len(H2O)*N} (1 absent/prompt + {len(H2O)} members) + prompt-gen + heldout")
    if args.dry:
        print("--dry: no API."); return

    pool = await gen_prompt_pool(N + args.n_heldout + 10)
    train_p, held_p = pool[:N], pool[N:N + args.n_heldout]
    print(f"prompt pool: {len(pool)} unique -> {len(train_p)} train, {len(held_p)} heldout")

    # absent negative: ONCE per prompt, shared across all members (paired by prompt)
    neg = await complete([render_h2o_absent(p) for p in train_p], "absent (shared)")

    raw_dir = RAW / DS
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in H2O:
        pos = await complete([render_h2o_cont(m["desc"], p) for p in train_p], f"pos@{m['slug']}")
        vid = f"h2o__{m['slug']}"
        for i, p in enumerate(train_p):
            rows.append(dict(vector_id=vid, prompt=p, completion=pos[i], polarity="positive"))
            rows.append(dict(vector_id=vid, prompt=p, completion=neg[i], polarity="negative"))
    with open(raw_dir / "outputs.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    with open(raw_dir / "heldout_prompts.jsonl", "w") as f:
        for m in H2O:
            vid = f"h2o__{m['slug']}"
            for idx, p in enumerate(held_p):
                f.write(json.dumps(dict(vector_id=vid, heldout_prompt_idx=idx, prompt=p)) + "\n")
    (raw_dir / "meta.json").write_text(json.dumps(dict(
        dataset=DS, n_vectors=len(H2O), n_prompts_default=len(train_p), n_heldout=len(held_p),
        members=[m["slug"] for m in H2O], model=GEN_MODEL), indent=2))
    npos = sum(r["polarity"] == "positive" for r in rows)
    print(f"\nDONE: {len(rows)} train rows ({npos} pos / {len(rows)-npos} neg), "
          f"{len(H2O)*len(held_p)} heldout -> {raw_dir}")
    print(f"  next: python pipeline/axbench/convert_to_axbench.py --vectors {vp} "
          f"--raw-dir {raw_dir} --out-dir data/axbench/{DS}")

if __name__ == "__main__":
    asyncio.run(main())
