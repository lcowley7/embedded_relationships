"""generate_temp_fahrenheit.py — temp_f1a..f, the dense FAHRENHEIT twin of temp_c1 on F's OWN 1°F grid.

Why: the Celsius manifold magnifies at 0°C/100°C (water's phase boundaries), but 0/100 are
also round numbers. The clean confound-killer is a dense F sweep: if the magnification is PHYSICAL it
must move to 32°F (freeze) and 212°F (boil) — which are NOT round — while the round F values
(0/50/100/150/200°F) should be unremarkable. Sampling F on its own 1°F grid (not at the C-corresponding
points) makes it a genuinely independent measurement.

Design = a TRUE twin of temp_c1: it reuses temp_c1's EXACT shared prompt pool (72 train + 30 heldout)
and the identical negative completions, so the only systematic difference from temp_c1 is the stated
value+unit. Fully LOCAL (templates from temperature_templates.py; no Gemini). Run after
generate_temp_celsius.py, whose data/raw/temp_c1a/ prompt pool it reuses.

Range -10..240°F (251 levels) covers freeze 32, boil 212, and round controls 0/50/100/150/200, with
margin matching temp_c1's physical span (-20..110°C = -4..230°F). Sharded into 6 chunks temp_f1a..f for
SLURM parallelism (same pattern as temp_c1a..d); recombine_temp_f1.py stitches them temp-ordered.

Writes data/inputs/<ds>_vectors.jsonl and data/raw/<ds>/.
Usage:
  python pipeline/generate/generate_temp_fahrenheit.py --dry      # plan only
  python pipeline/generate/generate_temp_fahrenheit.py            # write all 6 chunks (local, no API)
"""
import json, sys, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root
from relgeom.paths import INPUTS, RAW
from temperature_templates import body_f, render_absent, tslug   # F = C with °C->°F
POOL_SRC = RAW / "temp_c1a"     # the shared instrument-prompt pool to reuse

F_LO, F_HI = -10, 240
F_LEVELS = list(range(F_LO, F_HI + 1))                 # 251 levels, 1°F steps
CHUNK_NAMES = ["temp_f1a", "temp_f1b", "temp_f1c", "temp_f1d", "temp_f1e", "temp_f1f"]
# Explicit chunk bounds so every TEST point (freeze 32, boil 212, rounds 0/50/100/150/200) is INTERIOR
# to its chunk, never at a training-shard SEAM (seams at 22|23,64|65,106|107,148|149,190|191 = boring temps).
CHUNK_RANGES = [(-10, 22), (23, 64), (65, 106), (107, 148), (149, 190), (191, 240)]

def chunks(levels):
    return [[t for t in levels if lo <= t <= hi] for lo, hi in CHUNK_RANGES]

def load_pool():
    """temp_c1's shared pool, IN ORDER: 72 train prompts (by first appearance) + 30 heldout (by idx)."""
    seen, train = set(), []
    for line in open(POOL_SRC / "outputs.jsonl"):
        p = json.loads(line)["prompt"]
        if p not in seen:
            seen.add(p); train.append(p)
    held = {}
    for line in open(POOL_SRC / "heldout_prompts.jsonl"):
        r = json.loads(line); held.setdefault(r["heldout_prompt_idx"], r["prompt"])
    held = [held[i] for i in sorted(held)]
    return train, held

def spec_for(ds, t):
    slug = tslug(t)
    return dict(vector_id=f"{ds}__steer__{slug}", concept_id=slug, perspective="steer",
                concept=f"the response states a temperature of about {t}°F",
                domain="temperature_reading", magnitude_axis="temperature_f",
                source_magnitude={"id": "absent", "description": "no temperature stated"},
                target_magnitude={"id": slug, "description": f"{t}°F", "value": t, "unit": "F",
                                  "c_twin": round((t - 32) * 5 / 9)})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    parts = chunks(F_LEVELS)
    assert len(parts) == len(CHUNK_NAMES), f"{len(parts)} parts vs {len(CHUNK_NAMES)} names"
    landmarks = {32: "freeze", 212: "boil"}; rounds = [0, 50, 100, 150, 200]
    print(f"temp_f1: {len(F_LEVELS)} levels {F_LO}..{F_HI}°F (1°F) -> {len(CHUNK_NAMES)} chunks")
    for nm, lv in zip(CHUNK_NAMES, parts):
        tags = [f"{t}({landmarks.get(t,'round')})" for t in lv if t in landmarks or t in rounds]
        print(f"  {nm}: {len(lv)} levels {lv[0]}..{lv[-1]}°F  contains: {tags}")
    if args.dry:
        print("--dry: no files written."); return

    train_p, held_p = load_pool()
    print(f"reusing temp_c1 pool: {len(train_p)} train + {len(held_p)} heldout prompts (true-twin carriers)")
    assert len(train_p) == 72 and len(held_p) == 30, "unexpected pool size"

    for nm, levels in zip(CHUNK_NAMES, parts):
        # vector specs
        vp = INPUTS / f"{nm}_vectors.jsonl"
        with open(vp, "w") as f:
            for t in levels: f.write(json.dumps(spec_for(nm, t)) + "\n")
        # raw train/heldout
        raw = RAW / nm; raw.mkdir(parents=True, exist_ok=True)
        with open(raw / "outputs.jsonl", "w") as f:
            for t in levels:
                vid = f"{nm}__steer__{tslug(t)}"
                for i, p in enumerate(train_p):
                    f.write(json.dumps(dict(vector_id=vid, prompt=p, completion=body_f(i, t), polarity="positive")) + "\n")
                    f.write(json.dumps(dict(vector_id=vid, prompt=p, completion=render_absent(i), polarity="negative")) + "\n")
        with open(raw / "heldout_prompts.jsonl", "w") as f:
            for t in levels:
                vid = f"{nm}__steer__{tslug(t)}"
                for idx, p in enumerate(held_p):
                    f.write(json.dumps(dict(vector_id=vid, heldout_prompt_idx=idx, prompt=p)) + "\n")
        (raw / "meta.json").write_text(json.dumps(dict(
            dataset=nm, unit="F", n_vectors=len(levels), n_prompts_default=len(train_p),
            n_heldout=len(held_p), levels=levels, pool_source="temp_c1a"), indent=2))
        print(f"[{nm}] {len(levels)} vectors -> {2*len(levels)*len(train_p)} train rows + "
              f"{len(levels)*len(held_p)} heldout")
    print("\nnext:")
    print("  for DS in " + " ".join(CHUNK_NAMES) + "; do \\")
    print("    python pipeline/axbench/convert_to_axbench.py --vectors data/inputs/${DS}_vectors.jsonl \\")
    print("       --raw-dir data/raw/${DS} --out-dir data/axbench/${DS}; done")
    print("  python pipeline/axbench/make_configs.py " + " ".join(CHUNK_NAMES))
    print("  # then train / max_act / steer / judge with pipeline/slurm/ (see notebooks/rerun_pipeline.ipynb)")
    print("  # after training: python pipeline/axbench/recombine_temp_f1.py")

if __name__ == "__main__":
    main()
