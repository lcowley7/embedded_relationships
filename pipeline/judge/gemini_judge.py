"""gemini_judge.py -- the steering judge (gemini-3.1-flash-lite) for every model in the paper.

Rubric text, routing, parsing and aggregation come from rubrics.py; this script only runs the async
Gemini calls. Writes per_row_scores.parquet (one row per generation x rubric) and an AxBench-shaped
results.jsonl (per concept, the ratings at each steering factor).

Key (never logged): env GEMINI_API_KEY, else file ~/.api/gemini_api_key.

Usage:
  python gemini_judge.py --steering-data <steering.parquet> \
      --concept-index <concept_index.parquet> --out-dir <eval_dir> \
      [--model gemini-3.1-flash-lite] [--concurrency 48] [--limit N (smoke)]
"""
from __future__ import annotations
import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rubrics import (  # noqa: E402
    PERSPECTIVE_RUBRICS, INVERT_CONCEPT_SCORE, INVERTIBLE_RUBRICS,
    AXBENCH_RESULT_KEY, build_judge_prompts_for_row, parse_rating,
    harmonic_mean_with_zero_gate,
)
import json  # noqa: E402


def load_api_key() -> str:
    k = os.environ.get("GEMINI_API_KEY")
    if k:
        return k.strip()
    for p in (Path.home() / ".api" / "gemini_api_key",):
        if p.exists():
            return p.read_text().strip()
    raise SystemExit("No Gemini API key: set GEMINI_API_KEY or write ~/.api/gemini_api_key")


async def _judge_all(prompts, model, concurrency, max_output_tokens, thinking_budget):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=load_api_key())
    cfg_kwargs = dict(temperature=0.0, max_output_tokens=max_output_tokens)
    if thinking_budget is not None:
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    config = types.GenerateContentConfig(**cfg_kwargs)

    sem = asyncio.Semaphore(concurrency)
    texts: list[str] = [""] * len(prompts)
    done = {"n": 0}
    t0 = time.time()

    async def one(i, prompt):
        async with sem:
            for attempt in range(5):
                try:
                    r = await client.aio.models.generate_content(
                        model=model, contents=prompt, config=config)
                    texts[i] = r.text or ""
                    break
                except Exception as e:  # transient API errors -> backoff
                    if attempt == 4:
                        print(f"[gemini] prompt {i} failed after retries: {e}", flush=True)
                        texts[i] = ""
                    else:
                        await asyncio.sleep(1.5 * (attempt + 1))
            done["n"] += 1
            if done["n"] % 500 == 0:
                rate = done["n"] / max(time.time() - t0, 1e-6)
                print(f"  [{done['n']}/{len(prompts)}] {rate:.1f}/s "
                      f"ETA={(len(prompts)-done['n'])/max(rate,1e-6)/60:.1f}m", flush=True)

    await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts)))
    return texts


def run(args):
    steering = pd.read_parquet(args.steering_data)
    ci = pd.read_parquet(args.concept_index)
    ci_by_id = {int(r["concept_id"]): r for _, r in ci.iterrows()}

    baseline_df = steering[steering["factor"] == 0.0]
    baseline_lookup = {
        (int(r["concept_id"]), int(r["input_id"]), 0.0): r[f"{args.method}_steered_generation"]
        for _, r in baseline_df.iterrows()
    }

    all_prompts = []
    for _, row in steering.iterrows():
        ci_row = ci_by_id.get(int(row["concept_id"]))
        if ci_row is None:
            continue
        all_prompts.extend(build_judge_prompts_for_row(row, ci_row, baseline_lookup, method=args.method))
    if args.limit:
        all_prompts = all_prompts[:args.limit]
    print(f"[gemini] {len(all_prompts)} judge prompts ({len(steering)} steered rows), "
          f"model={args.model} concurrency={args.concurrency}", flush=True)

    texts = asyncio.run(_judge_all(
        [p["prompt"] for p in all_prompts], args.model, args.concurrency,
        args.max_output_tokens, args.thinking_budget))

    records = []
    n_unparsed = 0
    for meta, text in zip(all_prompts, texts):
        raw = parse_rating(text)
        if not text:
            n_unparsed += 1
        cid, iid, factor, rubric = meta["key"]
        persp = meta["perspective"]
        score = (2.0 - raw) if (persp in INVERT_CONCEPT_SCORE and rubric in INVERTIBLE_RUBRICS) else raw
        records.append({"concept_id": cid, "input_id": iid, "factor": factor, "rubric": rubric,
                        "raw_score": raw, "score": score, "perspective": persp, "judge_text": text})
    per_row = pd.DataFrame(records)
    print(f"[gemini] empty/blocked responses: {n_unparsed}/{len(texts)}", flush=True)

    # Baseline-confound flag (magnitude rows whose factor=0 already at target).
    confounded = set()
    for _, r in per_row[per_row.rubric == "magnitude_direction"].iterrows():
        if r["factor"] == 0.0 and r["raw_score"] >= 2.0:
            confounded.add((r["concept_id"], r["input_id"]))
    per_row["baseline_confounded"] = per_row.apply(
        lambda r: (r["concept_id"], r["input_id"]) in confounded
        and ci_by_id[r["concept_id"]]["perspective"] == "magnitude", axis=1)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    per_row.to_parquet(out_dir / "per_row_scores.parquet", index=False)

    results = []
    for cid, ci_row in ci_by_id.items():
        rubrics = PERSPECTIVE_RUBRICS[ci_row["perspective"]]
        rows = per_row[per_row.concept_id == cid]
        if len(rows) == 0:
            continue
        factors = sorted(rows["factor"].unique())
        per_rubric = {}
        for rn in rubrics:
            vals = []
            for f in factors:
                sel = rows[(rows.factor == f) & (rows.rubric == rn)]
                if ci_row["perspective"] == "magnitude":
                    sel = sel[~sel["baseline_confounded"]]
                vals.append(float(sel["score"].mean()) if len(sel) else 0.0)
            per_rubric[rn] = vals
        lm = [harmonic_mean_with_zero_gate([per_rubric[rn][fi] for rn in rubrics])
              for fi in range(len(factors))]
        block = {"lm_judge_rating": lm, "factor": [float(f) for f in factors]}
        for rn in rubrics:
            block[AXBENCH_RESULT_KEY[rn]] = per_rubric[rn]
        results.append({"concept_id": cid, "perspective": ci_row["perspective"],
                        "results": {"LMJudgeEvaluator": {args.method: block}}})
    with open(out_dir / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[gemini] wrote {len(per_row)} per-row scores + {len(results)} aggregates -> {out_dir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steering-data", required=True)
    ap.add_argument("--concept-index", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--method", default="DiffMean")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--max-output-tokens", type=int, default=2048)
    ap.add_argument("--thinking-budget", type=int, default=0,
                    help="0 disables Gemini 'thinking' (default; rubrics need only a short CoT + rating). "
                         "Set -1 for dynamic, or None via not passing.")
    ap.add_argument("--limit", type=int, default=None, help="smoke: judge only first N prompts")
    args = ap.parse_args()
    run(args)
