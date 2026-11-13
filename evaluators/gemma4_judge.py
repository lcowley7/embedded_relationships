"""Gemma-4-31B-it judge for AxBench-style steering evaluation.

Replaces axbench/evaluators/lm_judge.py's OpenAI/Qwen judge with a local
vLLM-offline Gemma-4 judge. Reads AxBench steering inference parquet,
writes AxBench-shaped results.jsonl + per_row_scores.parquet.

============================================================================
USER EDIT POINT
============================================================================
ALL RUBRIC TEXT IS A MODULE-LEVEL STRING CONSTANT IMMEDIATELY BELOW.
To rephrase a rubric, edit the constant in place — no logic changes needed.

The constants:
  RUBRIC_FLUENCY                   shared by all perspectives
  RUBRIC_INSTRUCTION_FOLLOWING     shared by all perspectives
  RUBRIC_CONCEPT_RELEVANCE         for steer / anti_steer
  RUBRIC_CONCEPT_AS_TOPIC          for steer_ref / anti_steer_ref
  RUBRIC_CONCEPT_AS_REGISTER       for steer_vocab / anti_steer_vocab
  RUBRIC_TARGET_MAGNITUDE          for magnitude (target presence)
  RUBRIC_MAGNITUDE_DIRECTION       for magnitude (src→tgt direction; uses baseline)

Template variables (substituted via str.format at call time):
  {prompt}            the input prompt sent to Gemma-2-2B
  {response}          the steered generation we're judging
  {baseline_response} the factor=0.0 generation (only for magnitude_direction)
  {concept}           the concept description, e.g. "happy (feeling pleasure or contentment)"
  {target_magnitude}  description of target magnitude, e.g. "Tuesday" or "full: every word..."
  {source_magnitude}  description of source magnitude

Output convention every rubric MUST enforce:
  The last line of the model's response is `Rating: [[N]]` where N in {0, 1, 2}.

Score semantics:
  0 = does not satisfy the rubric
  1 = partially satisfies
  2 = clearly satisfies

For anti_steer / anti_steer_ref / anti_steer_vocab the concept-presence score
is INVERTED at parse time (see INVERT_CONCEPT_SCORE below). The rubric itself
still measures "is the concept present?" — we just flip the sign so that
"successful anti-steering" → high final score, matching AxBench's
AlpacaEvalSuppress convention.
============================================================================
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# ============================================================================
# RUBRIC TEXT — edit these constants to rephrase rubrics.
# ============================================================================

RUBRIC_FLUENCY = """\
You are evaluating the FLUENCY of a model's response, independent of whether \
it addresses the prompt or contains any particular concept.

IMPORTANT NOTE ON ENDINGS: Responses are generated under a fixed token budget \
and the end-of-response marker is stripped before you see the text. A response \
may therefore end mid-sentence or without terminal punctuation. This is NOT a \
fluency defect. Evaluate only the quality of the text that is present; do not \
penalise the response for where it stops.

Score 0/1/2:
  0 = The response is genuinely broken — incoherent, severely ungrammatical, \
pathologically repetitive (the same phrase looping), or ends MID-WORD (a word \
is cut off mid-character). These indicate a real generation failure, not just \
an abrupt ending.
  1 = The response is partially fluent — minor grammar issues, some awkward \
phrasing, or noticeable but non-pathological repetition, but most of it reads \
as natural text.
  2 = The response is fully fluent natural English (or the prompt's language) \
throughout the text that was generated. No grammar errors, no pathological \
repetition. Ending mid-sentence is fine for score 2.

Ignore whether the response is on-topic, ignore concept presence, ignore \
whether instructions were followed. Score only the surface fluency.

Prompt:
{prompt}

Response:
{response}

Final line must be exactly: Rating: [[N]] where N is 0, 1, or 2.\
"""

RUBRIC_INSTRUCTION_FOLLOWING = """\
You are evaluating whether a model's response ADDRESSES the prompt — does it \
attempt the task the prompt asks for?

Score 0/1/2:
  0 = The response ignores the prompt, addresses an unrelated topic, or \
refuses without engaging.
  1 = The response partially addresses the prompt — touches on it but misses \
the main ask, drifts off-topic, or only loosely engages.
  2 = The response addresses the prompt directly and on-topic, attempting \
the task it asks for. (Quality of the answer is not scored here — only that it tries.)

Ignore fluency, ignore concept presence. Score only whether the response is \
ON-TASK and attempts to address the prompt.

Prompt:
{prompt}

Response:
{response}

Final line must be exactly: Rating: [[N]] where N is 0, 1, or 2.\
"""

RUBRIC_CONCEPT_RELEVANCE = """\
You are evaluating whether a model's response engages with a specific concept \
in any visible form — as topic, as content, as emotion, as imagery, anywhere.

Concept: {concept}

Score 0/1/2:
  0 = The response does not engage with the concept at all. No mention, no \
allusion, no implicit reference.
  1 = The response engages with the concept partially — a passing mention, \
indirect allusion, or one-sentence aside.
  2 = The response clearly and substantively engages with the concept — it \
is a central or recurring feature of the response.

Ignore fluency, ignore whether the prompt was addressed. Score only the \
presence of the concept in the response. Feel free to reason on lines before your answer to help you score.

Prompt:
{prompt}

Response:
{response}

Final line must be exactly: Rating: [[N]] where N is 0, 1, or 2.\
"""

RUBRIC_CONCEPT_AS_TOPIC = """\
You are evaluating whether a model's response engages with a concept AS A TOPIC \
or REFERENT — i.e. talks ABOUT the concept, describes it, references it, names \
it. (As opposed to using it as a register or style.)

Concept: {concept}

Score 0/1/2:
  0 = The concept is not used as a topic. The response does not refer to, \
mention, allude to, or describe the concept. (The response might still happen \
to be IN the concept's register, but that's separate.)
  1 = The concept appears as a topic partially — a passing reference, a \
single mention, or an indirect allusion.
  2 = The concept is clearly used as a topic — explicitly named, described, \
or substantially referenced.

Ignore fluency, ignore instruction-following, ignore whether the response is \
IN the concept's register. Score only the presence of the concept as a \
discussed topic/referent. Feel free to reason on lines before your answer to help you score.

Prompt:
{prompt}

Response:
{response}

Final line must be exactly: Rating: [[N]] where N is 0, 1, or 2.\
"""

RUBRIC_CONCEPT_AS_REGISTER = """\
You are evaluating whether a model's response is IN the concept's REGISTER or \
STYLE — i.e. performs the concept, uses its linguistic markers, embodies it. \
(As opposed to talking ABOUT the concept.)

Concept: {concept}

Score 0/1/2:
  0 = The response is not in the concept's register. The prose does not \
exhibit the style, vocabulary, or affective markers associated with the \
concept. (The response might still mention the concept by name, but that's \
separate.)
  1 = The response is partially in the concept's register — some markers \
appear, but the overall register is neutral or mixed.
  2 = The response is clearly in the concept's register — the style, \
vocabulary, or affective markers of the concept are pervasive in the prose itself.

Ignore fluency, ignore instruction-following, ignore whether the concept is \
discussed AS A TOPIC. Score only the register/style of the response. 
Feel free to reason on lines before your answer to help you score.

Prompt:
{prompt}

Response:
{response}

Final line must be exactly: Rating: [[N]] where N is 0, 1, or 2.\
"""

RUBRIC_TARGET_MAGNITUDE = """\
You are evaluating whether a model's response exhibits a concept AT A SPECIFIC \
TARGET MAGNITUDE.

Concept: {concept}
Target magnitude: {target_magnitude}

Score 0/1/2:
  0 = The response does not exhibit the concept at the target magnitude. \
(It may exhibit a different magnitude, or no concept at all.)
  1 = The response partially exhibits the concept at the target magnitude — \
mixed signals, partial alignment, or near-but-not-at the target.
  2 = The response clearly exhibits the concept at the target magnitude. \
The magnitude is unambiguously at the described target level.

Ignore fluency, ignore instruction-following. Score only whether the concept \
appears AT THE TARGET MAGNITUDE. Feel free to reason on lines before your answer to help you score.

Prompt:
{prompt}

Response:
{response}

Final line must be exactly: Rating: [[N]] where N is 0, 1, or 2.\
"""

RUBRIC_MAGNITUDE_DIRECTION = """\
You are evaluating whether a model's STEERED response has moved the concept's \
magnitude from a source magnitude/intensity toward a target magnitude/intensity, compared to an unsteered baseline.

Concept: {concept}
Source magnitude: {source_magnitude}
Target magnitude: {target_magnitude}

Baseline (unsteered) response:
{baseline_response}

Steered response:
{response}

Score 0/1/2:
  0 = The steered response is no closer to the target magnitude than the \
baseline (or has moved AWAY from target, or has no concept at all).
  1 = The steered response is partially closer to the target — some \
movement in the right direction, but does not clearly reach the target.
  2 = The steered response is clearly closer to the target than the baseline \
— it has moved meaningfully from source toward target.

Important: judge MOVEMENT (steered vs. baseline), not absolute target \
presence. If the baseline already exhibits the target magnitude, the steered \
response cannot improve much — note that case but score it on movement, not \
final state. Feel free to reason on lines before your answer to help you score.

Final line must be exactly: Rating: [[N]] where N is 0, 1, or 2.\
"""


# ============================================================================
# Per-perspective rubric routing.
# Read concept_index.parquet → perspective → list of rubric names below.
# ============================================================================

PERSPECTIVE_RUBRICS: dict[str, list[str]] = {
    "steer":              ["fluency", "instruction", "concept_relevance"],
    "anti_steer":         ["fluency", "instruction", "concept_relevance"],
    "steer_ref":          ["fluency", "instruction", "concept_as_topic"],
    "anti_steer_ref":     ["fluency", "instruction", "concept_as_topic"],
    "steer_vocab":        ["fluency", "instruction", "concept_as_register"],
    "anti_steer_vocab":   ["fluency", "instruction", "concept_as_register"],
    "magnitude":          ["fluency", "instruction", "target_magnitude", "magnitude_direction"],
}

# Perspectives where the concept-presence-style score is INVERTED at parse time.
# Successful anti-steering = concept absent in response = low raw score → invert → high final.
INVERT_CONCEPT_SCORE: set[str] = {"anti_steer", "anti_steer_ref", "anti_steer_vocab"}

# Map rubric name → the constant defined above.
RUBRIC_BY_NAME: dict[str, str] = {
    "fluency":              RUBRIC_FLUENCY,
    "instruction":          RUBRIC_INSTRUCTION_FOLLOWING,
    "concept_relevance":    RUBRIC_CONCEPT_RELEVANCE,
    "concept_as_topic":     RUBRIC_CONCEPT_AS_TOPIC,
    "concept_as_register":  RUBRIC_CONCEPT_AS_REGISTER,
    "target_magnitude":     RUBRIC_TARGET_MAGNITUDE,
    "magnitude_direction":  RUBRIC_MAGNITUDE_DIRECTION,
}

# Which rubrics use the concept-presence score that gets inverted for anti_*.
# (Only the "is the concept present" rubric is inverted — fluency / instruction
# / target_magnitude / magnitude_direction are NOT inverted regardless of perspective.)
INVERTIBLE_RUBRICS: set[str] = {"concept_relevance", "concept_as_topic", "concept_as_register"}


# ============================================================================
# Score parsing — verbatim from axbench/evaluators/lm_judge.py:25-51
# ============================================================================

_RATING_RE = re.compile(r"Rating:\s*\[\[\s*([0-2](?:\.\d+)?)\s*\]\]")


def parse_rating(text: str) -> float:
    """Parse 'Rating: [[N]]' from a judge response. Returns 0.0 if unparseable."""
    if not text:
        return 0.0
    match = _RATING_RE.search(text)
    if not match:
        return 0.0
    try:
        score = float(match.group(1))
    except ValueError:
        return 0.0
    if score < 0.0 or score > 2.0:
        return 0.0
    return score


# ============================================================================
# Aggregation — harmonic-mean-with-zero-gate, per axbench/evaluators/lm_judge.py:128-147
# ============================================================================

def harmonic_mean_with_zero_gate(scores: Iterable[float]) -> float:
    """If ANY score is 0, return 0 (strict gate).
    Otherwise return the harmonic mean of the scores."""
    scores = list(scores)
    if not scores:
        return 0.0
    if any(s <= 0 for s in scores):
        return 0.0
    n = len(scores)
    return n / sum(1.0 / s for s in scores)


# ============================================================================
# Stubs for the rest of the judge driver — implemented during stage 5.
# Currently raises NotImplementedError so this file is import-safe but obviously
# not a complete judge until built out.
# ============================================================================

@dataclass
class JudgeArgs:
    steering_data: str       # path to steering_data.parquet
    concept_index: str       # path to concept_index.parquet
    out_dir: str             # write results.jsonl + per_row_scores.parquet here
    tp: int = 1
    model: str = "google/gemma-4-31B-it"
    # Defaults sized to actual judge behavior: rubric responses are 70 toks median,
    # ~125 max in observed runs; judge prompts (rubric + baseline + steered response)
    # fit in ~2k toks. Earlier defaults (8192 / 1024) reserved 8× more KV-cache slot
    # than was ever used, capping in-flight concurrency at ~1.5x on the 31B judge
    # and causing widespread 2h SLURM timeouts. Smaller budgets here ≈ 4x concurrency.
    max_model_len: int = 3072
    judge_max_tokens: int = 256
    seed: int = 0
    concept_group: str | None = None  # if set, filter steering + concept_index to this group (for SLURM fan-out)
    vector_slice: str | None = None   # "start:end" — further restrict to concept_ids[start:end] within group (sub-array fan-out)
    method: str = "DiffMean"  # AxBench model name — column = f"{method}_steered_generation", results key = method


def dedupe_chat_template(chat_template: str) -> str:
    """Defensive de-duplication for a corrupted Gemma-4 chat template.

    The google/gemma-4-31B-it revision pulled on 2026-05-20 ships a
    chat_template.jinja that is the real template concatenated 4x (69864 bytes
    vs the clean 17466). A concatenated template makes apply_chat_template emit
    the conversation N times, inflating every prompt Nx — which silently blew
    past SLURM time budgets and max_model_len in the T3 eval runs.

    The legit template defines the `format_parameters` macro exactly once at the
    top. If it appears more than once, the template is concatenated; keep only
    the text before the 2nd occurrence (= one clean copy). Returns the input
    unchanged when it is not duplicated.
    """
    if not chat_template:
        return chat_template
    marker = "{%- macro format_parameters"
    positions = [i for i in range(len(chat_template)) if chat_template.startswith(marker, i)]
    if len(positions) > 1:
        return chat_template[:positions[1]]
    return chat_template


def render_judge_prompts(tokenizer, prompts):
    """Pre-render judge prompts to chat-formatted strings using a de-duplicated
    template. We DO NOT use vLLM's llm.chat() — it applies the (corrupted)
    bundled template and 4x-inflates prompts. Instead pre-render here and feed
    raw strings to llm.generate(), which tokenizes them verbatim."""
    clean_template = dedupe_chat_template(tokenizer.chat_template)
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p["prompt"]}],
            tokenize=False, add_generation_prompt=True, chat_template=clean_template,
        )
        for p in prompts
    ]


def build_judge_prompts_for_row(row, ci_row, baseline_lookup, method: str = "DiffMean"):
    """For one steered row × applicable rubrics, produce a list of judge-prompt
    dicts: {key: (concept_id, input_id, factor, rubric_name), prompt, perspective}.
    """
    perspective = ci_row["perspective"]
    rubrics = PERSPECTIVE_RUBRICS[perspective]
    concept_str = ci_row.get("concept_description") or ci_row["concept_group"]

    out = []
    for rubric_name in rubrics:
        rubric_text = RUBRIC_BY_NAME[rubric_name]
        params = {
            "prompt": row["original_prompt"],
            "response": row[f"{method}_steered_generation"],
            "concept": concept_str,
        }
        if rubric_name in ("target_magnitude", "magnitude_direction"):
            params["target_magnitude"] = ci_row.get("target_magnitude_desc") or ""
            params["source_magnitude"] = ci_row.get("source_magnitude_desc") or ""
        if rubric_name == "magnitude_direction":
            baseline_key = (int(row["concept_id"]), int(row["input_id"]), 0.0)
            params["baseline_response"] = baseline_lookup.get(baseline_key, "") or ""

        try:
            prompt_text = rubric_text.format(**params)
        except KeyError as e:
            # Missing template variable — surface as 0-score by emitting a stub prompt.
            print(f"[gemma4_judge] rubric template missing variable {e!s} for rubric={rubric_name}", flush=True)
            prompt_text = rubric_text  # un-formatted; judge will likely produce 0
        out.append({
            "key": (int(row["concept_id"]), int(row["input_id"]), float(row["factor"]), rubric_name),
            "prompt": prompt_text,
            "perspective": perspective,
        })
    return out


# AxBench-style result keys per rubric — preserved so any downstream analysis
# written against upstream's LMJudgeEvaluator schema keeps working.
AXBENCH_RESULT_KEY = {
    "fluency":             "fluency_ratings",
    "instruction":         "relevance_instruction_ratings",
    "concept_relevance":   "relevance_concept_ratings",
    "concept_as_topic":    "relevance_concept_topic_ratings",
    "concept_as_register": "relevance_concept_register_ratings",
    "target_magnitude":    "target_magnitude_ratings",
    "magnitude_direction": "magnitude_direction_ratings",
}


def run_judge(args: JudgeArgs):
    """Top-level driver. Loads steering data, builds judge prompts for each
    (row × applicable rubric), batches via vLLM, parses, aggregates, writes
    AxBench-shaped results.jsonl + per_row_scores.parquet."""
    import pandas as pd
    from vllm import LLM, SamplingParams

    steering = pd.read_parquet(args.steering_data)
    concept_index = pd.read_parquet(args.concept_index)

    if args.concept_group is not None:
        # SLURM array fan-out: process one concept_group per task. Both the
        # steering data and the concept_index are filtered to this group's ids.
        keep_ids = set(concept_index[concept_index.concept_group == args.concept_group]["concept_id"].astype(int))
        if not keep_ids:
            raise ValueError(f"concept_group={args.concept_group!r} matched 0 rows in concept_index")
        concept_index = concept_index[concept_index["concept_id"].isin(keep_ids)].copy()
        steering = steering[steering["concept_id"].astype(int).isin(keep_ids)].copy()
        print(f"[gemma4_judge] filtered to concept_group={args.concept_group!r}: "
              f"{len(keep_ids)} concepts, {len(steering)} steering rows", flush=True)

    if args.vector_slice is not None:
        # SLURM sub-array fan-out within a concept_group. Sort concept_ids in the
        # (already filtered) concept_index, then take [start:end]. Baseline rows
        # (factor=0.0) are also restricted to the slice — magnitude_direction
        # needs both members of any (cid, iid, 0.0) pair in the same task.
        start, end = (int(x) for x in args.vector_slice.split(":"))
        sorted_ids = sorted(int(c) for c in concept_index["concept_id"].unique())
        sliced_ids = set(sorted_ids[start:end])
        if not sliced_ids:
            raise ValueError(f"vector_slice {args.vector_slice!r} of {len(sorted_ids)} concepts yielded 0 ids")
        concept_index = concept_index[concept_index["concept_id"].astype(int).isin(sliced_ids)].copy()
        steering = steering[steering["concept_id"].astype(int).isin(sliced_ids)].copy()
        print(f"[gemma4_judge] vector_slice={args.vector_slice!r}: "
              f"{len(sliced_ids)} concepts of {len(sorted_ids)} in group, "
              f"{len(steering)} steering rows", flush=True)

    ci_by_id = {int(r["concept_id"]): r for _, r in concept_index.iterrows()}

    # Baseline lookup keyed on (concept_id, input_id, 0.0) — used by magnitude_direction rubric.
    baseline_df = steering[steering["factor"] == 0.0]
    baseline_lookup = {
        (int(r["concept_id"]), int(r["input_id"]), 0.0): r[f"{args.method}_steered_generation"]
        for _, r in baseline_df.iterrows()
    }

    # Build all judge prompts up-front. For 138k steering rows × ~3.3 rubrics avg
    # this is ~460k prompts at ~1-2KB each = ~1GB RAM — fits comfortably.
    all_prompts = []
    for _, row in steering.iterrows():
        cid = int(row["concept_id"])
        ci_row = ci_by_id.get(cid)
        if ci_row is None:
            print(f"[gemma4_judge] concept_id {cid} not in concept_index.parquet — skipping", flush=True)
            continue
        all_prompts.extend(build_judge_prompts_for_row(row, ci_row, baseline_lookup, method=args.method))
    print(f"[gemma4_judge] built {len(all_prompts)} judge prompts for {len(steering)} steered rows", flush=True)

    # vLLM offline batch generate.
    # NOTE: do not use llm.chat() — it applies the bundled chat template, which
    # is corrupted (concatenated 4x) in the 2026-05-20 gemma-4-31B-it revision
    # and 4x-inflates every prompt. Pre-render with a DE-DUPLICATED template and
    # feed raw strings to llm.generate(), which tokenizes them verbatim.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rendered = render_judge_prompts(tokenizer, all_prompts)
    print(f"[gemma4_judge] pre-rendered {len(rendered)} chat prompts (de-duplicated template)",
          flush=True)

    llm = LLM(
        model=args.model, max_model_len=args.max_model_len, dtype="bfloat16",
        tensor_parallel_size=args.tp, gpu_memory_utilization=0.92,
        max_num_seqs=64,
        limit_mm_per_prompt={"image": 0, "audio": 0},
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.judge_max_tokens, n=1, seed=args.seed)
    outputs = llm.generate(rendered, sampling_params=sp, use_tqdm=True)

    # Parse, invert for anti_* perspectives, collect per-row scores.
    records = []
    for meta, out in zip(all_prompts, outputs):
        text = out.outputs[0].text
        raw_score = parse_rating(text)
        cid, iid, factor, rubric_name = meta["key"]
        perspective = meta["perspective"]
        if perspective in INVERT_CONCEPT_SCORE and rubric_name in INVERTIBLE_RUBRICS:
            score = 2.0 - raw_score
        else:
            score = raw_score
        records.append({
            "concept_id":  cid,
            "input_id":    iid,
            "factor":      factor,
            "rubric":      rubric_name,
            "raw_score":   raw_score,
            "score":       score,
            "perspective": perspective,
            "judge_text":  text,
        })
    per_row_df = pd.DataFrame(records)

    # Baseline-confound flag: magnitude rows where factor=0.0 magnitude_direction rubric
    # already scores 2 (i.e. unsteered output is already at target magnitude). These rows
    # are excluded from headline metrics in analysis; rate is reported.
    confounded_pairs = set()
    for _, r in per_row_df[per_row_df.rubric == "magnitude_direction"].iterrows():
        if r["factor"] == 0.0 and r["raw_score"] >= 2.0:
            confounded_pairs.add((r["concept_id"], r["input_id"]))
    per_row_df["baseline_confounded"] = per_row_df.apply(
        lambda r: ((r["concept_id"], r["input_id"]) in confounded_pairs
                   and ci_by_id[r["concept_id"]]["perspective"] == "magnitude"),
        axis=1,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_row_df.to_parquet(out_dir / "per_row_scores.parquet", engine="pyarrow", compression="snappy")

    # Per-concept aggregation: per factor, harmonic-mean-with-zero-gate across rubrics.
    results = []
    for cid, ci_row in ci_by_id.items():
        perspective = ci_row["perspective"]
        rubrics = PERSPECTIVE_RUBRICS[perspective]
        cid_rows = per_row_df[per_row_df.concept_id == cid]
        if len(cid_rows) == 0:
            continue
        factor_list = sorted(cid_rows["factor"].unique())

        per_rubric_means: dict[str, list[float]] = {}
        for rubric_name in rubrics:
            scores_by_factor = []
            for f in factor_list:
                sel = cid_rows[(cid_rows.factor == f) & (cid_rows.rubric == rubric_name)]
                if perspective == "magnitude":
                    sel = sel[~sel["baseline_confounded"]]
                m = float(sel["score"].mean()) if len(sel) > 0 else 0.0
                scores_by_factor.append(m)
            per_rubric_means[rubric_name] = scores_by_factor

        lm_judge_rating = []
        for fi in range(len(factor_list)):
            per_factor_scores = [per_rubric_means[rn][fi] for rn in rubrics]
            lm_judge_rating.append(harmonic_mean_with_zero_gate(per_factor_scores))

        method_block = {
            "lm_judge_rating": lm_judge_rating,
            "factor":          [float(f) for f in factor_list],
        }
        for rubric_name in rubrics:
            method_block[AXBENCH_RESULT_KEY[rubric_name]] = per_rubric_means[rubric_name]

        results.append({
            "concept_id": cid,
            "perspective": perspective,
            "results": {"LMJudgeEvaluator": {args.method: method_block}},
        })

    with open(out_dir / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"[gemma4_judge] wrote {len(per_row_df)} per-row scores and "
          f"{len(results)} concept aggregates to {out_dir}", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steering-data", required=True)
    p.add_argument("--concept-index", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--concept-group", default=None,
                   help="If set, filter to this concept_group. Used for SLURM array fan-out.")
    p.add_argument("--vector-slice", default=None,
                   help="If set as 'start:end', further restrict to sorted_concept_ids[start:end] "
                        "within the filtered concept_group. Used for SLURM sub-array fan-out when "
                        "a single group is too large for the time budget.")
    p.add_argument("--method", default="DiffMean",
                   help="AxBench model name. Steering column = f'{method}_steered_generation', "
                        "results.jsonl key = method. Defaults to 'DiffMean' so pending eval tasks "
                        "stay zero-diff. Pass 'PreferenceVector' for the RePS parallel run.")
    p.add_argument("--judge-max-tokens", type=int, default=1024,
                   help="Max new tokens for the judge model's response (default 1024). "
                        "512 caused silent zeros on magnitude_direction (78%% of zeros were "
                        "judge-truncated, confirmed 2026-05-16). 1024 with max_model_len=8192 "
                        "gives adequate headroom for CoT reasoning + rating line.")
    args = p.parse_args()
    run_judge(JudgeArgs(
        steering_data=args.steering_data,
        concept_index=args.concept_index,
        out_dir=args.out_dir,
        tp=args.tp,
        concept_group=args.concept_group,
        vector_slice=args.vector_slice,
        method=args.method,
        judge_max_tokens=args.judge_max_tokens,
    ))
