"""rubrics.py -- the steering judge's rubric text, routing, score parsing and aggregation.

Used by gemini_judge.py (the judge in the paper: gemini-3.1-flash-lite). Every vector in the paper
has perspective "steer", so it is judged on three rubrics: RUBRIC_FLUENCY, RUBRIC_INSTRUCTION_FOLLOWING
and RUBRIC_CONCEPT_RELEVANCE (reproduced in prompts/judge_rubrics.md). The other rubrics are routing for
perspectives the paper does not use.

Template variables (substituted via str.format at call time):
  {prompt}            the held-out prompt
  {response}          the steered generation being judged
  {concept}           the concept description

Every rubric ends with `Rating: [[N]]`, N in {0, 1, 2}: 0 = does not satisfy, 1 = partially, 2 = clearly.
Scores are combined per generation by a harmonic mean with a zero gate (as AxBench).
"""
from __future__ import annotations

import re
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
            print(f"[rubrics] rubric template missing variable {e!s} for rubric={rubric_name}", flush=True)
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


