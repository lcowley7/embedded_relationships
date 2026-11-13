# Steering-evaluation judge rubrics (verbatim)

Judge: `gemini-3.1-flash-lite` via `olmo/gemini_judge.py` (rubric text, routing and parsing live in `evaluators/gemma4_judge.py`). `{prompt}`, `{response}` and `{concept}` are substituted at call time. The three scores (0/1/2) are combined by a harmonic mean with a zero gate; a concept's Overall score is the maximum over the 15 steering factors.

## Fluency

```
You are evaluating the FLUENCY of a model's response, independent of whether it addresses the prompt or contains any particular concept.

IMPORTANT NOTE ON ENDINGS: Responses are generated under a fixed token budget and the end-of-response marker is stripped before you see the text. A response may therefore end mid-sentence or without terminal punctuation. This is NOT a fluency defect. Evaluate only the quality of the text that is present; do not penalise the response for where it stops.

Score 0/1/2:
  0 = The response is genuinely broken — incoherent, severely ungrammatical, pathologically repetitive (the same phrase looping), or ends MID-WORD (a word is cut off mid-character). These indicate a real generation failure, not just an abrupt ending.
  1 = The response is partially fluent — minor grammar issues, some awkward phrasing, or noticeable but non-pathological repetition, but most of it reads as natural text.
  2 = The response is fully fluent natural English (or the prompt's language) throughout the text that was generated. No grammar errors, no pathological repetition. Ending mid-sentence is fine for score 2.

Ignore whether the response is on-topic, ignore concept presence, ignore whether instructions were followed. Score only the surface fluency.

Prompt:
{prompt}

Response:
{response}

Final line must be exactly: Rating: [[N]] where N is 0, 1, or 2.
```

## Instruction following

```
You are evaluating whether a model's response ADDRESSES the prompt — does it attempt the task the prompt asks for?

Score 0/1/2:
  0 = The response ignores the prompt, addresses an unrelated topic, or refuses without engaging.
  1 = The response partially addresses the prompt — touches on it but misses the main ask, drifts off-topic, or only loosely engages.
  2 = The response addresses the prompt directly and on-topic, attempting the task it asks for. (Quality of the answer is not scored here — only that it tries.)

Ignore fluency, ignore concept presence. Score only whether the response is ON-TASK and attempts to address the prompt.

Prompt:
{prompt}

Response:
{response}

Final line must be exactly: Rating: [[N]] where N is 0, 1, or 2.
```

## Concept relevance

```
You are evaluating whether a model's response engages with a specific concept in any visible form — as topic, as content, as emotion, as imagery, anywhere.

Concept: {concept}

Score 0/1/2:
  0 = The response does not engage with the concept at all. No mention, no allusion, no implicit reference.
  1 = The response engages with the concept partially — a passing mention, indirect allusion, or one-sentence aside.
  2 = The response clearly and substantively engages with the concept — it is a central or recurring feature of the response.

Ignore fluency, ignore whether the prompt was addressed. Score only the presence of the concept in the response. Feel free to reason on lines before your answer to help you score.

Prompt:
{prompt}

Response:
{response}

Final line must be exactly: Rating: [[N]] where N is 0, 1, or 2.
```

Rubrics used for the `steer` perspective (all vectors in the paper): `['fluency', 'instruction', 'concept_relevance']`
