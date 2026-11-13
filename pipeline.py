"""Generation pipeline for Gemma 4 31B (vLLM offline).

One input-JSONL row = one steering vector to learn. The pipeline produces
N prompts per vector (default 72), each with a positive + negative continuation
pair, suitable for DiffMean-style vector estimation downstream.

Perspectives (consistent convention: pos = side the steering vector should push
toward, so v = mean(pos_acts) - mean(neg_acts) is the desired direction):

  steer        prompt where concept would NOT naturally appear.
               pos = with-concept,    neg = neutral baseline
               (1 prompt -> 2 contrastive continuations)
  anti_steer   prompt where concept WOULD naturally appear.
               pos = concept-scrubbed, neg = with-concept (natural)
               (1 prompt -> 2 contrastive continuations)
  magnitude    prompt magnitude-agnostic, biased to source magnitude.
               pos = target-magnitude, neg = source-magnitude
               (1 prompt -> 2 contrastive continuations)
  ff_contrast  paired prompts: pos prompt naturally evokes the concept;
               neg prompt evokes a *different* concept of the same form.
               Each continued naturally (no concept-specific instruction).
               (2 prompts -> 1 continuation each)

Input row schema:
  { vector_id, concept_id, concept, perspective, domain,
    functional_form_id?, functional_form?,
    source_magnitude?, target_magnitude?,   # for magnitude perspective
    ff_pos_instruction?, ff_neg_instruction?,  # for ff_contrast perspective
    n_prompts? }

Output:
  out_dir/prompts.jsonl   one row per (vector, prompt)
  out_dir/outputs.jsonl   one row per (vector, prompt, polarity, completion_idx)
  out_dir/meta.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
import uuid
from pathlib import Path

GEN_MODEL = "google/gemma-4-31B-it"

# Chunking: split the N prompts of one vector across CHUNKS LLM calls so each
# chunk gets its own inspiration word -> variety without per-prompt overhead.
CHUNKS = 6

# ============================================================
# Inspiration words (stand-in for WordNet sample; easy to swap for nltk later)
# ============================================================

INSPIRATIONS = [
    "anchor", "lantern", "compass", "thread", "harbor", "echo", "ribbon",
    "kettle", "willow", "marble", "ember", "satchel", "ladder", "quilt",
    "sparrow", "pebble", "cobalt", "fennel", "linen", "amber", "drift",
    "hollow", "meadow", "thistle", "saffron", "pewter", "quiver", "trellis",
    "almond", "cedar", "lichen", "garnet", "grove", "mariner", "verdant",
    "burrow", "sienna", "sapphire", "iris", "wisp", "loom", "bramble",
    "frost", "tide", "kindling", "marquee", "orchard", "plume", "quartz",
    "ridge", "slumber", "tinder", "umber", "vellum", "willowherb", "zephyr",
    "ash", "bark", "calico", "dune", "ember", "fjord", "glade", "haze",
    "ink", "juniper", "kiln", "loft", "mire", "nettle", "oak", "pine",
    "reed", "stone", "thorn", "vine", "wax", "yarrow", "azure", "bronze",
    "coal", "dusk", "elm", "feather", "gold", "heron", "ivory", "jade",
    "kestrel", "lark", "moss", "nimbus", "obsidian", "petal", "quill",
    "raven", "silver", "tarn", "umber", "vesper", "wren", "yew", "zest",
]

def sample_inspirations(seed: int, n: int) -> list[str]:
    rng = random.Random(seed)
    return rng.sample(INSPIRATIONS, n) if n <= len(INSPIRATIONS) else rng.choices(INSPIRATIONS, k=n)

# ============================================================
# Prompt templates
# ============================================================

_FF = "\n- match the functional form: {ff}"

# Shared prompt-quality guidance. Inserted as {guidance} in every T_*_PROMPTS
# template. Broad capability constraints, not a banned-word list — Gemma-4
# treats them as positive instructions to write self-contained, varied prompts.
_GUIDANCE_BULLETS = (
    "- be a complete, self-contained request answerable using only the text of the prompt itself\n"
    "- not refer to attachments, images, files, links, documents, prior messages, or anything not present in the prompt\n"
    "- end with terminal punctuation"
)

# Inserted as {diversity} in every T_*_PROMPTS template.
_DIVERSITY_LINE = (
    "Across the {n} prompts: vary the opening phrasing and the specific subject matter — "
    "do not reuse the same opener template or near-identical subject."
)

T_STEER_PROMPTS = """Generate {n} short, distinct prompts for a language model.

Each prompt MUST:
- be naturally relevant to the context: {context}{ff_clause}
- be the kind of prompt where a typical language-model continuation would NOT mention or evoke the concept "{concept}" (the concept is *not* the natural answer)
- be a real-world request someone might actually make in this context, NOT a question about the concept itself
- not be phrased as a yes/no question
{guidance}

Loose theme word (use as inspiration, do not force into the prompt): {inspiration}

{diversity}

Return only the {n} prompts, one per line, prefixed "1. ", "2. ", ..., "{n}. ". No commentary."""

T_STEER_POS = """Continue the prompt below with a fluent, on-topic response that naturally weaves in the concept "{concept}". The concept must clearly appear (or be unmistakably evoked) in the response.

Prompt: {prompt}

Return only the response. No preamble, no quotes."""

T_STEER_NEG = """Continue the prompt below with a fluent, on-topic response. Do NOT mention or evoke the concept "{concept}".

Prompt: {prompt}

Return only the response. No preamble, no quotes."""

T_ANTISTEER_PROMPTS = """Generate {n} short, distinct prompts for a language model.

Each prompt MUST:
- be naturally relevant to the context: {context}{ff_clause}
- be the kind of prompt where a typical language-model continuation WOULD naturally include or rely on the concept "{concept}"
- be a real-world request someone might actually make in this context
- not be phrased as a yes/no question, and not ask explicitly *about* the concept
{guidance}

Loose theme word (use as inspiration, do not force into the prompt): {inspiration}

{diversity}

Return only the {n} prompts, one per line, prefixed "1. ", "2. ", ..., "{n}. ". No commentary."""

T_ANTISTEER_POS = """Continue the prompt below with a fluent, on-topic response that addresses the prompt directly but does NOT mention or evoke the concept "{concept}", even though a typical answer would. Do not sound evasive.

Prompt: {prompt}

Return only the response. No preamble, no quotes."""

T_ANTISTEER_NEG = """Continue the prompt below with a fluent, on-topic response. The concept "{concept}" should appear naturally, as it would in a typical answer.

Prompt: {prompt}

Return only the response. No preamble, no quotes."""

T_MAG_PROMPTS = """Generate {n} short, distinct prompts for a language model.

Each prompt MUST:
- be naturally relevant to the context: {context}{ff_clause}
- be the kind of prompt where the natural continuation would involve "{concept}" at this magnitude / level / variant:
    {source_id}: {source_desc}
- NOT specify or commit to a magnitude in the prompt itself (the magnitude should emerge in the continuation, not be named in the prompt)
- be a real-world request, not a question about the concept
{guidance}

Loose theme word (use as inspiration, do not force into the prompt): {inspiration}

{diversity}

Return only the {n} prompts, one per line, prefixed "1. ", "2. ", ..., "{n}. ". No commentary."""

T_MAG_POS = """Continue the prompt below with a fluent, on-topic response. The concept "{concept}" should appear in the response at this magnitude / level / variant:
    {target_id}: {target_desc}

Prompt: {prompt}

Return only the response. No preamble, no quotes."""

T_MAG_NEG = """Continue the prompt below with a fluent, on-topic response. The concept "{concept}" should appear in the response at this magnitude / level / variant:
    {source_id}: {source_desc}

Prompt: {prompt}

Return only the response. No preamble, no quotes."""

# Days-of-week-only magnitude continuation templates. The default T_MAG_POS/NEG
# emit a literal "{id}: {desc}" tag (e.g. "tuesday: the day Tuesday") which
# Gemma echoes verbatim ~62% of the time on calendar_formal/scheduling FFs,
# because the day-name id and its description are redundant. These templates
# drop the id/colon construct and reference the day in plain English.
T_MAG_POS_DAYS = """Continue the prompt below with a fluent, on-topic response. The response should naturally reference {target_desc} as the specific day in question, without naming any other day of the week.

Prompt: {prompt}

Return only the response. No preamble, no quotes."""

T_MAG_NEG_DAYS = """Continue the prompt below with a fluent, on-topic response. The response should naturally reference {source_desc} as the specific day in question, without naming any other day of the week.

Prompt: {prompt}

Return only the response. No preamble, no quotes."""

T_FF_PROMPTS = """Generate {n} short, distinct prompts for a language model.

{instruction}

Each prompt MUST:
- read fluently as a real-world request, statement, or short question
- not be phrased as a yes/no question
- if relevant, fit loosely into the context: {context}
{guidance}

Loose theme word (use as inspiration, do not force into the prompt): {inspiration}

{diversity}

Return only the {n} prompts, one per line, prefixed "1. ", "2. ", ..., "{n}. ". No commentary."""

T_FF_CONT = """Continue the prompt below with a natural, fluent response. Address the prompt directly.

Prompt: {prompt}

Return only the response. No preamble, no quotes."""

# Reference vs vocab split perspectives (steer_ref, anti_steer_ref, steer_vocab,
# anti_steer_vocab). Generic templates; the concept-specific instruction strings
# live on each vector spec (prompt_instruction / pos_cont_instruction /
# neg_cont_instruction) and are authored in inputs/ref_vocab_instructions.py.
T_SPLIT_PROMPTS = """Generate {n} short, distinct prompts for a language model.

{prompt_instruction}

Each prompt MUST:
- be naturally relevant to the context: {context}
- be a real-world request someone might actually make in this context
- not be phrased as a yes/no question
{guidance}

Loose theme word (use as inspiration, do not force into the prompt): {inspiration}

{diversity}

Return only the {n} prompts, one per line, prefixed "1. ", "2. ", ..., "{n}. ". No commentary."""

T_SPLIT_POS = """{pos_cont_instruction}

Prompt: {prompt}

Return only the response. No preamble, no quotes."""

T_SPLIT_NEG = """{neg_cont_instruction}

Prompt: {prompt}

Return only the response. No preamble, no quotes."""

_SPLIT_PERSPECTIVES = ("steer_ref", "anti_steer_ref", "steer_vocab", "anti_steer_vocab")

# ============================================================
# Parsing
# ============================================================

_NUMBERED = re.compile(r"^\s*\d+[\.\)]\s*(.+?)\s*$")

def parse_numbered(text: str, n: int) -> list[str]:
    out = []
    for line in text.splitlines():
        m = _NUMBERED.match(line)
        if m:
            out.append(m.group(1))
    return out[:n]

# ============================================================
# JSONL I/O
# ============================================================

def jsonl_read(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def jsonl_append(fh, row):
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    fh.flush()

# ============================================================
# vLLM wrapper
# ============================================================

def make_llm(*, max_model_len=8192, mem_util=0.90, tp=1):
    from vllm import LLM
    return LLM(
        model=GEN_MODEL,
        max_model_len=max_model_len,
        gpu_memory_utilization=mem_util,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 0, "audio": 0},
    )

def chat(llm, prompts: list[str], *, temperature, top_p=0.95, max_tokens, n=1, seed=0):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n, seed=seed)
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    outs = llm.chat(msgs, sampling_params=sp, use_tqdm=False)
    return [[c.text for c in o.outputs] for o in outs]

# ============================================================
# Per-perspective rendering
# ============================================================

def _ff_clause(vec):
    ff = vec.get("functional_form")
    return _FF.format(ff=ff) if ff else ""

def render_prompt_gen(vec, n_chunk, inspiration, polarity="shared"):
    """Build a prompt-gen instruction for a chunk of `n_chunk` prompts.

    `polarity` is "shared" for steer/anti_steer/magnitude (same prompt set drives
    both pos and neg continuations), or "positive"/"negative" for ff_contrast
    (where pos and neg are entirely different prompt sets).
    """
    p = vec["perspective"]
    common = dict(
        concept=vec["concept"], context=vec["domain"],
        ff_clause=_ff_clause(vec), n=n_chunk, inspiration=inspiration,
        guidance=_GUIDANCE_BULLETS, diversity=_DIVERSITY_LINE.format(n=n_chunk),
    )
    if p == "steer":
        return T_STEER_PROMPTS.format(**common)
    if p == "anti_steer":
        return T_ANTISTEER_PROMPTS.format(**common)
    if p == "magnitude":
        return T_MAG_PROMPTS.format(
            **common,
            source_id=vec["source_magnitude"]["id"],
            source_desc=vec["source_magnitude"]["description"],
        )
    if p == "ff_contrast":
        instr = vec["ff_pos_instruction"] if polarity == "positive" else vec["ff_neg_instruction"]
        return T_FF_PROMPTS.format(
            n=n_chunk, instruction=instr,
            context=vec["domain"], inspiration=inspiration,
            guidance=_GUIDANCE_BULLETS, diversity=_DIVERSITY_LINE.format(n=n_chunk),
        )
    if p in _SPLIT_PERSPECTIVES:
        return T_SPLIT_PROMPTS.format(
            n=n_chunk, inspiration=inspiration, context=vec["domain"],
            prompt_instruction=vec["prompt_instruction"],
            guidance=_GUIDANCE_BULLETS, diversity=_DIVERSITY_LINE.format(n=n_chunk),
        )
    raise ValueError(f"unknown perspective: {p}")

def render_output(vec, prompt, polarity):
    """polarity: 'positive' or 'negative'."""
    p = vec["perspective"]
    c = vec["concept"]
    if p == "steer":
        tpl = T_STEER_POS if polarity == "positive" else T_STEER_NEG
        return tpl.format(concept=c, prompt=prompt)
    if p == "anti_steer":
        tpl = T_ANTISTEER_POS if polarity == "positive" else T_ANTISTEER_NEG
        return tpl.format(concept=c, prompt=prompt)
    if p == "magnitude":
        src, tgt = vec["source_magnitude"], vec["target_magnitude"]
        if vec.get("concept_id") == "days_of_week":
            if polarity == "positive":
                return T_MAG_POS_DAYS.format(target_desc=tgt["description"], prompt=prompt)
            return T_MAG_NEG_DAYS.format(source_desc=src["description"], prompt=prompt)
        if polarity == "positive":
            return T_MAG_POS.format(concept=c, prompt=prompt,
                                    target_id=tgt["id"], target_desc=tgt["description"])
        return T_MAG_NEG.format(concept=c, prompt=prompt,
                                source_id=src["id"], source_desc=src["description"])
    if p == "ff_contrast":
        return T_FF_CONT.format(prompt=prompt)
    if p in _SPLIT_PERSPECTIVES:
        if polarity == "positive":
            return T_SPLIT_POS.format(pos_cont_instruction=vec["pos_cont_instruction"], prompt=prompt)
        return T_SPLIT_NEG.format(neg_cont_instruction=vec["neg_cont_instruction"], prompt=prompt)
    raise ValueError(f"unknown perspective: {p}")

# ============================================================
# Prompt quality filters
#
# Hard filters DROP a prompt (and trigger re-roll if we're under target).
# Soft filters FLAG a prompt in the audit log but KEEP it — some flagged shapes
# (e.g. no terminal punctuation) can be legitimate high-signal long imperatives
# that got cut by the gen cap.
# ============================================================

_HARD_BOGUS_PATTERNS = [
    ("attachment_ref", re.compile(r"\b(attached|attachment|the file|the document|the link|uploaded|provided document|the doc)\b", re.IGNORECASE)),
    ("image_ref",      re.compile(r"\b(the image|the picture|the photo|the screenshot|the chart|the graph|the diagram|the video|in the image|in the picture)\b", re.IGNORECASE)),
    ("backref",        re.compile(r"\b(as i mentioned|as i said|as discussed|previously discussed|in your last|in your previous|as we discussed|the previous response)\b", re.IGNORECASE)),
    ("pronoun_orphan", re.compile(r"^(That|This|It|These|Those|They|He|She)\s+(is|are|was|were|will|would|seems|seem|looks|look)\b")),
    ("too_short",      re.compile(r"^.{0,25}$")),
]

_SOFT_FLAG_PATTERNS = [
    ("no_terminal_punct", re.compile(r"[^.!?\"')\]]\s*$")),
]


def _classify_prompt(prompt: str) -> tuple[str, str]:
    """Return ('hard', reason) | ('soft', reason) | ('ok', '')."""
    for name, pat in _HARD_BOGUS_PATTERNS:
        if pat.search(prompt):
            return ("hard", name)
    for name, pat in _SOFT_FLAG_PATTERNS:
        if pat.search(prompt):
            return ("soft", name)
    return ("ok", "")


def _trigrams(s: str) -> set[str]:
    """Lowercased character-trigram set, non-alphanumerics collapsed to single
    spaces. Captures slot-fill clones like "I am struggling ... low-light forest"
    vs "I am struggling ... low-light dusk" without being confused by case or
    incidental punctuation."""
    norm = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    return {norm[i:i+3] for i in range(max(0, len(norm) - 2))}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def _filter_batch(candidates, *, target, threshold, accepted_grams):
    """Filter one batch of candidates and return only NEW kept items.

    `accepted_grams` is the running list of trigram sets from previously kept
    prompts (across earlier batches + any pre-seed e.g. training prompts when
    we're generating heldout).
    """
    new_kept, new_grams = [], []
    audit = {"hard_killed": [], "near_dup_killed": [], "soft_flagged": []}
    for p in candidates:
        if len(new_kept) >= target:
            break
        cls, reason = _classify_prompt(p)
        if cls == "hard":
            audit["hard_killed"].append((reason, p))
            continue
        if cls == "soft":
            audit["soft_flagged"].append((reason, p))
        g = _trigrams(p)
        worst = max(
            (_jaccard(g, kg) for kg in (accepted_grams + new_grams)),
            default=0.0,
        )
        if worst >= threshold:
            audit["near_dup_killed"].append((round(worst, 3), p))
            continue
        new_kept.append(p)
        new_grams.append(g)
    return new_kept, new_grams, audit


def _audit_summary(audit, n_attempts):
    """Compact summary suitable for jsonl logging."""
    return {
        "n_attempts": n_attempts,
        "n_hard_killed": len(audit["hard_killed"]),
        "n_near_dup_killed": len(audit["near_dup_killed"]),
        "n_soft_flagged": len(audit["soft_flagged"]),
        "hard_examples":     [(r, p[:120]) for r, p in audit["hard_killed"][:5]],
        "near_dup_examples": [(s, p[:120]) for s, p in audit["near_dup_killed"][:5]],
        "soft_examples":     [(r, p[:120]) for r, p in audit["soft_flagged"][:5]],
    }


# ============================================================
# Generation
# ============================================================

def _vec_meta(vec):
    """The fields propagated onto every output row for this vector."""
    keys = ["vector_id", "concept_id", "concept", "perspective", "domain",
            "functional_form_id", "functional_form",
            "source_magnitude", "target_magnitude"]
    return {k: vec[k] for k in keys if k in vec}

def _gen_prompts(llm, vec, n_prompts, polarity, seed, *, salt="train_0"):
    """Generate `n_prompts` prompts for the given polarity, chunked across CHUNKS calls.

    `salt` differentiates re-roll attempts so each produces distinct inspirations
    + LLM samples. Default 'train_0' preserves first-attempt behavior; subsequent
    attempts pass 'train_1', 'train_2', ... Heldout passes 'heldout_<attempt>'.
    """
    chunk_sizes = [n_prompts // CHUNKS] * CHUNKS
    for i in range(n_prompts % CHUNKS):
        chunk_sizes[i] += 1
    insp_seed = hash((vec["vector_id"], polarity, salt)) & 0xFFFFFFFF
    inspirations = sample_inspirations(seed=insp_seed, n=CHUNKS)

    inputs = [render_prompt_gen(vec, k, ins, polarity)
              for k, ins in zip(chunk_sizes, inspirations) if k > 0]
    llm_seed = (seed ^ insp_seed) & 0xFFFFFFFF
    chunks = chat(llm, inputs, temperature=1.0, max_tokens=2048, n=1, seed=llm_seed)

    prompts: list[str] = []
    for raw, want in zip(chunks, chunk_sizes):
        prompts.extend(parse_numbered(raw[0], want))
    return prompts


def _generate_with_filter(gen_fn, *, n_target, threshold, max_attempts, pre_seed_grams=()):
    """Run gen_fn(ask, attempt) -> list[str] in a re-roll loop. Each pass over-
    generates ~50%, runs hard-bogus + near-dup filtering, accumulates kept
    prompts and a running gram set. pre_seed_grams lets the heldout path
    inherit the training-set gram pool so heldout dedups against training too.

    Returns (kept, kept_grams, audit_summary_dict).
    """
    kept, kept_grams = [], list(pre_seed_grams)
    agg = {"hard_killed": [], "near_dup_killed": [], "soft_flagged": []}
    n_attempts_used = 0
    for attempt in range(max_attempts):
        n_attempts_used = attempt + 1
        remaining = n_target - len(kept)
        if remaining <= 0:
            break
        ask = remaining + max(4, remaining // 2)
        candidates = gen_fn(ask, attempt)
        new_kept, new_grams, audit = _filter_batch(
            candidates, target=remaining, threshold=threshold,
            accepted_grams=kept_grams,
        )
        kept.extend(new_kept)
        kept_grams.extend(new_grams)
        for k in agg:
            agg[k].extend(audit[k])
    return kept, kept_grams, _audit_summary(agg, n_attempts_used)


def run_vector(llm, vec, n_prompts, n_completions, prompts_fh, outputs_fh, audit_fh,
               seed, *, dedup_threshold, max_attempts):
    is_contrast = vec["perspective"] == "ff_contrast"

    base = _vec_meta(vec)
    audit_rows = {}

    def _gen_call(polarity, ask, attempt):
        return _gen_prompts(llm, vec, ask, polarity, seed, salt=f"train_{attempt}")

    # 1. Generate prompts. ff_contrast needs two disjoint sets; others share.
    if is_contrast:
        pos, _, pos_audit = _generate_with_filter(
            lambda ask, a: _gen_call("positive", ask, a),
            n_target=n_prompts, threshold=dedup_threshold, max_attempts=max_attempts)
        neg, _, neg_audit = _generate_with_filter(
            lambda ask, a: _gen_call("negative", ask, a),
            n_target=n_prompts, threshold=dedup_threshold, max_attempts=max_attempts)
        prompts_by_pol = {"positive": pos, "negative": neg}
        audit_rows["positive"] = pos_audit
        audit_rows["negative"] = neg_audit
        n_min = min(len(pos), len(neg))
    else:
        shared, _, shared_audit = _generate_with_filter(
            lambda ask, a: _gen_call("shared", ask, a),
            n_target=n_prompts, threshold=dedup_threshold, max_attempts=max_attempts)
        prompts_by_pol = {"positive": shared, "negative": shared}
        audit_rows["shared"] = shared_audit
        n_min = len(shared)

    # Persist audit row regardless of error: even partial runs benefit from logs.
    if audit_fh is not None:
        jsonl_append(audit_fh, {
            **base, "n_target": n_prompts, "n_kept": n_min, "by_polarity": audit_rows,
        })

    if n_min < n_prompts // 2:
        raise RuntimeError(
            f"[{vec['vector_id']}] only parsed {n_min}/{n_prompts} prompts (perspective={vec['perspective']})")

    if is_contrast:
        for polarity, prs in prompts_by_pol.items():
            for i, p in enumerate(prs):
                jsonl_append(prompts_fh, {**base, "prompt_idx": i, "polarity": polarity, "prompt": p})
    else:
        for i, p in enumerate(prompts_by_pol["positive"]):
            jsonl_append(prompts_fh, {**base, "prompt_idx": i, "prompt": p})

    # 2. Continuations.
    requests = []
    for polarity in ("positive", "negative"):
        for i, p in enumerate(prompts_by_pol[polarity]):
            requests.append((i, polarity, render_output(vec, p, polarity)))

    completions = chat(llm, [r[2] for r in requests],
                       temperature=0.7, max_tokens=512, n=n_completions, seed=seed + 1)

    for (p_idx, polarity, _), comps in zip(requests, completions):
        prompt_text = prompts_by_pol[polarity][p_idx]
        for c_idx, text in enumerate(comps):
            jsonl_append(outputs_fh, {
                **base,
                "prompt_idx": p_idx, "polarity": polarity, "completion_idx": c_idx,
                "prompt": prompt_text, "completion": text,
                "record_id": str(uuid.uuid4()),
            })

    return n_min

def run_generate(llm, input_path, out_dir: Path, n_prompts, n_completions, seed,
                 *, dedup_threshold, max_attempts):
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors = list(jsonl_read(input_path))
    print(f"[gen] {len(vectors)} vectors -> {out_dir} "
          f"(dedup_threshold={dedup_threshold}, max_attempts={max_attempts})", flush=True)

    started = time.time()
    counts = []
    with open(out_dir / "prompts.jsonl", "w") as pf, \
         open(out_dir / "outputs.jsonl", "w") as of, \
         open(out_dir / "prompt_gen_audit.jsonl", "w") as af:
        for i, vec in enumerate(vectors):
            n = vec.get("n_prompts", n_prompts)
            t0 = time.time()
            n_parsed = run_vector(
                llm, vec, n, n_completions, pf, of, af, seed,
                dedup_threshold=dedup_threshold, max_attempts=max_attempts,
            )
            counts.append({"vector_id": vec["vector_id"], "n_prompts_parsed": n_parsed})
            print(f"  [{i+1}/{len(vectors)}] {vec['vector_id']} "
                  f"prompts={n_parsed}/{n} ({time.time()-t0:.1f}s)", flush=True)

    (out_dir / "meta.json").write_text(json.dumps({
        "model": GEN_MODEL,
        "n_vectors": len(vectors),
        "n_prompts_default": n_prompts,
        "n_completions": n_completions,
        "seed": seed,
        "dedup_threshold": dedup_threshold,
        "max_attempts": max_attempts,
        "elapsed_sec": round(time.time() - started, 1),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vector_counts": counts,
    }, indent=2))

# ============================================================
# Held-out prompt generation
#
# Held-out prompts use the SAME prompt-generation templates as training,
# differing only in RNG seed so the generated inspirations + LLM samples
# don't collide with training. After generation we explicitly dedupe each
# vector's held-out prompts against its training prompts. If a collision
# survives, one extra re-roll is done; any remaining deficit is accepted
# and logged.
# ============================================================

def run_vector_heldout(llm, vec, n_heldout, prompts_fh, audit_fh, base_seed,
                       training_prompt_set, *, dedup_threshold, max_attempts):
    """Generate held-out prompts for one vector. Hard-bogus filter + near-dup
    filter + re-roll. Dedup against training is enforced by pre-seeding the
    gram pool with trigram sets from `training_prompt_set`. Returns
    (n_written, audit_summary).
    """
    if vec.get("perspective") == "ff_contrast":
        print(f"  [SKIP heldout] {vec['vector_id']} — ff_contrast not supported in heldout path", flush=True)
        return 0, {}

    training_grams = [_trigrams(p) for p in training_prompt_set]

    def _gen_call(ask, attempt):
        # Reuse _gen_prompts with a heldout-specific salt so inspirations diverge
        # from any training-time generation.
        return _gen_prompts(llm, vec, ask, "shared", base_seed, salt=f"heldout_{attempt}")

    kept, _, audit = _generate_with_filter(
        _gen_call,
        n_target=n_heldout, threshold=dedup_threshold,
        max_attempts=max_attempts, pre_seed_grams=training_grams,
    )

    base = _vec_meta(vec)
    for i, p in enumerate(kept):
        jsonl_append(prompts_fh, {**base, "heldout_prompt_idx": i, "prompt": p})
    if audit_fh is not None:
        jsonl_append(audit_fh, {
            **base, "n_target": n_heldout, "n_kept": len(kept),
            "n_training_pool": len(training_prompt_set),
            **audit,
        })
    return len(kept), audit


def run_heldout(llm, input_path, raw_dir: Path, n_heldout, base_seed,
                *, dedup_threshold, max_attempts):
    """Generate held-out prompts. Writes raw_dir/heldout_prompts.jsonl,
    raw_dir/heldout_meta.json, and raw_dir/heldout_audit.jsonl. Reads
    raw_dir/prompts.jsonl for training-set dedup if present."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    training_by_vec: dict[str, set[str]] = {}
    train_path = raw_dir / "prompts.jsonl"
    if train_path.exists():
        for row in jsonl_read(train_path):
            training_by_vec.setdefault(row["vector_id"], set()).add(row["prompt"])
        n_train = sum(len(s) for s in training_by_vec.values())
        print(f"[heldout] loaded {n_train} training prompts across {len(training_by_vec)} vectors", flush=True)
    else:
        print(f"[heldout] WARNING: no training prompts.jsonl at {train_path} — training dedup disabled", flush=True)

    vectors = list(jsonl_read(input_path))
    print(f"[heldout] {len(vectors)} vectors -> {raw_dir} "
          f"(dedup_threshold={dedup_threshold}, max_attempts={max_attempts})", flush=True)

    started = time.time()
    counts = []
    out_path = raw_dir / "heldout_prompts.jsonl"
    audit_path = raw_dir / "heldout_audit.jsonl"
    with open(out_path, "w") as pf, open(audit_path, "w") as af:
        for i, vec in enumerate(vectors):
            training_set = training_by_vec.get(vec["vector_id"], set())
            t0 = time.time()
            n_written, audit = run_vector_heldout(
                llm, vec, n_heldout, pf, af, base_seed, training_set,
                dedup_threshold=dedup_threshold, max_attempts=max_attempts,
            )
            counts.append({
                "vector_id": vec["vector_id"],
                "n_heldout_parsed": n_written,
                "n_attempts": audit.get("n_attempts", 0),
                "n_hard_killed": audit.get("n_hard_killed", 0),
                "n_near_dup_killed": audit.get("n_near_dup_killed", 0),
                "n_soft_flagged": audit.get("n_soft_flagged", 0),
            })
            print(f"  [{i+1}/{len(vectors)}] {vec['vector_id']} "
                  f"heldout={n_written}/{n_heldout} "
                  f"attempts={audit.get('n_attempts','?')} "
                  f"(hard={audit.get('n_hard_killed','?')}, "
                  f"near_dup={audit.get('n_near_dup_killed','?')}, "
                  f"soft={audit.get('n_soft_flagged','?')}, "
                  f"{time.time()-t0:.1f}s)", flush=True)

    (raw_dir / "heldout_meta.json").write_text(json.dumps({
        "model": GEN_MODEL,
        "n_vectors": len(vectors),
        "n_heldout": n_heldout,
        "seed": base_seed,
        "salt": "heldout",
        "training_prompts_path": str(train_path),
        "dedup_threshold": dedup_threshold,
        "max_attempts": max_attempts,
        "elapsed_sec": round(time.time() - started, 1),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vector_counts": counts,
    }, indent=2))

# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--input", required=True, help="Vectors JSONL")
    g.add_argument("--out-dir", required=True)
    g.add_argument("--n-prompts", type=int, default=72)
    g.add_argument("--n-completions", type=int, default=1)
    g.add_argument("--max-model-len", type=int, default=8192)
    g.add_argument("--mem-util", type=float, default=0.90)
    g.add_argument("--tp", type=int, default=1)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--dedup-threshold", type=float, default=0.8,
                   help="Char-3gram Jaccard threshold above which two prompts are treated as duplicates. "
                        "Training default 0.8 (lenient — slot-fill variants are signal).")
    g.add_argument("--max-attempts", type=int, default=4,
                   help="Max re-roll attempts when under target after filtering.")

    h = sub.add_parser("heldout", help="Generate held-out prompts only (no completions)")
    h.add_argument("--input", required=True, help="Vectors JSONL")
    h.add_argument("--raw-dir", required=True,
                   help="Same dir as 'generate' --out-dir; reads prompts.jsonl for dedup, writes heldout_prompts.jsonl")
    h.add_argument("--n-heldout", type=int, default=12)
    h.add_argument("--max-model-len", type=int, default=8192)
    h.add_argument("--mem-util", type=float, default=0.90)
    h.add_argument("--tp", type=int, default=1)
    h.add_argument("--seed", type=int, default=42)
    h.add_argument("--dedup-threshold", type=float, default=0.6,
                   help="Heldout default 0.6 (stricter — want maximum prompt diversity for eval).")
    h.add_argument("--max-attempts", type=int, default=4)

    args = ap.parse_args()
    llm = make_llm(max_model_len=args.max_model_len, mem_util=args.mem_util, tp=args.tp)

    if args.cmd == "generate":
        run_generate(llm, args.input, Path(args.out_dir),
                     args.n_prompts, args.n_completions, args.seed,
                     dedup_threshold=args.dedup_threshold, max_attempts=args.max_attempts)
    elif args.cmd == "heldout":
        run_heldout(llm, args.input, Path(args.raw_dir),
                    args.n_heldout, args.seed,
                    dedup_threshold=args.dedup_threshold, max_attempts=args.max_attempts)

if __name__ == "__main__":
    main()
