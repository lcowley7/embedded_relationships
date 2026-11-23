"""Prompt templates for the analogy datasets (size_imp_v1, war_arg_v1, food_ideas_v1, frames_v1).

The per-concept text (items, glosses, sense locks, pools, scrubs, ban lists) is in
relgeom/analogies.py; this file holds the three templates and the prompt-pool filter.
"""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_utils as P
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from relgeom import analogies as W


# Prompt pool: one per experiment, shared by both clouds (relgeom.analogies.POOLS).
PROMPTGEN = W.POOLS

# Positive: the concept features in a continuation of the request.
#   {lock}   sense lock: literal reading only (the blended figurative uses are held out)
#   {scale}  for the ordered size/importance ladders, the rung's two neighbours
#   {term}   ask for the concept's own word, used literally, never as a simile
#   {kind}   soft co-mention line: avoid the cloud's other concepts (measured, not banned)
#   {avoid}  the OTHER cloud's vocabulary is off limits, so the clouds share no surface strings
# The length cap and "plain prose, no markdown" are identical on the negative side, so
# neither differs systematically between positive and negative.
CONT = """Write one short, natural response an AI assistant would give to the request below, clearly weaving {desc} into it so that it features naturally.
Here {desc} means: {gloss}. {lock}{scale}
Use the word "{term}" itself somewhere in the response rather than only implying it, and let it really be there in what is described - part of the scene or of the answer itself, never a comparison or a simile for something else (no "just like", "much like", "as if", "think of it as"), and never something you the assistant are doing or pausing to do.
Avoid mentioning other {kind} that are not {desc}, that includes both {kind} that are closely related to {desc}, and those that are very different.
{avoid}
Three or four sentences, no more than 70 words.

Request: {prompt}

Response only - plain prose, no preamble, no markdown."""

# Negative: the same request with the cloud's domain scrubbed; generated once per prompt and
# shared by every concept in the cloud.
CONT_ABSENT = """Write one short, natural response an AI assistant would give to the request below.
{scrub} Three or four sentences, no more than 70 words.

Request: {prompt}

Response only - plain prose, no preamble, no markdown."""


def render_promptgen(ds, n):
    return PROMPTGEN[ds].format(n=n)

def render_absent(cloud, prompt):
    return CONT_ABSENT.format(scrub=W.SCRUB[cloud], prompt=prompt)

def render_cont(rec, prompt):
    return CONT.format(desc=rec["desc"], gloss=rec["gloss"], lock=rec["lock"],
                             scale=W.scale_note(rec), term=W.head_term(rec),
                             kind=W.CLOUD_KIND[rec["cloud"]],
                             avoid=W.AVOID_CLAUSE[rec["avoid_key"]], prompt=prompt)

# Reject any pool prompt that carries either cloud's own domain words, since the pool is shared
# by every vector in the experiment.
_POOL_BAN = {ds: sorted({w for cl in cls for w in W.OWN_BAN[cl]})
             for ds, cls in W.EXPERIMENT_CLOUDS.items()}

def pool_screen(ds, prompt):
    """[] if the prompt is clean for this experiment, else the domain words it carries."""
    pat = re.compile(r"\b(" + "|".join(re.escape(w) for w in
                     sorted(_POOL_BAN[ds], key=len, reverse=True)) +
                     r")(s|es|ed|d|ing|ion|ly)?\b", re.I)
    return sorted({m.group(1).lower() for m in pat.finditer(prompt)})

def filter_prompts(cands_, n, th=0.8, ds=None):
    kept, grams = [], []
    for p in cands_:
        if len(kept) >= n: break
        if P._classify_prompt(p)[0] == "hard": continue
        if ds is not None and pool_screen(ds, p): continue
        g = P._trigrams(p)
        if max((P._jaccard(g, kg) for kg in grams), default=0) >= th: continue
        kept.append(p); grams.append(g)
    return kept

