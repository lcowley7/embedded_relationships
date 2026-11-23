"""Prompt templates and level grids for the temperature (temp_c1, temp_f1) and water-word (h2o_v1)
datasets."""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_utils as P     # parse_numbered / _classify_prompt / _trigrams / _jaccard


C1 = list(range(-20, 111, 1))         # -20,...,110      (131 levels)

def chunkify(lst, k):
    n = math.ceil(len(lst) / k)
    return [lst[i:i+n] for i in range(0, len(lst), n)]
C1_CHUNKS = chunkify(C1, 4)           # 4 contiguous chunks ~33 each -> separate SLURM datasets temp_c1a..d

def tslug(t): return (f"neg{abs(t)}" if t < 0 else str(t))   # filesafe vector_id token


# Instrument-reading questions: any value -20..110 is coherent (thermometer/gauge/probe can read anything),
# so every (prompt, t) pair is sensible and the prompts stay identical across all levels.
TEMP_PROMPTGEN = """Write {n} short, varied questions a user might ask an AI assistant to read off or state the current temperature shown on an instrument (a thermometer, gauge, sensor, probe, thermostat, or digital display) in a plausible everyday, kitchen, lab, or outdoor setting.
Each names the setting briefly but does NOT state the temperature value or hint whether it is hot or cold. One sentence each, self-contained (no files / images / links), nothing that presumes a previous conversation, history or task. Do NOT state any temperature yourself.
Return \"1.\" ... \"{n}.\" only."""

# answer frames that STATE a temperature — rotated across prompts for phrasing diversity. {t} is the only slot.
BODY_C = [
    "The thermometer reads about {t}°C.",
    "It's showing roughly {t}°C right now.",
    "The reading comes to {t}°C.",
    "Right now it's sitting at {t}°C.",
    "The display has settled at {t}°C.",
]
BODY_F = [s.replace("°C", "°F") for s in BODY_C]
# answer frames giving NO figure (steer-negative).
ABSENT = [
    "The instrument isn't giving a steady reading right now — hard to call a number.",
    "I can't make out a specific figure from the readout at the moment.",
    "There's no clear value on the display just now.",
]

def render_promptgen(n): return TEMP_PROMPTGEN.format(n=n)
def body_c(i, t):        return BODY_C[i % len(BODY_C)].format(t=t)
def body_f(i, t):        return BODY_F[i % len(BODY_F)].format(t=t)
def render_absent(i):    return ABSENT[i % len(ABSENT)]

def filter_prompts(cands, n, th=0.8):
    kept, grams = [], []
    for p in cands:
        if len(kept) >= n: break
        if P._classify_prompt(p)[0] == "hard": continue
        g = P._trigrams(p)
        if max((P._jaccard(g, kg) for kg in grams), default=0) >= th: continue
        kept.append(p); grams.append(g)
    return kept


H2O = [
    # frozen cluster (<0°C) — should clump on the temp axis but stay distinct in identity
    dict(slug="snow",   label="snow",                  desc="falling or fallen snow",         temp_c=-5,  group="frozen"),
    dict(slug="ice",    label="ice",                   desc="solid ice",                       temp_c=-5,  group="frozen"),
    dict(slug="hail",   label="hail",                  desc="hail / hailstones",               temp_c=-5,  group="frozen"),
    dict(slug="frost",  label="frost",                 desc="frost on surfaces",               temp_c=-3,  group="frozen"),
    dict(slug="sleet",  label="sleet",                 desc="sleet (rain mixed with ice)",     temp_c=0,   group="frozen"),
    dict(slug="slush",  label="slush",                 desc="slush (partly melted snow)",      temp_c=0,   group="frozen"),
    # liquid forms (mild)
    dict(slug="rain",   label="rain",                  desc="rain / rainfall",                 temp_c=12,  group="liquid"),
    dict(slug="dew",    label="dew",                   desc="dew on grass or leaves",          temp_c=12,  group="liquid"),
    dict(slug="fog",    label="fog",                   desc="fog or mist",                     temp_c=12,  group="liquid"),
    dict(slug="puddle", label="liquid water",          desc="a puddle of liquid water",        temp_c=15,  group="liquid"),
    # gas (hot)
    dict(slug="steam",  label="steam",                 desc="steam / water vapour rising",     temp_c=100, group="gas"),
    # adjective-qualified liquid water — a temperature sweep on a FIXED substance, in words (not numbers)
    dict(slug="icy_water",      label="icy water",              desc="icy water",              temp_c=1,   group="adj"),
    dict(slug="cold_water",     label="cold water",             desc="cold water",             temp_c=7,   group="adj"),
    dict(slug="cool_water",     label="cool water",             desc="cool water",             temp_c=15,  group="adj"),
    dict(slug="roomtemp_water", label="room-temperature water", desc="room-temperature water", temp_c=22,  group="adj"),
    dict(slug="warm_water",     label="warm water",             desc="warm water",             temp_c=40,  group="adj"),
    dict(slug="hot_water",      label="hot water",              desc="hot water",              temp_c=65,  group="adj"),
    dict(slug="boiling_water",  label="boiling water",          desc="boiling water",          temp_c=100, group="adj"),
]


# H2O is built like frequency STEER: pos and neg are two continuations of the SAME neutral scene-request.
# absent neg is generated once per prompt and shared across all 18 members (paired by prompt) -> every
# member vector = mean(pos_m) - mean(neg) over identical prompts; only the member differs. NO pooled batches.

H2O_PROMPTGEN = """Write {n} short, varied requests a user might send an AI assistant asking it to describe or narrate a brief, concrete everyday scene or moment (a morning routine, a view from a window, a walk, a kitchen, a street, a garden), indoor or outdoor.
One sentence each, self-contained (no files / images / links), nothing that presumes a previous conversation, history or task, not a yes/no question. Do NOT mention water, weather, or temperature yourself.
Return "1." ... "{n}." only."""

# pos: continue the SAME request so the member features prominently and naturally.
H2O_CONT = """Write one short, natural response an AI assistant would give to the request below, describing the scene so that {desc} features prominently and naturally in it.
Weave it in concretely with a sensory detail or two (a few sentences); do not state any numeric temperature.

Request: {prompt}

Response only - no preamble."""

# neg (absent, SHARED across members): continue the SAME request with no water in any form.
H2O_CONT_ABSENT = """Write one short, natural response an AI assistant would give to the request below, describing the scene naturally.
Do NOT mention water in any form - no rain, snow, ice, frost, hail, sleet, slush, dew, fog, mist, steam, vapour, puddles, or drinking water - and do not state any temperature. A few sentences.

Request: {prompt}

Response only - no preamble."""

def render_h2o_promptgen(n):       return H2O_PROMPTGEN.format(n=n)
def render_h2o_cont(desc, prompt): return H2O_CONT.format(desc=desc, prompt=prompt)
def render_h2o_absent(prompt):     return H2O_CONT_ABSENT.format(prompt=prompt)
# held-out steering-eval prompts reuse the neutral scene-request generator (distinct batch; open-ended).
