# Temperature data-generation prompts (verbatim)

Extracted from `pipeline/generate/temperature_templates.py`, which the generators (`pipeline/generate/generate_temp_celsius.py`, `generate_temp_fahrenheit.py`, `generate_water_words.py`) import. All model calls go to `gemini-3.5-flash` with `thinking_budget=0`. Prompt pools: batches of `{n}`=24 at temperature 1.0; continuations: one call per prompt at temperature 0.7, `max_output_tokens=400`.

## Numeric temperatures (`temp_c1`, `temp_f1`)

### Prompt pool (generated once, shared by every °C and °F level)

```
Write {n} short, varied questions a user might ask an AI assistant to read off or state the current temperature shown on an instrument (a thermometer, gauge, sensor, probe, thermostat, or digital display) in a plausible everyday, kitchen, lab, or outdoor setting.
Each names the setting briefly but does NOT state the temperature value or hint whether it is hot or cold. One sentence each, self-contained (no files / images / links), nothing that presumes a previous conversation, history or task. Do NOT state any temperature yourself.
Return "1." ... "{n}." only.
```

### Positive continuation frames (templated; rotated across prompts; `{t}` = target temperature)

°C:

```
The thermometer reads about {t}°C.
It's showing roughly {t}°C right now.
The reading comes to {t}°C.
Right now it's sitting at {t}°C.
The display has settled at {t}°C.
```

°F (`temp_f1`, same frames with the unit swapped):

```
The thermometer reads about {t}°F.
It's showing roughly {t}°F right now.
The reading comes to {t}°F.
Right now it's sitting at {t}°F.
The display has settled at {t}°F.
```

### Negative continuation frames (templated; no figure)

```
The instrument isn't giving a steady reading right now — hard to call a number.
I can't make out a specific figure from the readout at the moment.
There's no clear value on the display just now.
```

## Water words (`h2o_v1`)

### Prompt pool

```
Write {n} short, varied requests a user might send an AI assistant asking it to describe or narrate a brief, concrete everyday scene or moment (a morning routine, a view from a window, a walk, a kitchen, a street, a garden), indoor or outdoor.
One sentence each, self-contained (no files / images / links), nothing that presumes a previous conversation, history or task, not a yes/no question. Do NOT mention water, weather, or temperature yourself.
Return "1." ... "{n}." only.
```

### Positive continuation (one call per member × prompt)

```
Write one short, natural response an AI assistant would give to the request below, describing the scene so that {desc} features prominently and naturally in it.
Weave it in concretely with a sensory detail or two (a few sentences); do not state any numeric temperature.

Request: {prompt}

Response only - no preamble.
```

### Negative continuation (one call per prompt, shared by all 18 members)

```
Write one short, natural response an AI assistant would give to the request below, describing the scene naturally.
Do NOT mention water in any form - no rain, snow, ice, frost, hail, sleet, slush, dew, fog, mist, steam, vapour, puddles, or drinking water - and do not state any temperature. A few sentences.

Request: {prompt}

Response only - no preamble.
```

### The 18 members and their `{desc}` fill

The nominal temperatures used in the analysis (paper Table 1) are `H2O_TEMP` in `relgeom/harness.py`; the generator never sees a temperature.

| slug | `{desc}` |
|---|---|
| `snow` | falling or fallen snow |
| `ice` | solid ice |
| `hail` | hail / hailstones |
| `frost` | frost on surfaces |
| `sleet` | sleet (rain mixed with ice) |
| `slush` | slush (partly melted snow) |
| `rain` | rain / rainfall |
| `dew` | dew on grass or leaves |
| `fog` | fog or mist |
| `puddle` | a puddle of liquid water |
| `steam` | steam / water vapour rising |
| `icy_water` | icy water |
| `cold_water` | cold water |
| `cool_water` | cool water |
| `roomtemp_water` | room-temperature water |
| `warm_water` | warm water |
| `hot_water` | hot water |
| `boiling_water` | boiling water |
