"""ladders.py -- explicit ladder specifications for the temperature move analysis.

A ladder is a list of water words. Its moves are ALL pairs of words on it, oriented from the
colder to the warmer word by H.H2O_TEMP (the paper's water-word table). Pairs of words at the
same table temperature have no temperature move (the numeric difference vector is zero) and are
dropped. A word pair on two ladders is counted once.

"water" is the label the ladder files use for the puddle probe; it is kept here so the sorted
word order (and so the permutation draws) match the published pipeline exactly.

  current   the paper as drafted: 3 ladders, 14 words, 33 moves (reproduces the ladder files)
  proposed  dew, fog -> phase ; frost, hail -> precipitation (all 18 words)
  alt       'surface' (forms or lies in place) vs 'airborne' (falls or hangs in the air)
  allpairs  no ladders: every pair of the 18 words
"""
import sys
sys.path.insert(0, "./fractal_analysis")
import analysis_harness as H  # noqa

CANON = lambda w: "puddle" if w == "water" else w
T = lambda w: float(H.H2O_TEMP[CANON(w)])

ADJ = ["icy_water", "cold_water", "cool_water", "roomtemp_water", "warm_water", "hot_water", "boiling_water"]
SPECS = {
    "current": {"phase": ["ice", "slush", "water", "steam"],
                "precipitation": ["snow", "sleet", "rain", "steam"],
                "adjectival": ADJ},
    "proposed": {"phase": ["ice", "slush", "dew", "fog", "water", "steam"],
                 "precipitation": ["hail", "snow", "frost", "sleet", "rain", "steam"],
                 "adjectival": ADJ},
    "alt": {"surface": ["ice", "frost", "slush", "dew", "water", "steam"],
            "airborne": ["hail", "snow", "sleet", "rain", "fog", "steam"],
            "adjectival": ADJ},
    "allpairs": {"all": sorted(["ice", "snow", "hail", "frost", "sleet", "slush", "rain", "dew", "fog",
                                "water", "steam"] + ADJ, key=T)},
}


def edges(spec):
    """oriented cold->hot moves for a spec; also returns the dropped equal-temperature pairs."""
    seen, out, ties = set(), [], []
    for name, lad in SPECS[spec].items():
        temps = [T(w) for w in lad]
        assert temps == sorted(temps), f"{spec}/{name} is not listed in table-temperature order"
        for i in range(len(lad)):
            for j in range(i + 1, len(lad)):
                a, b = lad[i], lad[j]
                k = frozenset((a, b))
                if k in seen:
                    continue
                seen.add(k)
                if T(a) == T(b):
                    ties.append((name, a, b)); continue
                out.append(dict(src=a, tgt=b, sT=T(a), tT=T(b), ladder=name))
    return out, ties


if __name__ == "__main__":
    for s in SPECS:
        e, t = edges(s)
        words = sorted({x["src"] for x in e} | {x["tgt"] for x in e})
        print(f"{s:9s} {len(words):2d} words  {len(e):3d} moves  ties dropped: {[(a, b) for _, a, b in t]}")
        for name, lad in SPECS[s].items():
            print(f"   {name:13s} " + ", ".join(f"{w}({T(w):g})" for w in lad))
