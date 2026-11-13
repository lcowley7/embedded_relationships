"""build_appendix_analogy_prompts.py -> latex/appendix_analogy_prompts.tex

The Appendix C subsection for the four analogy experiments, in the house format of the
water-word subsection (app:genprompts-h2o): short prose, then the prompt pool, the positive
continuation and the shared negative in footnotesize verbatim, hard-wrapped to 68 columns.

Every string is read from what the generator actually executed -- the template cell of
wave1_workshop.ipynb (generate_wave1.py exec's it by marker) and wave1_families.py -- so the
appendix cannot drift from the run. Self-check: one positive and one negative call are
re-rendered and compared byte-for-byte with prompts/01_rendered.md before anything is written.

Run: python paper/build_appendix_analogy_prompts.py
"""
import json, re, sys, textwrap
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent                   # extension/paper
EXT = HERE.parent
REPO = Path(".")
sys.path.insert(0, str(EXT)); import wave1_families as W
OUT = HERE / "latex" / "appendix_analogy_prompts.tex"
WIDTH = 68

# ---- the templates, from the cell the generator exec'd --------------------------------------
nb = json.loads((EXT / "wave1_workshop.ipynb").read_text())
tpl = next("".join(c["source"]) for c in nb["cells"]
           if c["cell_type"] == "code" and "WAVE1_PROMPTGEN =" in "".join(c["source"]))
def grab(name):
    i = tpl.index(f'{name} = """') + len(f'{name} = """')
    return tpl[i:tpl.index('"""', i)]
POS, NEG = grab("WAVE1_CONT"), grab("WAVE1_CONT_ABSENT")

def render_pos(rec, prompt):
    return POS.format(desc=rec["desc"], gloss=rec["gloss"], lock=rec["lock"],
                      scale=W.scale_note(rec), term=W.head_term(rec),
                      kind=W.CLOUD_KIND[rec["cloud"]],
                      avoid=W.AVOID_CLAUSE[rec["avoid_key"]], prompt=prompt)

# ---- self-check against the rendered calls extracted on 3 Sep --------------------------------
md = (HERE / "prompts/01_rendered.md").read_text()
sec = md.split("## `size_imp_v1`")[1].split("\n## ")[0]
prompt = re.search(r"Pooled prompt: \*(.+?)\*", sec).group(1)
fence = lambda s: s.split("```")[1].strip("\n")
want_pos = fence(sec.split("### positive, `size__microscopic`")[1])
want_neg = fence(sec.split("### negative, shared by the whole `size` cloud")[1])
rec = next(r for r in W.CLOUDS["size"] if r["id"] == "microscopic")
assert render_pos(rec, prompt).strip("\n") == want_pos, "positive template drifted"
assert NEG.format(scrub=W.SCRUB["size"], prompt=prompt).strip("\n") == want_neg, "negative drifted"
print("self-check: positive and negative re-render byte-identical to prompts/01_rendered.md")

# ---- formatting helpers ------------------------------------------------------------------------
def vwrap(text):
    out = []
    for line in text.split("\n"):
        out.extend(textwrap.wrap(line, WIDTH, break_long_words=False, break_on_hyphens=False)
                   if line.strip() else [""])
    return "\n".join(out)

def verbatim(body):
    for line in body.split("\n"):
        assert len(line) <= WIDTH or " " not in line.strip(), f"unwrapped line: {line!r}"
    return "\\begin{footnotesize}\n\\begin{verbatim}\n" + body + "\n\\end{verbatim}\n\\end{footnotesize}\n"

TEX_SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
               "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}
def tex(s):
    s = "".join(TEX_SPECIAL.get(ch, ch) for ch in s)
    assert '"' not in s, f"straight quote in table text: {s!r}"
    return s.replace("--", "-{}-")        # keep a literal double hyphen from becoming an en dash

DS = ["size_imp_v1", "war_arg_v1", "food_ideas_v1", "frames_v1"]
SIDE = {"size": "source", "importance": "target", "war": "source", "argument": "target",
        "food": "source", "ideas": "target"}
def cloud_label(cl):
    return f"{cl} ({SIDE[cl]} cloud)" if cl in SIDE else f"{cl} ({W.F1_FRAME_NAME[cl]})"

def slot_block(ds):
    lines = []
    for cl in W.EXPERIMENT_CLOUDS[ds]:
        recs = W.CLOUDS[cl]; r0 = recs[0]
        assert all(r["lock"] == r0["lock"] and r["avoid_key"] == r0["avoid_key"] for r in recs)
        if lines: lines.append("")
        lines.append(cloud_label(cl))
        for lab, txt in [("{kind}", W.CLOUD_KIND[cl]), ("{lock}", r0["lock"]),
                         ("{avoid}", W.AVOID_CLAUSE[r0["avoid_key"]]), ("{scrub}", W.SCRUB[cl])]:
            pre = f"  {lab:<8} "
            lines += textwrap.wrap(txt, WIDTH, initial_indent=pre, subsequent_indent=" " * len(pre),
                                   break_long_words=False, break_on_hyphens=False)
    return "\n".join(lines)

meta = {ds: json.loads((REPO / f"data/raw/{ds}/meta.json").read_text()) for ds in DS}
nvec = {ds: meta[ds]["n_vectors"] for ds in DS}
assert all(meta[ds]["n_prompts_default"] == 72 and meta[ds]["n_heldout"] == 30 for ds in DS)
cm = pd.read_csv(EXT / "tables/wave1_comention.csv")
size_cm = float(cm[cm.experiment == "size_imp_v1"].comention_rho.iloc[0])
small = next(r for r in W.CLOUDS["size"] if r["id"] == "small")
T = lambda s: "\\texttt{" + s.replace("_", "\\_").replace("{", "\\{").replace("}", "\\}") + "}"

# ---- the subsection ----------------------------------------------------------------------------
L = []
L.append(r"""% =====================================================================
% GENERATED by paper/build_appendix_analogy_prompts.py from wave1_workshop.ipynb (the template
% cell the generator exec'd) and wave1_families.py. Re-run the builder instead of hand-editing
% the verbatim blocks. Goes after the water-word subsection (app:genprompts-h2o) and before
% "Held-out evaluation prompts".
% =====================================================================
\subsection{Analogies (\texttt{size\_imp\_v1}, \texttt{war\_arg\_v1}, \texttt{food\_ideas\_v1}, \texttt{frames\_v1})}
\label{app:genprompts-analogy}
""")
L.append(
 f"{sum(nvec.values())} concepts in four experiments: the three metaphor mappings of "
 r"Table~\ref{tab:analogy_pairs} "
 f"({nvec['size_imp_v1']}, {nvec['war_arg_v1']} and {nvec['food_ideas_v1']} vectors) and the "
 r"professional frames of Table~\ref{tab:frames_grid} "
 f"({nvec['frames_v1']} vectors). "
 r"""Each metaphor has two clouds, a source (\emph{size}, \emph{war}, \emph{food}) and a
target (\emph{importance}, \emph{argument}, \emph{ideas}); the frames have five, one per
profession.

Construction mirrors the body-part and water-word experiments. Positive and negative are two
continuations of the same pooled prompt. There is one pool per experiment, shared by all of its
clouds, so every vector in an experiment is estimated over the same 72 prompts. The negative is
generated once per prompt for each cloud and shared by every concept in that cloud; it removes
that cloud's own domain and nothing else. The two sides of a mapping are written in independent
calls, and no call mentions the other side's concepts, so the text generator never sees the
correspondence.

The positive template is the body-part template with additions. Two are part of the design.
\texttt{\{lock\}} keeps each concept in its literal sense: \emph{attack} is an armed assault,
never a criticism. \texttt{\{avoid\}} bans the other cloud's vocabulary, so the two sides of a
mapping cannot share surface words, and the phrases English has already fused (``attack her
argument'', ``a half-baked idea'') never appear in training text. The rest were added after pilot
generations showed specific failures: the concept arriving as a simile for the scene (``much
like a commander executing a Fabian strategy'') or as something the assistant itself does, the
concept being implied without being named, positives running longer than negatives, and
markdown appearing on the positive side only. Both templates therefore carry the same hard
length cap and the same plain-prose instruction.

Each pool excludes both domains and also the relational structure under test: the pool for
\textsc{argument is war}, for instance, may not set up two opposing sides of anything, which would
hand both clouds the correspondence through their shared prompts. Candidate prompts pass the
common filter plus a domain screen, which discards any prompt containing a word from either
cloud's own domain list (for \texttt{war\_arg\_v1}, words such as \emph{war}, \emph{battle},
\emph{trench}, \emph{argue}, \emph{debate} and \emph{claim}). The screen was added after a pilot
pool asked for a scene at ``the bottom of the Mariana Trench'', which put a war word into every
completion written for that prompt.
""")

L.append("\\subsubsection*{Prompt pools}\n\nOne per experiment, shared by all of its clouds.\n")
for ds in DS:
    L.append(f"\\paragraph{{{T(ds)}}}\n" + verbatim(vwrap(W.POOLS[ds])))

L.append(r"""\subsubsection*{Positive continuation}

\texttt{\{desc\}} is the concept as a noun phrase (\emph{a military attack},
\emph{spoon-feeding someone}, \emph{a dental chart}), \texttt{\{gloss\}} a hand-written
definition of its literal sense, and \texttt{\{term\}} the concept's own word, which the
response is asked to use (\emph{attack}, \emph{spoon-feed}, \emph{dental chart}). The concepts
are listed in Tables~\ref{tab:analogy_pairs} and~\ref{tab:frames_grid}. \texttt{\{kind\}}, \texttt{\{lock\}} and
\texttt{\{avoid\}} are fixed per cloud and given below. As in the body-part template, the
\texttt{\{kind\}} line discourages mentioning the cloud's other concepts but does not forbid it.
""")
L.append(verbatim(vwrap(POS)))
L.append(
 r"""\texttt{\{scale\}} is empty except in \textsc{importance is size}, the one ordered family,
where it names the concept's neighbours on its own ladder. For \emph{small} it reads:
""" + verbatim(vwrap(W.scale_note(small).strip())) +
 r"""It was added because, without it, the generator collapsed neighbouring rungs onto one
intensifier (\emph{sizable} and \emph{colossal} both came out as \emph{massive}). It never
mentions the other ladder, so the pairing stays hidden, but it does give each ladder's own order
to the generator, and whether the two orders match is most of what that family's test measures.
The rungs' co-mention pattern in the training text is not shared between the two ladders
""" + f"($\\rho = {size_cm:+.2f}$" + r""" under the pairing), so the neighbour names did not
reach the text directly.
""")

L.append(r"""\subsubsection*{Negative continuation (shared per cloud)}

\texttt{\{scrub\}} removes the cloud's own domain and is given per cloud below.
""")
L.append(verbatim(vwrap(NEG)))

L.append(r"""\subsubsection*{Per-cloud slots}

The fixed slots for each cloud, exactly as substituted. \texttt{\{avoid\}} always withholds the
\emph{other} cloud's domain: the \emph{size} cloud is told not to say whether anything matters,
and the \emph{importance} cloud not to describe anything by its size.
""")

def row(lab, a, b):
    return f"\\texttt{{\\{{{lab}\\}}}} & \\raggedright {tex(a)} & \\raggedright {tex(b)} \\tabularnewline"

for ds in DS[:3]:
    a, b = W.EXPERIMENT_CLOUDS[ds]
    ra, rb = W.CLOUDS[a][0], W.CLOUDS[b][0]
    for cl in (a, b):
        r0 = W.CLOUDS[cl][0]
        assert all(r["lock"] == r0["lock"] and r["avoid_key"] == r0["avoid_key"] for r in W.CLOUDS[cl])
    body = "\n".join([
        row("kind", W.CLOUD_KIND[a], W.CLOUD_KIND[b]),
        row("lock", ra["lock"], rb["lock"]),
        row("avoid", W.AVOID_CLAUSE[ra["avoid_key"]], W.AVOID_CLAUSE[rb["avoid_key"]]),
        row("scrub", W.SCRUB[a], W.SCRUB[b])])
    L.append(f"\\paragraph{{{T(ds)}}}\\mbox{{}}\\par\\noindent{{\\footnotesize\\setlength{{\\tabcolsep}}{{4pt}}"
             "\\begin{tabular}{@{}l p{6.9cm} p{6.9cm}@{}}\n\\toprule\n"
             f" & \\emph{{{a}}} (source) & \\emph{{{b}}} (target) \\\\\n\\midrule\n{body}\n"
             "\\bottomrule\n\\end{tabular}}\\par\n")

# frames: lock and avoid are one sentence with the practice substituted -- print the template once
FR = W.F1_FRAMES; d0 = W.CLOUDS[FR[0]][0]; name0 = W.F1_FRAME_NAME[FR[0]]
lock_t = d0["lock"].replace(name0, "{frame}")
avoid_t = W.AVOID_CLAUSE[d0["avoid_key"]].replace(name0, "{frame}")
kinds = {W.CLOUD_KIND[f] for f in FR}
assert len(kinds) == 1
for f in FR:
    r0 = W.CLOUDS[f][0]; nm = W.F1_FRAME_NAME[f]
    assert r0["lock"] == lock_t.replace("{frame}", nm), f
    assert W.AVOID_CLAUSE[r0["avoid_key"]] == avoid_t.replace("{frame}", nm), f
    assert all(r["lock"] == r0["lock"] and r["avoid_key"] == r0["avoid_key"] for r in W.CLOUDS[f])
blk = []
for lab, txt in [("{kind}", kinds.pop()), ("{lock}", lock_t), ("{avoid}", avoid_t)]:
    pre = f"{lab:<8} "
    blk += textwrap.wrap(txt, WIDTH, initial_indent=pre, subsequent_indent=" " * len(pre),
                         break_long_words=False, break_on_hyphens=False)
frows = "\n".join(f"{f} & \\raggedright {tex(W.F1_FRAME_NAME[f])} & \\raggedright {tex(W.SCRUB[f])} \\tabularnewline" for f in FR)
L.append("\\paragraph{" + T("frames_v1") + "}\nThe five professions share \\texttt{\\{kind\\}}, and their "
         "\\texttt{\\{lock\\}} and \\texttt{\\{avoid\\}} are one sentence each with the profession's practice "
         "substituted for \\texttt{\\{frame\\}}:\n" + verbatim("\n".join(blk)) +
         "\\par\\noindent{\\footnotesize\\setlength{\\tabcolsep}{4pt}"
         "\\begin{tabular}{@{}l p{3.3cm} p{10.3cm}@{}}\n\\toprule\ncloud & \\texttt{\\{frame\\}} & "
         "\\texttt{\\{scrub\\}} \\\\\n\\midrule\n" + frows + "\n\\bottomrule\n\\end{tabular}}\\par\n")

# ---- optional companion file: every concept's desc / gloss / term ------------------------------
OUT2 = HERE / "latex" / "appendix_analogy_concept_slots.tex"
C = [r"""% =====================================================================
% OPTIONAL companion to appendix_analogy_prompts.tex (GENERATED by the same builder).
% Every concept's {term}, {desc} and {gloss}, exactly as substituted. Self-contained: drop it
% and nothing in the main subsection dangles.
% =====================================================================
\subsubsection*{Per-concept slots}

Every concept's \texttt{\{term\}}, \texttt{\{desc\}} and \texttt{\{gloss\}}, exactly as
substituted into the positive template.
"""]
for ds in DS:
    rows = []
    for cl in W.EXPERIMENT_CLOUDS[ds]:
        rows.append(f"\\multicolumn{{3}}{{@{{}}l}}{{\\emph{{{tex(cloud_label(cl))}}}}} \\\\")
        for r in W.CLOUDS[cl]:
            rows.append(f"{tex(W.head_term(r))} & \\raggedright {tex(r['desc'])} & \\raggedright {tex(r['gloss'])} \\tabularnewline")
    C.append("\\begin{table}[H]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
             "\\begin{tabular}{@{}l p{3.9cm} p{8.3cm}@{}}\n\\toprule\n"
             "\\texttt{\\{term\\}} & \\texttt{\\{desc\\}} & \\texttt{\\{gloss\\}} \\\\\n\\midrule\n"
             + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
             f"\\caption{{Per-concept slots for {T(ds)} ({nvec[ds]} vectors).}}\n"
             f"\\label{{tab:genprompts-slots-{ds.replace('_', '-')}}}\n\\end{{table}}\n")
OUT2.write_text("\n".join(C))
print(f"wrote {OUT2}  ({OUT2.stat().st_size} bytes)")

OUT.write_text("\n".join(L))
print(f"wrote {OUT}  ({OUT.stat().st_size} bytes)")
print(f"vectors {nvec}  | size_imp co-mention rho {size_cm:+.3f}")
