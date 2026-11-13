"""Generate the wave-1 metaphor + parallel-frames datasets.

Four datasets, eleven clouds, 94 vectors. Construction is h2o_v1/bodypart_v1's:

  v = (concept | prompt) - (domain absent | prompt)

pos and neg are two continuations of the SAME ordinary-writing request. The absent neg is
generated ONCE per prompt and shared across every concept IN THAT CLOUD (paired by prompt),
so every vector = mean(pos_c) - mean(neg_cloud) over identical prompts and the only
systematic difference is the concept. One prompt pool per DATASET, shared by both of its
clouds, so the carrier is common-mode across the very pair being compared.

Items, glosses, sense-locks, pools, scrubs and ban lists live in wave1_families.py.
Prompt TEMPLATES live in wave1_workshop.ipynb, which is the source of truth — this script
holds no prompt text of its own, it execs the notebook's cells by marker, exactly as
generate_h2o_v1.py / generate_bodypart_v1.py do.

~7.6k calls. Checkpoints after every concept, per dataset, so --resume costs nothing
already paid for.

Usage:
  python generate_wave1.py --dry                       # plan + vector specs, no API
  python generate_wave1.py --n-prompts 4               # smoke
  python generate_wave1.py --concurrency 48            # full (n=72)
  python generate_wave1.py --resume --concurrency 48   # continue an interrupted run
  python generate_wave1.py --datasets size_imp_v1      # one dataset only
"""
import json, sys, os, asyncio, argparse
from pathlib import Path

REPO = Path(".")
EXT = REPO / "fractal_analysis" / "extension"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(EXT))

# Which family module and which workshop notebook. Defaults reproduce the wave-1 run
# exactly; --families swaps in another family that exposes the same interface (see
# wave2_kg/kg_families.py). Parsed here rather than in main() because the module has to be
# imported before the vector specs are built.
_FAM = "wave1_families"
_NB_PATH = EXT / "wave1_workshop.ipynb"
_MARK = "WAVE1_PROMPTGEN ="
if "--families" in sys.argv:
    _FAM = sys.argv[sys.argv.index("--families") + 1]
    if _FAM != "wave1_families":
        _dir = next((d for d in EXT.iterdir() if d.is_dir() and (d / f"{_FAM}.py").exists()),
                    None)
        if _dir is None:
            raise SystemExit(f"no {_FAM}.py under {EXT}")
        sys.path.insert(0, str(_dir))
        _cands = sorted(_dir.glob("*_workshop.ipynb"))
        if len(_cands) != 1:
            raise SystemExit(f"expected exactly one *_workshop.ipynb in {_dir}, got {_cands}")
        _NB_PATH = _cands[0]
        # Each workshop names its own templates dict; the render_* function names are
        # what stays stable across them (see the note above).
        _txt = _NB_PATH.read_text()
        _MARK = next((m for m in ("KG_PROMPTGEN =", "SCAN_PROMPTGEN =",
                                  "TRANSFORM_PROMPTGEN =") if m in _txt),
                     _MARK)

W = __import__(_FAM)

# ---- templates come from the notebook, not from here ------------------------------------
_nb = json.loads(_NB_PATH.read_text())
def _cell(marker):
    for c in _nb["cells"]:
        if c["cell_type"] == "code" and marker in "".join(c["source"]):
            return "".join(c["source"])
    raise RuntimeError(f"{_NB_PATH.name} has no cell containing {marker!r}")

ns = {}
for _m in ["import pipeline as P", "async def gemini", _MARK]:
    exec(_cell(_m), ns)
ns["MAX_CALLS"] = 10_000_000          # the notebook's 120 is an interactive guardrail

P              = ns["P"]
gem            = ns["gemini"]
filter_prompts = ns["filter_prompts"]
render_promptgen = ns["render_wave1_promptgen"]  # name is stable across workshops
render_cont      = ns["render_wave1_cont"]
render_absent    = ns["render_wave1_absent"]

RAWB = REPO / "data" / "raw"
INP = REPO / "inputs"


def spec_for(rec):
    """Same schema as h2o_v1 / bodypart_v1, plus the fields the RSA joins on:
    `cloud` (which family this vector belongs to), `rank` (position in T) and, for
    frames_v1, `role` -- role is the shared index X that lines two frames up."""
    tgt = {"id": rec["id"], "description": rec["desc"], "cloud": rec["cloud"],
           "rank": rec["rank"], "term": W.head_term(rec)}
    if "role" in rec:
        tgt["role"] = rec["role"]
        tgt["frame"] = rec["frame"]
    return dict(
        vector_id=rec["vector_id"], concept_id=rec["id"], perspective="steer",
        concept=rec["concept"], domain=rec["domain"],
        magnitude_axis=rec["magnitude_axis"],
        source_magnitude={"id": "absent",
                          "description": f"an ordinary scene with no {W.CLOUD_KIND[rec['cloud']]}"},
        target_magnitude=tgt)


async def gen_pool(ds, n_total):
    pool, per, screened = [], 24, 0
    for _ in range(24):
        if len(pool) >= n_total: break
        raws = await gem([render_promptgen(ds, per)], temperature=1.0,
                         max_output_tokens=1600, concurrency=4)
        cands = []
        for r in raws:
            cands.extend(P.parse_numbered(r, per))
        screened += sum(1 for c in cands if ns["pool_screen"](ds, c))
        pool = filter_prompts(pool + cands, 10_000, ds=ds)
    if len(pool) < n_total:
        raise RuntimeError(f"[{ds}] prompt pool exhausted: {len(pool)}/{n_total} after dedup")
    print(f"[{ds}] pool: {len(pool)} unique ({screened} rejected by the domain screen)",
          flush=True)
    return pool


async def complete(reqs, label, conc):
    print(f"  [{label}] {len(reqs)}", flush=True)
    return await gem(reqs, temperature=0.7, max_output_tokens=400, concurrency=conc)


def _lock(raw, ds):
    """Refuse to write a dataset another live process is already writing.

    This exists because it actually happened: a local run was sent SIGTERM, did not die,
    and kept generating alongside a SLURM job doing the same datasets. Both appended to one
    outputs.jsonl, so frames_v1 ended up with two prompt pools mixed into it (142 distinct
    prompts) -- which silently destroys the invariant the whole design rests on, that every
    vector is estimated over ONE identical prompt set.
    """
    lk = raw / "GENERATING.lock"
    if lk.exists():
        try:
            pid = int(json.loads(lk.read_text())["pid"])
            os.kill(pid, 0)
        except (ValueError, KeyError, json.JSONDecodeError, ProcessLookupError):
            pass                      # stale lock from a dead process -- take it
        except PermissionError:
            raise SystemExit(f"[{ds}] locked by live pid {pid} ({lk}); refusing to write")
        else:
            raise SystemExit(f"[{ds}] locked by live pid {pid} ({lk}); refusing to write")
    lk.write_text(json.dumps(dict(pid=os.getpid(), host=os.uname().nodename)))
    return lk


def check_integrity(raw, ds, recs, n_prompts):
    """Every vector must have the SAME prompt set, in the same order, with exactly one
    positive and one negative per prompt, and one shared negative per cloud. Anything else
    means the file was written by more than one run -- fail loudly rather than train on it."""
    rows = [json.loads(l) for l in open(raw / "outputs.jsonl")]
    byv, negs, bad = {}, {}, []
    for r in rows:
        byv.setdefault(r["vector_id"], []).append(r)
        if r["polarity"] == "negative":
            negs.setdefault(r["vector_id"].split("__")[0], {}).setdefault(
                r["prompt"], set()).add(r["completion"])
    pool = None
    for vid, rs in byv.items():
        pr = [r["prompt"] for r in rs if r["polarity"] == "positive"]
        if pool is None: pool = pr
        if len(rs) != 2 * n_prompts: bad.append(f"{vid}: {len(rs)} rows != {2*n_prompts}")
        if pr != pool: bad.append(f"{vid}: prompt set/order differs from the first vector")
    for cl, d in negs.items():
        n = sum(1 for v in d.values() if len(v) > 1)
        if n: bad.append(f"cloud {cl}: {n} prompts have more than one 'shared' negative")
    if len(byv) != len(recs): bad.append(f"{len(byv)} vectors on disk != {len(recs)} expected")
    if bad:
        raise SystemExit(f"[{ds}] INTEGRITY FAILURE -- not writing meta.json:\n  " +
                         "\n  ".join(bad[:10]))
    print(f"[{ds}] integrity OK: {len(byv)} vectors x {n_prompts} prompts, "
          f"one shared negative per cloud", flush=True)


async def run_dataset(ds, args):
    clouds = W.EXPERIMENT_CLOUDS[ds]
    recs = [r for cl in clouds for r in W.CLOUDS[cl]]
    N = args.n_prompts
    raw = RAWB / ds
    raw.mkdir(parents=True, exist_ok=True)
    lk = _lock(raw, ds)
    ckpt = raw / "progress.json"

    # Resume reuses the SAME prompt pool and the SAME shared negatives -- a different pool
    # would break the pairing that makes every vector in a cloud share one negative.
    if args.resume and ckpt.exists():
        st = json.loads(ckpt.read_text())
        train_p, held_p, neg, done = st["train_p"], st["held_p"], st["neg"], set(st["done"])
        print(f"[{ds}] resuming: {len(done)}/{len(recs)} concepts done", flush=True)
        # A kill between writing a concept's rows and updating the checkpoint would leave
        # rows on disk for a vector the checkpoint still calls undone, and resume would then
        # append them a second time. The checkpoint is the authority: drop any row whose
        # vector_id it does not list.
        op = raw / "outputs.jsonl"
        if op.exists():
            keep = [l for l in op.read_text().splitlines(True)
                    if json.loads(l)["vector_id"] in done]
            n_before = sum(1 for _ in open(op))
            if len(keep) != n_before:
                print(f"[{ds}] pruned {n_before - len(keep)} rows from an interrupted "
                      f"concept", flush=True)
            op.write_text("".join(keep))
    else:
        pool = await gen_pool(ds, N + args.n_heldout + 10)
        train_p, held_p = pool[:N], pool[N:N + args.n_heldout]
        neg = {}
        for cl in clouds:
            neg[cl] = await complete([render_absent(cl, p) for p in train_p],
                                     f"{ds} absent:{cl} (shared)", args.concurrency)
        done = set()
        (raw / "outputs.jsonl").write_text("")
        ckpt.write_text(json.dumps(dict(train_p=train_p, held_p=held_p, neg=neg, done=[])))

    out = open(raw / "outputs.jsonl", "a")
    for k, rec in enumerate(recs, 1):
        if rec["vector_id"] in done: continue
        pos = await complete([render_cont(rec, p) for p in train_p],
                             f"{ds} {k}/{len(recs)} {rec['vector_id']}", args.concurrency)
        nc = neg[rec["cloud"]]
        for i, p in enumerate(train_p):
            out.write(json.dumps(dict(vector_id=rec["vector_id"], prompt=p,
                                      completion=pos[i], polarity="positive")) + "\n")
            out.write(json.dumps(dict(vector_id=rec["vector_id"], prompt=p,
                                      completion=nc[i], polarity="negative")) + "\n")
        out.flush()
        done.add(rec["vector_id"])
        ckpt.write_text(json.dumps(dict(train_p=train_p, held_p=held_p, neg=neg,
                                        done=sorted(done))))
    out.close()

    check_integrity(raw, ds, recs, len(train_p))

    with open(raw / "heldout_prompts.jsonl", "w") as f:
        for rec in recs:
            for idx, p in enumerate(held_p):
                f.write(json.dumps(dict(vector_id=rec["vector_id"],
                                        heldout_prompt_idx=idx, prompt=p)) + "\n")
    (raw / "meta.json").write_text(json.dumps(dict(
        dataset=ds, n_vectors=len(recs), n_prompts_default=len(train_p),
        n_heldout=len(held_p), clouds=clouds, model=ns.get("GEN_MODEL"),
        correspondence=W.EXPERIMENTS[ds]["T"], shape=W.EXPERIMENTS[ds]["shape"],
        vectors=[r["vector_id"] for r in recs]), indent=2))

    n = sum(1 for _ in open(raw / "outputs.jsonl"))
    errs = sum(1 for l in open(raw / "outputs.jsonl") if '"[ERROR' in l)
    lk.unlink(missing_ok=True)
    print(f"[{ds}] DONE {n} rows ({n//2} pos / {n//2} neg), "
          f"{len(recs)*len(held_p)} heldout, {errs} API errors -> {raw}", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=72)
    ap.add_argument("--n-heldout", type=int, default=30)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--datasets", nargs="*", default=list(W.EXPERIMENTS))
    ap.add_argument("--families", default="wave1_families")  # consumed above
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    total = 0
    for ds in args.datasets:
        recs = [r for cl in W.EXPERIMENT_CLOUDS[ds] for r in W.CLOUDS[cl]]
        with open(INP / f"{ds}_vectors.jsonl", "w") as f:
            for r in recs: f.write(json.dumps(spec_for(r)) + "\n")
        calls = args.n_prompts * (len(recs) + len(W.EXPERIMENT_CLOUDS[ds]))
        total += calls
        print(f"[{ds}] {len(recs):3d} vectors, {len(W.EXPERIMENT_CLOUDS[ds])} clouds "
              f"-> {ds}_vectors.jsonl | ~{calls} continuation calls")
    print(f"projected ~{total} continuation calls + prompt-gen")
    if args.dry:
        print("--dry: no API.\n\n--- sample positive that would be sent ---")
        r0 = W.CLOUDS[W.EXPERIMENT_CLOUDS[args.datasets[0]][0]][0]
        print(render_cont(r0, "Explain how you get a stuck jar open."))
        return

    for ds in args.datasets:
        await run_dataset(ds, args)

    print("\nnext, per dataset:")
    for ds in args.datasets:
        print(f"  python convert_to_axbench.py "
              f"--vectors inputs/{ds}_vectors.jsonl "
              f"--raw-dir data/raw/{ds} "
              f"--out-dir data/axbench/{ds}")


if __name__ == "__main__":
    asyncio.run(main())
