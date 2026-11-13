"""steer.py -- steering evaluation for a big model at one layer (PLAN.md S6), matched to the paper's
Gemma protocol so the steering appendix and the efficacy sweep compare like for like.

For every dataset, at layer --layer, using the AxBench-span vectors (train/DiffMean_weight.pt):
  1. max_act per concept, exactly as olmo/olmo_maxact.py (the chat-template adaptation of
     compute_max_act.py, which set the paper's Gemma max_act): pool = 12 held-out + 24 own-train
     + up to 36 same-concept_group prompts, each chat-wrapped as at steering time; project the layer-L
     residual on the unit vector; max over tokens after the chat prefix; max over the pool.
  2. steered generation, as AxBench HeldoutLocal + DiffMean steering did for Gemma:
       prompts   each concept's first 12 held-out prompts (sorted by input_id), one user turn,
                 add_generation_prompt=True, the model's chat_kwargs (Qwen3: thinking off)
       steering  a hook on model.model.layers[L] adds factor * max_act * v at EVERY position -- the prompt
                 pass and every decode step (pyvene with unit_locations=None steers all tokens)
       factors   Gemma's 15: 0.0 .. 5.0
       sampling  do_sample, temperature 1.0, top_k 50, top_p 1.0 (what Gemma got: HF defaults, as its
                 generation_config sets none); max_new_tokens 512 (the Gemma config)
Outputs, in the AxBench layout gemini_judge.py and build_scores.py read:
  <root>/models/<tag>/axbench/<ds>/diffmean_l<L>/inference/
      max_activations.json, max_act_diagnostics.parquet, steering_data.parquet
Resumable: a dataset with steering_data.parquet is skipped; generations are checkpointed to parts/ every
--save-every batches and reloaded on restart. On CUDA OOM the batch is halved and retried.
Usage: python steer.py --tag olmo31_32b --root ~/lt --layer 25 [--datasets all|a,b] [--after-dataset CMD]
"""
from __future__ import annotations
import argparse, json, random, subprocess, sys, time
from pathlib import Path

import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import MODELS, DATASETS          # noqa: E402
from extract import load_model, ct_ids      # noqa: E402

FACTORS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0]
N_PROMPTS = 12                                   # AxBench steering_num_of_examples
POOL_HELDOUT, POOL_TRAIN, POOL_CONCEPT = 12, 24, 36
DEFAULT_MAX_ACT, SEED = 50.0, 42


def user_ids(tok, kw, prompt):
    return ct_ids(tok, [{"role": "user", "content": prompt}], True, kw)


def prefix_len(tok, kw):
    """Tokens before the user content (AxBench get_prefix_length, with the model's chat_kwargs)."""
    a = ct_ids(tok, [{"role": "user", "content": "1"}], False, kw)
    b = ct_ids(tok, [{"role": "user", "content": "2"}], False, kw)
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def left_pad(seqs, pad_id, device):
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    am = torch.zeros(len(seqs), T, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, T - len(s):] = torch.tensor(s)
        am[i, T - len(s):] = 1
    return ids.to(device), am.to(device)


@torch.no_grad()
def max_acts(model, tok, kw, layer, W, ci, heldout, train, bs, device):
    """olmo_maxact.py, with sorted() so the pool does not depend on PYTHONHASHSEED."""
    P = prefix_len(tok, kw)
    train_by = {int(c): sorted(set(g["input"])) for c, g in train.groupby("concept_id")}
    sib_by = {grp: sorted({p for c in g.concept_id for p in train_by.get(int(c), [])})
              for grp, g in ci.groupby("concept_group")}
    cap = {}
    h = model.model.layers[layer].register_forward_hook(
        lambda m, i, o: cap.__setitem__("a", o[0] if isinstance(o, tuple) else o))
    out, diags = {}, []
    try:
        for _, row in ci.iterrows():
            cid = int(row.concept_id)
            rng = random.Random(hash((SEED, cid)) & 0xFFFFFFFF)
            own = train_by.get(cid, [])
            pool = heldout[heldout.concept_id == cid].sort_values("input_id")["input"].tolist()[:POOL_HELDOUT]
            pool += rng.sample(own, min(POOL_TRAIN, len(own)))
            sib = [p for p in sib_by.get(row.concept_group, []) if p not in set(own)]
            pool += rng.sample(sib, min(POOL_CONCEPT, len(sib)))
            v = W[cid].to(device, torch.float32)
            best = []
            for i in range(0, len(pool), bs):
                seqs = [user_ids(tok, kw, p) for p in pool[i:i + bs]]
                ids, am = left_pad(seqs, tok.pad_token_id, device)
                model.model(input_ids=ids, attention_mask=am, use_cache=False)
                a = cap["a"]
                for j, s in enumerate(seqs):
                    real = a[j, ids.shape[1] - len(s):].float()          # left padding: real tokens at the end
                    if real.shape[0] > P:
                        best.append(float((real[P:] @ v).max()))
            finite = [b for b in best if np.isfinite(b)]
            fell = not finite or max(finite) <= 0
            m = DEFAULT_MAX_ACT if fell else max(finite)
            out[cid] = m
            diags.append(dict(concept_id=cid, vector_id=row.vector_id, max_act=m, n_pool=len(pool),
                              median=float(np.median(finite)) if finite else float("nan"), fell_back=fell))
    finally:
        h.remove()
    return out, pd.DataFrame(diags)


def run_dataset(ds, a, cfg, tok, model, device):
    kw = cfg["chat_kwargs"]
    src = Path(a.root) / "inputs" / ds
    d = Path(a.root) / "models" / a.tag / "axbench" / ds / f"diffmean_l{a.layer}"
    inf = d / "inference"
    if (inf / "steering_data.parquet").exists() and not a.force:
        print(f"[steer] {ds} L{a.layer}: done already, skipping", flush=True)
        return
    inf.mkdir(parents=True, exist_ok=True)
    ci = pd.read_parquet(src / "concept_index.parquet").sort_values("concept_id").reset_index(drop=True)
    if a.limit_concepts:
        ci = ci.head(a.limit_concepts)
    heldout = pd.read_parquet(src / "heldout_eval.parquet")
    W = torch.load(d / "train" / "DiffMean_weight.pt", weights_only=False).float()
    assert (W.norm(dim=1) - 1).abs().max() < 1e-3 and W.shape[0] >= len(ci)

    # ---- 1. max_act
    ma_path = inf / "max_activations.json"
    if ma_path.exists():
        ma = {int(k): v for k, v in json.loads(ma_path.read_text()).items()}
    else:
        t0 = time.time()
        ma, diag = max_acts(model, tok, kw, a.layer, W, ci, heldout,
                            pd.read_parquet(src / "train_data.parquet"), a.batch_size, device)
        ma_path.write_text(json.dumps({str(k): v for k, v in sorted(ma.items())}, indent=2))
        diag.to_parquet(inf / "max_act_diagnostics.parquet", index=False)
        print(f"[steer] {ds} L{a.layer}: max_act median {diag.max_act.median():.2f} "
              f"(fell back {int(diag.fell_back.sum())}) in {time.time() - t0:.0f}s", flush=True)

    # ---- 2. requests: concept x first 12 held-out prompts x factors (AxBench HeldoutLocal order)
    reqs = []
    for _, r in ci.iterrows():
        cid = int(r.concept_id)
        hp = heldout[heldout.concept_id == cid].sort_values("input_id").head(a.max_prompts)
        for _, hr in hp.iterrows():
            for f in a.factors:
                reqs.append(dict(dataset_name="HeldoutLocal", concept_id=cid, input_concept=r.vector_id,
                                 input_id=int(hr.input_id), factor=float(f),
                                 original_prompt=hr.original_prompt, input=hr["input"]))
    parts = inf / "parts"
    parts.mkdir(exist_ok=True)
    done = pd.concat([pd.read_parquet(p) for p in sorted(parts.glob("part_*.parquet"))], ignore_index=True) \
        if any(parts.glob("part_*.parquet")) else pd.DataFrame()
    key = lambda r: (r["concept_id"], r["input_id"], round(r["factor"], 4))
    have = set(map(key, done.to_dict("records"))) if len(done) else set()
    todo = [r for r in reqs if key(r) not in have]
    todo.sort(key=lambda r: len(r["input"]))                       # similar prompt lengths per batch
    print(f"[steer] {ds} L{a.layer}: {len(reqs)} generations ({len(have)} already done)", flush=True)

    # ---- steering hook: every position, prompt pass and each decode step
    Wd = W.to(device, torch.bfloat16)
    ma_t = torch.zeros(W.shape[0], dtype=torch.bfloat16, device=device)
    for k, v in ma.items():
        ma_t[k] = v
    st = {}

    def hook(mod, inp, out):
        if not st:
            return out
        h = out[0] if isinstance(out, tuple) else out
        add = ((ma_t[st["cid"]] * st["fac"]).unsqueeze(-1) * Wd[st["cid"]]).unsqueeze(1)   # (B,1,D)
        if h.shape[1] > 1:                                        # prompt pass: real (unpadded) positions
            add = add * st["am"][:, -h.shape[1]:].unsqueeze(-1).to(add.dtype)
        hn = h + add.to(h.dtype)
        return (hn,) + tuple(out[1:]) if isinstance(out, tuple) else hn

    stop = [t for t in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
            if isinstance(t, int) and t >= 0 and t != tok.unk_token_id]
    handle = model.model.layers[a.layer].register_forward_hook(hook)
    t0, n_gen, n_tok, bs, i, k = time.time(), 0, 0, a.batch_size, 0, len(list(parts.glob("part_*.parquet")))
    buf = []
    try:
        while i < len(todo):
            batch = todo[i:i + bs]
            seqs = [user_ids(tok, kw, r["input"]) for r in batch]
            ids, am = left_pad(seqs, tok.pad_token_id, device)
            st.update(cid=torch.tensor([r["concept_id"] for r in batch], device=device),
                      fac=torch.tensor([r["factor"] for r in batch], dtype=torch.bfloat16, device=device), am=am)
            torch.manual_seed(SEED + i)
            try:
                gen = model.generate(input_ids=ids, attention_mask=am, max_new_tokens=a.max_new_tokens,
                                     do_sample=True, temperature=1.0, top_k=50, top_p=1.0,
                                     repetition_penalty=1.0, eos_token_id=stop, pad_token_id=tok.pad_token_id)
            except torch.cuda.OutOfMemoryError:
                st.clear(); torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                print(f"[steer] OOM -> batch size {bs}", flush=True)
                continue
            finally:
                st.clear()
            new = gen[:, ids.shape[1]:]
            for r, row in zip(batch, new):
                r = dict(r)
                r["DiffMean_steered_generation"] = tok.decode(row, skip_special_tokens=True)
                buf.append(r)
                n_tok += int((row != tok.pad_token_id).sum())
            n_gen += len(batch)
            i += len(batch)
            if len(buf) >= a.save_every * bs or i >= len(todo):
                pd.DataFrame(buf).to_parquet(parts / f"part_{k:05d}.parquet", index=False)
                k += 1
                buf = []
                el = time.time() - t0
                print(f"[steer] {ds} L{a.layer}: {len(have) + i}/{len(reqs)} | {n_gen / el:.2f} gen/s, "
                      f"{n_tok / el:,.0f} tok/s | bs {bs} | ETA {(len(todo) - i) / max(n_gen / el, 1e-9) / 3600:.1f} h",
                      flush=True)
    finally:
        handle.remove()
        if buf:
            pd.DataFrame(buf).to_parquet(parts / f"part_{k:05d}.parquet", index=False)

    allp = pd.concat([pd.read_parquet(p) for p in sorted(parts.glob("part_*.parquet"))], ignore_index=True)
    allp = allp.drop_duplicates(subset=["concept_id", "input_id", "factor"])
    assert len(allp) == len(reqs), (len(allp), len(reqs))
    allp.sort_values(["concept_id", "input_id", "factor"]).to_parquet(inf / "steering_data.parquet", index=False)
    for p in parts.glob("part_*.parquet"):
        p.unlink()
    parts.rmdir()
    json.dump(dict(tag=a.tag, layer=a.layer, factors=a.factors, n_prompts=a.max_prompts,
                   max_new_tokens=a.max_new_tokens, sampling=dict(temperature=1.0, top_k=50, top_p=1.0),
                   steering="all positions, prompt + every decode step", stop_ids=stop,
                   n_generations=len(allp), seconds=round(time.time() - t0)),
              open(inf / "steer_meta.json", "w"), indent=2)
    print(f"[steer] {ds} L{a.layer}: DONE {len(allp)} generations -> {inf / 'steering_data.parquet'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=sorted(MODELS))
    ap.add_argument("--root", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--factors", type=float, nargs="*", default=FACTORS)
    ap.add_argument("--max-prompts", type=int, default=N_PROMPTS)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--save-every", type=int, default=4, help="checkpoint every N batches")
    ap.add_argument("--limit-concepts", type=int, default=None, help="pilots only")
    ap.add_argument("--after-dataset", default=None, help="shell command run after each dataset (e.g. sync)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = MODELS[a.tag]
    dss = DATASETS if a.datasets == "all" else a.datasets.split(",")
    tok, model = load_model(cfg, "cuda")
    tok.padding_side = "left"
    device = next(model.parameters()).device
    print(f"[steer] {a.tag} L{a.layer} | {len(dss)} datasets | factors {a.factors} | "
          f"{a.max_prompts} prompts | {a.max_new_tokens} tokens | bs {a.batch_size}", flush=True)
    for ds in dss:
        run_dataset(ds, a, cfg, tok, model, device)
        if a.after_dataset:        # placeholders {tag} {ds} {layer}, e.g. to upload just this folder
            subprocess.run(a.after_dataset.format(tag=a.tag, ds=ds, layer=a.layer), shell=True)


if __name__ == "__main__":
    main()
