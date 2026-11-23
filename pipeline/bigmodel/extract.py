"""extract.py -- DiffMean vectors at every layer of a model, one forward pass per unique text.

Reproduces AxBench DiffMean (axbench/models/mean.py + scripts/train.py::prepare_df +
models/probe.py::make_data_module), which is how the paper's Gemma vectors were made:

  text  = decode( chat_template([user: prompt, assistant: response], add_generation_prompt=True)
                  [strip leading BOS] [: -user_turn_suffix] )
  ids   = tokenizer(text, max_length=1024, truncation=True)        (adds BOS if the tokenizer does)
  pool  = every token from the first user-content token to the end of `ids`, per layer, over all
          72 positive rows vs all 72 negative rows of the concept (token-weighted means)
  v     = mean_pos - mean_neg, unit-normalised

Two deliberate, documented differences, both no-ops for Gemma:
  * the leading token is stripped only if it IS the BOS (AxBench strips it unconditionally, which
    would delete a real `<|im_start|>` for OLMo/Qwen);
  * the prefix length is measured in the space of the ids actually pooled (so a BOS added by the
    tokenizer is accounted for).

Two spans come out of the same forward pass:
  train/          span "axb"      the AxBench span above (the paper's procedure)
  train_content/  span "content"  first user-content token .. last response token (no template
                                  tokens after the response) -- a robustness variant

Gate G2 is built in: before any compute, one real row is rendered, both spans are decoded, and the
run aborts if the content span does not start at the prompt and end at the response.

Engineering: all requested layers are hooked at once and reduced on the GPU (masked token sums in
fp32); identical (prompt, response) rows -- the shared negatives -- are run once and credited to
every concept that uses them, which is exact because DiffMean is a token-weighted mean; texts are
length-sorted and packed to a token budget; a dataset is skipped if its outputs already exist.

Layout: data/models/<tag>/axbench/<dataset>/{concept_index.parquet, extract_meta.json,
        diffmean_l<L>/{train,train_content}/DiffMean_weight.pt, diffmean_l<L>/means.pt (--save-means)}
Inputs: data/axbench/<dataset>/train_data.parquet etc. (pipeline/axbench/convert_to_axbench.py).
--root DIR reads DIR/inputs/<dataset>/ and writes DIR/models/<tag>/... instead (a scratch disk on a GPU VM).
Usage:  python pipeline/bigmodel/extract.py --tag olmo31_32b [--datasets all|a,b] [--layers all|5,10]
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys, time
from pathlib import Path

import numpy as np, pandas as pd, torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root
from relgeom.models import MODELS, DATASETS   # noqa: E402
from relgeom.paths import AXBENCH, MODELS as MODELS_DIR   # noqa: E402

def src_dir(a, ds):
    """AxBench-format inputs of ds: data/axbench/<ds>/ in this repository, or <root>/inputs/<ds>/ with --root."""
    return Path(a.root) / "inputs" / ds if a.root else AXBENCH / ds


def model_dir(a):
    """Where this model's vectors and steering outputs live: data/models/<tag>/axbench/, or <root>/models/... ."""
    return (Path(a.root) / "models" if a.root else MODELS_DIR) / a.tag / "axbench"


MAX_LEN = 1024                    # AxBench probe.make_data_module truncation
N_PER_POLARITY = 72               # AxBench max_num_of_examples 144 -> head(72) each


class _Stop(Exception):
    pass


# ------------------------------------------------------------------ templates ----------------
def ct_ids(tok, msgs, agp, kw):
    out = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=agp, **kw)
    if hasattr(out, "input_ids"):                   # transformers 5.x returns a BatchEncoding
        out = out["input_ids"]
    if torch.is_tensor(out):
        out = out.tolist()
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class Spans:
    """Tokenisation + span bookkeeping for one model."""

    def __init__(self, tok, kw):
        self.tok, self.kw = tok, kw
        u = lambda c: [{"role": "user", "content": c}]
        # AxBench get_suffix_length: common tail of two user-only renders
        a, b = ct_ids(tok, u("1"), False, kw), ct_ids(tok, u("2"), False, kw)
        self.user_suffix = common_prefix(a[::-1], b[::-1])
        self.bos = tok.bos_token_id
        # prefix and tail measured in the space of the ids we actually pool
        self.prefix = common_prefix(self.ids("1", "X"), self.ids("2", "X"))
        x, y = self.ids("Q", "1"), self.ids("Q", "2")
        self.tail = common_prefix(x[::-1], y[::-1])   # template tokens after the response

    def text(self, prompt, response):
        ids = ct_ids(self.tok, [{"role": "user", "content": prompt},
                                {"role": "assistant", "content": response}], True, self.kw)
        if ids and self.bos is not None and ids[0] == self.bos:
            ids = ids[1:]
        if self.user_suffix:
            ids = ids[:len(ids) - self.user_suffix]
        return self.tok.decode(ids)

    def ids(self, prompt, response):
        return self.tok(self.text(prompt, response), max_length=MAX_LEN,
                        truncation=True)["input_ids"]

    def bounds(self, n_ids, full_len):
        """(start, end_axb, end_content) for a row whose untruncated length was full_len."""
        return self.prefix, n_ids, min(n_ids, full_len - self.tail)


def norm_ws(s):
    return re.sub(r"\s+", "", s)


def gate_g2(sp, row):
    """Abort unless the content span is exactly prompt .. response."""
    ids = sp.ids(row["input"], row["output"])
    full = len(sp.tok(sp.text(row["input"], row["output"]))["input_ids"])
    s, e_axb, e_con = sp.bounds(len(ids), full)
    con = sp.tok.decode(ids[s:e_con])
    tail = sp.tok.decode(ids[e_con:e_axb])
    info = dict(prefix=sp.prefix, user_suffix=sp.user_suffix, tail_tokens=sp.tail,
                bos_id=sp.bos, rendered_head=sp.tok.decode(ids[:s]), content_head=con[:120],
                content_tail=con[-120:], axb_tail_after_response=tail)
    ok = (norm_ws(con).startswith(norm_ws(row["input"])[:40])
          and norm_ws(con).endswith(norm_ws(row["output"])[-40:]))
    for k, v in info.items():
        print(f"[G2] {k}: {v!r}", flush=True)
    if not ok:
        raise SystemExit("[G2] FAILED: content span is not prompt..response -- see above")
    print("[G2] passed", flush=True)
    return info


# ------------------------------------------------------------------ extraction ---------------
def snapshot_path(cfg):
    """Local folder of the pinned commit (downloads it only if HF_HUB_OFFLINE is not set)."""
    import os
    from huggingface_hub import snapshot_download
    return snapshot_download(cfg["hf_id"], revision=cfg["revision"],
                             local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1")


def load_model(cfg, device):
    path = snapshot_path(cfg)                  # load from the folder: no revision lookup at load time
    print(f"[extract] loading {path}", flush=True)
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dkey = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    dt = torch.bfloat16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        path, **{dkey: dt}, device_map=device,
        attn_implementation=cfg.get("attn", "eager" if "gemma" in cfg["hf_id"] else "sdpa")).eval()
    return tok, model


def run_dataset(ds, a, cfg, tok, model, sp, layers, out_root, g2_info):
    src = src_dir(a, ds)
    ci = pd.read_parquet(src / "concept_index.parquet").sort_values("concept_id").reset_index(drop=True)
    td = pd.read_parquet(src / "train_data.parquet")
    N = len(ci) if a.limit_concepts is None else min(a.limit_concepts, len(ci))
    assert list(ci.concept_id[:N]) == list(range(N)), f"{ds}: concept_id not contiguous"

    # rows per concept exactly as AxBench prepare_df selects them (head 72 each, file order)
    uses = {}                       # (prompt, response) -> list of slots (2c = pos, 2c+1 = neg)
    for c in range(N):
        pos = td[(td.concept_id == c) & (td.category == "positive")].head(N_PER_POLARITY)
        neg = td[(td.concept_id == c) & (td.category == "negative")].head(N_PER_POLARITY)
        assert len(pos) == len(neg) == N_PER_POLARITY, (ds, c, len(pos), len(neg))
        assert pos.output_concept.nunique() == 1 and neg.output_concept.eq("EEEEE").all()
        for slot, df in ((2 * c, pos), (2 * c + 1, neg)):
            for p, r in zip(df.input, df.output):
                uses.setdefault((p, r), []).append(slot)

    texts = list(uses)
    enc = [sp.ids(p, r) for p, r in texts]
    full = [len(tok(sp.text(p, r))["input_ids"]) for p, r in texts]
    n_trunc = sum(f > MAX_LEN for f in full)
    order = np.argsort([len(e) for e in enc])

    L, H, S = len(layers), model.config.hidden_size, 2 * N
    dev = next(model.parameters()).device
    acc = {k: torch.zeros(S, L, H, dtype=torch.float32, device=dev) for k in ("axb", "content")}
    cnt = {k: torch.zeros(S, dtype=torch.float64) for k in ("axb", "content")}
    shared = {}                                     # text index -> {span: (L,H)} for multi-slot texts
    li_of = {l: i for i, l in enumerate(layers)}
    buf, masks = {}, {}

    def hook_for(l):
        def hook(mod, inp, out):
            h = (out[0] if isinstance(out, tuple) else out).float()
            for k in ("axb", "content"):
                buf[k][:, li_of[l]] = torch.einsum("bsh,bs->bh", h, masks[k])
            if l == layers[-1] and a.early_exit:
                raise _Stop
        return hook

    handles = [model.model.layers[l].register_forward_hook(hook_for(l)) for l in layers]
    t0, n_tok, i = time.time(), 0, 0
    try:
        while i < len(order):
            # pack by token budget (texts are length-sorted, so padding stays small)
            j = i
            while j < len(order) and (j - i + 1) * len(enc[order[j]]) <= a.token_budget:
                j += 1
            j = max(j, i + 1)
            idx = order[i:j]
            B, T = len(idx), max(len(enc[k]) for k in idx)
            ids = torch.full((B, T), tok.pad_token_id, dtype=torch.long)
            am = torch.zeros(B, T, dtype=torch.long)
            mk = {k: torch.zeros(B, T, dtype=torch.float32) for k in ("axb", "content")}
            for b, k in enumerate(idx):
                e = enc[k]
                ids[b, :len(e)] = torch.tensor(e)
                am[b, :len(e)] = 1
                s, e_axb, e_con = sp.bounds(len(e), full[k])
                mk["axb"][b, s:e_axb] = 1
                mk["content"][b, s:e_con] = 1
            for k in mk:
                masks[k] = mk[k].to(dev)
                buf[k] = torch.zeros(B, L, H, dtype=torch.float32, device=dev)
            try:
                with torch.no_grad():
                    model.model(input_ids=ids.to(dev), attention_mask=am.to(dev), use_cache=False)
            except _Stop:
                pass
            for b, k in enumerate(idx):
                slots = uses[texts[k]]
                for sk in ("axb", "content"):
                    n = float(mk[sk][b].sum())
                    for s_ in slots:
                        cnt[sk][s_] += n
                if len(slots) == 1:
                    for sk in ("axb", "content"):
                        acc[sk][slots[0]] += buf[sk][b]
                else:
                    shared[k] = {sk: buf[sk][b].clone() for sk in ("axb", "content")}
            n_tok += int(am.sum())
            i = j
            if a.progress and (i == len(order) or i % max(1, len(order) // 10) < B):
                el = time.time() - t0
                print(f"[extract] {ds} {i}/{len(order)} texts | {n_tok/el:,.0f} tok/s", flush=True)
    finally:
        for h_ in handles:
            h_.remove()

    for k, v in shared.items():                     # credit shared texts to every concept using them
        slots = torch.tensor(uses[texts[k]], device=dev)
        for sk in ("axb", "content"):
            acc[sk].index_add_(0, slots, v[sk].unsqueeze(0).expand(len(slots), L, H))

    dst = out_root / ds
    for l in layers:
        li = li_of[l]
        means = {}
        for sk, sub in (("axb", "train"), ("content", "train_content")):
            c_ = cnt[sk].to(dev, torch.float32).clamp_min(1)[:, None]
            mean = acc[sk][:, li] / c_                       # (S, H)
            mp, mn = mean[0::2], mean[1::2]
            W = mp - mn
            W = (W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)).cpu()
            d = dst / f"diffmean_l{l}" / sub
            d.mkdir(parents=True, exist_ok=True)
            torch.save(W, d / "DiffMean_weight.pt")
            torch.save(torch.zeros(N), d / "DiffMean_bias.pt")   # placeholder: recombine stitches it
            if sk == "axb":
                means = {"mean_pos": mp.cpu(), "mean_neg": mn.cpu()}
        if a.save_means:
            torch.save(means, dst / f"diffmean_l{l}" / "means.pt")

    ci.iloc[:N].to_parquet(dst / "concept_index.parquet", index=False)
    el = time.time() - t0
    sha = {fn: hashlib.sha256((src / fn).read_bytes()).hexdigest()
           for fn in ("train_data.parquet", "concept_index.parquet")}
    meta = dict(tag=a.tag, model=cfg["hf_id"], revision=cfg["revision"], dataset=ds,
                layers=layers, n_concepts=N, n_texts=len(texts),
                n_shared_texts=len(shared), n_truncated=n_trunc, tokens=n_tok, seconds=round(el, 1),
                tokens_per_s=round(n_tok / max(el, 1e-9)), input_sha256=sha, g2=g2_info,
                transformers=transformers.__version__, torch=torch.__version__,
                mean_tokens_axb=float(cnt["axb"].mean()), mean_tokens_content=float(cnt["content"].mean()))
    json.dump(meta, open(dst / "extract_meta.json", "w"), indent=2)
    print(f"[extract] {ds}: {N} concepts, {len(texts)} texts ({len(shared)} shared), {n_tok:,} tok, "
          f"{el/60:.1f} min, {meta['tokens_per_s']:,} tok/s, truncated {n_trunc}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=sorted(MODELS))
    ap.add_argument("--root", default=None)
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--token-budget", type=int, default=16384)
    ap.add_argument("--limit-concepts", type=int, default=None, help="smoke tests only")
    ap.add_argument("--save-means", action="store_true")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cfg = MODELS[a.tag]
    layers = (list(range(cfg["n_layers"])) if a.layers == "all"
              else sorted(int(x) for x in a.layers.split(",")))
    a.early_exit = layers[-1] < cfg["n_layers"] - 1
    dss = DATASETS if a.datasets == "all" else a.datasets.split(",")
    out_root = model_dir(a)

    todo = [d for d in dss if a.force or not (out_root / d / "extract_meta.json").exists()]
    print(f"[extract] {a.tag} ({cfg['hf_id']} @ {cfg['revision'][:12]}) | layers {layers[0]}..{layers[-1]} "
          f"({len(layers)}) | {len(todo)}/{len(dss)} datasets to do", flush=True)
    if not todo:
        return
    tok, model = load_model(cfg, a.device)
    assert model.config.num_hidden_layers == cfg["n_layers"], model.config.num_hidden_layers
    assert model.config.hidden_size == cfg["hidden"], model.config.hidden_size
    sp = Spans(tok, cfg["chat_kwargs"])
    first = pd.read_parquet(src_dir(a, todo[0]) / "train_data.parquet")
    g2 = gate_g2(sp, first[first.category == "positive"].iloc[0])
    for ds in todo:
        run_dataset(ds, a, cfg, tok, model, sp, layers, out_root, g2)


if __name__ == "__main__":
    main()
