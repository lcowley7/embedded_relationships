"""extract_sae_acts.py -- "DiffMean in SAE feature space" (activation view).

Re-runs gemma-2-2b-it over the SAME training texts the DiffMean steering vectors were built
from, with the SAME AxBench formatting (chat template, user=input, model=output, BOS + 4-token
chat prefix excluded, every remaining token pooled), and at layers 10 and 20 stores per concept:

  sum_resid  [2304]   sum of raw residual activations          -> reconstructs DiffMean (sanity check)
  sum_act    [16384]  sum of GemmaScope JumpReLU feature acts  -> mean SAE code per concept
  sum_fire   [16384]  number of tokens on which each feature fires
  ntok                number of pooled tokens

for every concept's positives and for each dataset's (shared) negatives.
SAEs: google/gemma-scope-2b-pt-res, width_16k, average_l0_77 (L10) / average_l0_71 (L20)
-- the Neuronpedia canonical 16k residual SAEs. NB: trained on BASE gemma-2-2b, applied here to
-it residuals (standard practice; reconstruction quality is reported by the job).

Needs data/axbench/<ds>/train_data.parquet (pipeline/axbench/convert_to_axbench.py) and the GemmaScope weights.
Output: data/sae_acts/acts_<dataset>.npz
Run (GPU): sbatch pipeline/slurm/sae_extract.slurm
"""
import os, sys, time
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # repository root
from relgeom.paths import AXBENCH, SAE_ACTS, ensure
DB = str(AXBENCH)
OUT = str(ensure(SAE_ACTS))
SNAP = (os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")) + ("/hub/models--google--gemma-scope-2b-pt-res/"
        "snapshots/fd571b47c1c64851e9b1989792367b9babb4af63"))
SAE_PATH = {10: f"{SNAP}/layer_10/width_16k/average_l0_77/params.npz",
            20: f"{SNAP}/layer_20/width_16k/average_l0_71/params.npz"}
LAYERS = [10, 20]
DATASETS = (["h2o_v1"] + [f"temp_c1{s}" for s in "abcd"] + [f"temp_f1{s}" for s in "abcdef"]
            + ["size_imp_v1", "war_arg_v1", "food_ideas_v1"])
if len(sys.argv) > 1: DATASETS = sys.argv[1:]
BS = 32
dev = "cuda"

tok = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", torch_dtype=torch.float32,
                                             attn_implementation="eager").to(dev).eval()
model.model.layers = model.model.layers[: max(LAYERS) + 1]   # nothing past L20 is needed


def sae(L):
    p = np.load(SAE_PATH[L])
    return {k: torch.tensor(p[k], dtype=torch.float32, device=dev) for k in p.files}
SAES = {L: sae(L) for L in LAYERS}

# AxBench prefix / suffix lengths (utils/model_utils.py)
def _pre_suf():
    a = tok.apply_chat_template([{"role": "user", "content": "1"}], tokenize=True)
    b = tok.apply_chat_template([{"role": "user", "content": "2"}], tokenize=True)
    pre = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
    suf = next(i for i, (x, y) in enumerate(zip(reversed(a), reversed(b))) if x != y)
    return pre, suf
PREFIX, SUFFIX = _pre_suf()
print(f"prefix_length={PREFIX} suffix_length={SUFFIX}", flush=True)


def fmt(inp, out):
    """AxBench scripts/train.py binarize branch for chat models without system prompt."""
    ids = tok.apply_chat_template([{"role": "user", "content": inp}, {"role": "assistant", "content": out}],
                                  tokenize=True, add_generation_prompt=True)[1:-SUFFIX]
    return tok.decode(ids)


acts = {}
hooks = [model.model.layers[L].register_forward_hook(
    (lambda L: lambda m, i, o: acts.__setitem__(L, o[0] if isinstance(o, tuple) else o))(L)) for L in LAYERS]


@torch.no_grad()
def pool(texts):
    """token-pooled sums over a list of formatted texts (AxBench token weighting)."""
    r = {L: dict(sum_resid=torch.zeros(2304, device=dev, dtype=torch.float64),
                 sum_act=torch.zeros(16384, device=dev, dtype=torch.float64),
                 sum_fire=torch.zeros(16384, device=dev, dtype=torch.float64),
                 sse=0.0, ssx=0.0, l0=0.0) for L in LAYERS}
    n = 0
    order = np.argsort([len(t) for t in texts])
    for k in range(0, len(texts), BS):
        batch = [texts[i] for i in order[k:k + BS]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(dev)
        model(**enc)
        m = enc["attention_mask"].clone(); m[:, :PREFIX] = 0; m = m.bool()
        n += int(m.sum())
        for L in LAYERS:
            x = acts[L][m].float()                                     # [T, 2304]
            S = SAES[L]
            pre = x @ S["W_enc"] + S["b_enc"]
            f = pre * (pre > S["threshold"])
            rec = f @ S["W_dec"] + S["b_dec"]
            d = r[L]
            d["sum_resid"] += x.sum(0).double(); d["sum_act"] += f.sum(0).double()
            d["sum_fire"] += (f > 0).sum(0).double()
            d["sse"] += float(((rec - x) ** 2).sum()); d["ssx"] += float(((x - x.mean(0)) ** 2).sum())
            d["l0"] += float((f > 0).sum())
    return r, n


tok.padding_side = "right"
for ds in DATASETS:
    t0 = time.time()
    df = pd.read_parquet(f"{DB}/{ds}/train_data.parquet")
    ci = pd.read_parquet(f"{DB}/{ds}/concept_index.parquet")
    pos = df[df.category == "positive"]
    negall = df[df.category == "negative"]
    cid = dict(zip(ci.vector_id, ci.concept_id))
    vids = ci.sort_values("concept_id").vector_id.tolist()
    # AxBench trains each concept against the negatives carrying its concept_id; in the analogy
    # datasets each cloud has its own negative pool, so negatives are pooled per distinct set.
    negkey, negsets = {}, {}
    for v in vids:
        g = negall[negall.concept_id == cid[v]].drop_duplicates(["input", "output"])
        k = hash(tuple(sorted(zip(g.input, g.output))))
        negkey[v] = k; negsets.setdefault(k, g)
    nkeys = list(negsets)
    res = {L: {k: [] for k in ("sum_resid", "sum_act", "sum_fire")} for L in LAYERS}
    ntok, stats = [], {L: [0.0, 0.0, 0.0] for L in LAYERS}
    groups = [(f"__NEG{i}__", negsets[k]) for i, k in enumerate(nkeys)] + [(v, pos[pos.output_concept == v]) for v in vids]
    for v, g in groups:
        assert len(g) > 0, (ds, v)
        r, n = pool([fmt(a, b) for a, b in zip(g.input, g.output)])
        ntok.append(n)
        for L in LAYERS:
            for k in res[L]: res[L][k].append(r[L][k].cpu().numpy())
            stats[L][0] += r[L]["sse"]; stats[L][1] += r[L]["ssx"]; stats[L][2] += r[L]["l0"]
    neg = negall.drop_duplicates(["input", "output"])
    save = dict(vector_id=np.array([g[0] for g in groups]), ntok=np.array(ntok),
                n_texts=np.array([len(g[1]) for g in groups]),
                neg_index=np.array([nkeys.index(negkey[v]) for v in vids]))
    for L in LAYERS:
        for k in res[L]: save[f"L{L}_{k}"] = np.stack(res[L][k]).astype(np.float32)
        save[f"L{L}_fvu"] = np.array(stats[L][0] / stats[L][1]); save[f"L{L}_l0"] = np.array(stats[L][2] / sum(ntok))
    np.savez_compressed(f"{OUT}/acts_{ds}.npz", **save)
    print(f"{ds}: {len(vids)} concepts, {len(neg)} unique negatives, {sum(ntok)} tokens, "
          + " ".join(f"L{L} FVU={stats[L][0]/stats[L][1]:.3f} L0={stats[L][2]/sum(ntok):.1f}" for L in LAYERS)
          + f"  [{time.time()-t0:.0f}s]", flush=True)
print("done")
