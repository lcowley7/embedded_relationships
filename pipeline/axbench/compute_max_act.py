"""Compute per-vector max_act from heldout + train + concept-pool prompts.

Replaces AxBench `inference.py --mode latent` for our pipeline. Does not require
a backing OpenAI/vLLM endpoint or AxBench's seed_sentences/seed_instructions HF datasets.

For each trained DiffMean vector v_i (concept_id i):
  pool = heldout(v_i)        # 12 prompts (vector's own held-out)
       + train_inputs(v_i)   # 24 sampled from v_i's training inputs (unique)
       + concept_pool(c)     # 36 sampled from sibling vectors' training inputs
                             #   where c = v_i's concept_group
  for each prompt in pool:
    forward through base model
    gather residual at layer L
    exclude first `prefix_length` tokens (chat-template wrap)
    project onto SIGN_TABLE[perspective] * v_i
    per-prompt max = max over remaining content tokens
  max_act = max over prompts in pool

Outputs (in <dump_dir>/inference/):
  - max_activations.json       : {concept_id_str: float}
  - latent_data.parquet         : AxBench-compatible per-vector parquet
                                  (consumed by pre_compute_mean_activations transparently)
  - latent_diagnostics.parquet  : per-vector audit trail

Sign convention: identity (no flip). Every vector in the paper is a "steer" vector, used on its
own held-out prompts in its natural direction.

Fallback: if max_act <= 0 across the entire pool, falls back to DEFAULT_MAX_ACT (50.0)
and sets fell_back_to_default=True in diagnostics.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SIGN_TABLE = {
    "steer": +1,
    "anti_steer": +1,
    "steer_ref": +1,
    "anti_steer_ref": +1,
    "steer_vocab": +1,
    "anti_steer_vocab": +1,
    "magnitude": +1,
}

POOL_HELDOUT = 12
POOL_TRAIN = 24
POOL_CONCEPT = 36
DEFAULT_MAX_ACT = 50.0
SEED = 42
MODEL_NAME = "google/gemma-2-2b-it"


def get_prefix_length(tokenizer):
    a = tokenizer.apply_chat_template([{"role": "user", "content": "1"}], tokenize=True)
    b = tokenizer.apply_chat_template([{"role": "user", "content": "2"}], tokenize=True)
    for i, (ta, tb) in enumerate(zip(a, b)):
        if ta != tb:
            return i
    return min(len(a), len(b))


def gather_residual_at_layer(model, layer_idx, input_ids, attention_mask):
    captured = []

    def hook(mod, inputs, outputs):
        captured.append(outputs[0])
        return outputs

    h = model.model.layers[layer_idx].register_forward_hook(hook, always_call=True)
    with torch.no_grad():
        model.forward(input_ids=input_ids, attention_mask=attention_mask)
    h.remove()
    return captured[0]


def build_pool(cid, concept_group, heldout, train_inputs_by_cid, sibling_pool_by_group, rng):
    pool = []
    h = heldout[heldout["concept_id"] == cid]["input"].tolist()[:POOL_HELDOUT]
    for p in h:
        pool.append(("heldout", p))

    t = train_inputs_by_cid.get(cid, [])
    own_train_set = set(t)
    if t:
        n = min(POOL_TRAIN, len(t))
        for p in rng.sample(t, n):
            pool.append(("train", p))

    sib = sibling_pool_by_group.get(concept_group, [])
    sib_excl = [p for p in sib if p not in own_train_set]
    if sib_excl:
        n = min(POOL_CONCEPT, len(sib_excl))
        for p in rng.sample(sib_excl, n):
            pool.append(("concept_pool", p))

    return pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--data-dir", required=True, help="e.g. data/axbench/h2o_v1")
    parser.add_argument("--max-concepts", type=int, default=None,
                        help="If set, only process the first N concept_ids (for smoke testing)")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--method", default="DiffMean",
                        choices=["DiffMean", "PreferenceVector"],
                        help="AxBench model name. Used for dump-dir name "
                             "(diffmean_lN vs preference_vector_lN), weight "
                             "filename, and latent_data column.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    # DiffMean → diffmean_l{N}; PreferenceVector → preference_vector_l{N}.
    method_dir = "diffmean" if args.method == "DiffMean" else "preference_vector"
    dump_dir = data_dir / f"{method_dir}_l{args.layer}"
    train_dir = dump_dir / "train"
    out_dir = dump_dir / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[compute_max_act] layer={args.layer}  dump_dir={dump_dir}  max_concepts={args.max_concepts}")

    print("[compute_max_act] loading inputs")
    ci = pd.read_parquet(data_dir / "concept_index.parquet").sort_values("concept_id").reset_index(drop=True)
    heldout = pd.read_parquet(data_dir / "heldout_eval.parquet")
    train_data = pd.read_parquet(data_dir / "train_data.parquet")

    if args.max_concepts is not None:
        ci = ci.iloc[:args.max_concepts].reset_index(drop=True)
        print(f"[compute_max_act] limited to first {len(ci)} concepts")

    train_inputs_by_cid = {
        int(cid): list(set(grp["input"].tolist()))
        for cid, grp in train_data.groupby("concept_id")
    }
    full_ci = pd.read_parquet(data_dir / "concept_index.parquet")
    sibling_pool_by_group = {}
    for grp_name, grp in full_ci.groupby("concept_group"):
        prompts = set()
        for cid in grp["concept_id"].tolist():
            prompts.update(train_inputs_by_cid.get(int(cid), []))
        sibling_pool_by_group[grp_name] = list(prompts)

    weight_file = f"{args.method}_weight.pt"
    print(f"[compute_max_act] loading vectors from {train_dir}/{weight_file}")
    weights = torch.load(train_dir / weight_file, map_location="cpu", weights_only=False).to(torch.float32)
    norms = weights.norm(dim=1)
    print(f"[compute_max_act] vectors: shape={tuple(weights.shape)}  norm_range=[{float(norms.min()):.4f},{float(norms.max()):.4f}]")
    if args.method == "DiffMean":
        # DiffMean vectors are unit-normalized at training time. bf16 training can leave a
        # ~1% norm drift on the odd vector (layer-dependent), which tripped a too-tight 1e-2
        # assert. Renormalize that rounding away, but still hard-fail a genuinely degenerate
        # vector (>10% off = likely zero/NaN, not rounding). PreferenceVector: arbitrary norm, skip.
        dev = float((norms - 1.0).abs().max())
        assert dev < 0.1, f"vectors badly non-unit (max dev {dev:.3f}) — likely degenerate, not bf16 rounding"
        if dev >= 1e-2:
            print(f"[compute_max_act] note: max unit-norm deviation {dev:.4f} (bf16 rounding) -> renormalizing")
        weights = weights / norms.clamp(min=1e-8).unsqueeze(1)

    print(f"[compute_max_act] loading {MODEL_NAME}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    prefix_length = get_prefix_length(tokenizer)
    print(f"[compute_max_act] model loaded in {time.time()-t0:.1f}s, prefix_length={prefix_length}")

    diagnostics = []
    max_acts = {}
    t_loop = time.time()

    for idx, row in ci.iterrows():
        cid = int(row["concept_id"])
        perspective = row["perspective"]
        concept_group = row["concept_group"]
        sign = SIGN_TABLE.get(perspective, +1)

        rng = random.Random(hash((SEED, cid)) & 0xFFFFFFFF)
        pool = build_pool(cid, concept_group, heldout, train_inputs_by_cid, sibling_pool_by_group, rng)

        raw_v = weights[cid].to(model.device)
        if args.method == "DiffMean":
            # DiffMean vectors are unit-norm, so projecting onto the raw vector is
            # the same as normalising first.
            v = raw_v * sign
        else:
            # PreferenceVector has arbitrary norm — normalize so max_act
            # describes "how much activation aligns with this direction",
            # comparable to DiffMean. The trained vector's *magnitude* is
            # used separately as steering strength at inference time.
            v = (raw_v / (raw_v.norm() + 1e-12)) * sign

        per_prompt_max = []
        per_prompt_source = []
        per_prompt_argmax = []

        for i in range(0, len(pool), args.batch_size):
            batch = pool[i:i + args.batch_size]
            tags, prompts = zip(*batch)
            enc = tokenizer(list(prompts), return_tensors="pt", padding=True, add_special_tokens=True).to(model.device)
            act = gather_residual_at_layer(model, args.layer, enc["input_ids"], enc["attention_mask"])
            mask = enc["attention_mask"].bool()
            for j in range(len(batch)):
                seq_act = act[j][mask[j]].to(torch.float32)
                if seq_act.shape[0] <= prefix_length:
                    per_prompt_max.append(float("-inf"))
                    per_prompt_argmax.append(-1)
                    per_prompt_source.append(tags[j])
                    continue
                content = seq_act[prefix_length:]
                proj = (content @ v).cpu().numpy()
                per_prompt_max.append(float(proj.max()))
                per_prompt_argmax.append(int(proj.argmax()))
                per_prompt_source.append(tags[j])

        arr = np.array(per_prompt_max)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            max_act = DEFAULT_MAX_ACT
            fell_back = True
            median = float("nan")
            min_v = float("nan")
            max_raw = float("nan")
        else:
            max_raw = float(finite.max())
            median = float(np.median(finite))
            min_v = float(finite.min())
            if max_raw <= 0:
                max_act = DEFAULT_MAX_ACT
                fell_back = True
                print(f"[compute_max_act] WARN cid={cid} ({perspective}) max_act_raw={max_raw:.3f} <= 0; falling back to {DEFAULT_MAX_ACT}")
            else:
                max_act = max_raw
                fell_back = False

        max_acts[cid] = max_act

        diagnostics.append({
            "concept_id": cid,
            "vector_id": row["vector_id"],
            "perspective": perspective,
            "concept_group": concept_group,
            "domain": row["domain"],
            "n_pool": len(pool),
            "n_heldout": per_prompt_source.count("heldout"),
            "n_train": per_prompt_source.count("train"),
            "n_concept_pool": per_prompt_source.count("concept_pool"),
            "max_act": max_act,
            "max_act_raw": max_raw,
            "median": median,
            "min": min_v,
            "n_neg_or_zero_prompts": int((arr <= 0).sum()),
            "fell_back_to_default": fell_back,
            "sign_used": sign,
            "per_prompt_max": arr.tolist(),
            "per_prompt_source": list(per_prompt_source),
            "per_prompt_argmax_pos": per_prompt_argmax,
        })

        if (idx + 1) % 50 == 0 or idx == 0 or idx == len(ci) - 1:
            elapsed = time.time() - t_loop
            avg = elapsed / (idx + 1)
            eta = avg * (len(ci) - idx - 1) / 60
            print(f"[compute_max_act] vec {idx+1}/{len(ci)} cid={cid} {perspective:20s} max_act={max_act:7.2f} fell_back={str(fell_back):5s} elapsed={elapsed/60:5.1f}m eta={eta:5.1f}m")

    print(f"[compute_max_act] loop done in {(time.time()-t_loop)/60:.1f}m")

    diag_df = pd.DataFrame(diagnostics)
    # NOTE: file name must NOT match AxBench's `latent_*.parquet` glob — its
    # pre_compute_mean_activations (model.py:399) iterates all matching files and
    # demands a {ModelName}_max_act column. Diagnostics have richer per-vector
    # info but not that column; so use a name outside the glob.
    diag_path = out_dir / "max_act_diagnostics.parquet"
    diag_df.to_parquet(diag_path)
    print(f"[compute_max_act] wrote {diag_path}  ({len(diag_df)} rows)")

    json_path = out_dir / "max_activations.json"
    json_path.write_text(json.dumps({str(k): v for k, v in sorted(max_acts.items())}, indent=2))
    print(f"[compute_max_act] wrote {json_path}")

    ax_df = pd.DataFrame({
        "concept_id": [cid for cid, _ in sorted(max_acts.items())],
        f"{args.method}_max_act": [v for _, v in sorted(max_acts.items())],
    })
    ax_path = out_dir / "latent_data.parquet"
    ax_df.to_parquet(ax_path)
    print(f"[compute_max_act] wrote {ax_path}  (AxBench-compatible; consumed by pre_compute_mean_activations)")

    print("\n[compute_max_act] === SUMMARY ===")
    print(f"  Total vectors: {len(diag_df)}")
    print(f"  Fell back to default (max_act_raw <= 0): {diag_df['fell_back_to_default'].sum()}")
    print(f"  max_act distribution: "
          f"min={diag_df['max_act'].min():.2f}  "
          f"median={diag_df['max_act'].median():.2f}  "
          f"mean={diag_df['max_act'].mean():.2f}  "
          f"max={diag_df['max_act'].max():.2f}")
    print(f"  Per-perspective stats:")
    summary = diag_df.groupby("perspective")["max_act"].agg(["count", "min", "median", "mean", "max", lambda s: (s == DEFAULT_MAX_ACT).sum()])
    summary.columns = ["n", "min", "median", "mean", "max", "n_fallback"]
    print(summary.to_string())


if __name__ == "__main__":
    main()
