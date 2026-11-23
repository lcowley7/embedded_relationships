"""Convert one dataset's generated text (data/raw/<ds>/) to AxBench-format parquets (data/axbench/<ds>/).

Reads:
  --vectors   data/inputs/<ds>_vectors.jsonl  (one row per vector_id; defines stable ordering)
  --raw-dir   data/raw/<ds>/                  (outputs.jsonl + heldout_prompts.jsonl)

Writes (to --out-dir):
  metadata.jsonl          AxBench-format, one line per vector. `concept` field = vector_id.
  train_data.parquet      Columns: input, output, output_concept, concept_genre, category,
                          dataset_category, concept_id.
  heldout_eval.parquet    Columns: concept_id, input_id, original_prompt, input.
  concept_index.parquet   Our extra metadata for rubric routing. NOT consumed by AxBench.
  conversion_report.json  Counts + validation summary.

============================================================================
EEEEE PROVENANCE
============================================================================
AxBench's negative-row sentinel is the literal string `"EEEEE"` in `output_concept`.
Upstream AxBench's `prepare_df()` pools ALL rows with `output_concept == "EEEEE"`
across concepts when training a vector — fine for AxBench's own pooled-negatives
setup, but WRONG for us.

Our negatives are PROMPT-PAIRED: each negative shares the prompt with its matched
positive, with an opposing continuation. To preserve this pairing through AxBench's
data loader, we write every negative with:
   - output_concept = "EEEEE"          (sentinel required by AxBench's filter)
   - concept_id     = <int of the paired vector>   (matches the positive)

Then the patched `prepare_df()` (pipeline/axbench/axbench.patch)
filters by `concept_id` to recover only this vector's paired negatives.

**Do NOT change `"EEEEE"` to anything else** — AxBench's filter greps for it literally.
============================================================================
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd

EMPTY_CONCEPT = "EEEEE"   # AxBench sentinel; do not change. See module docstring.


def load_vectors(vectors_path: Path) -> list[dict]:
    rows = []
    with open(vectors_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_concept_index(vectors: list[dict]) -> pd.DataFrame:
    """Stable concept_id assignment by source order in vectors_path.

    Re-running on the same input must produce the same concept_id mapping.
    """
    rows = []
    for cid, vec in enumerate(vectors):
        src_mag = vec.get("source_magnitude") or {}
        tgt_mag = vec.get("target_magnitude") or {}
        rows.append({
            "concept_id":            cid,
            "vector_id":             vec["vector_id"],
            "concept_group":         vec["concept_id"],          # SHORT name like "happy" (NB field shadowing in source)
            "concept_description":   vec.get("concept"),         # human-readable, e.g. "happy (feeling pleasure or contentment)"; used by judge rubrics
            "perspective":           vec["perspective"],
            "domain":                vec["domain"],
            "functional_form_id":    vec.get("functional_form_id"),
            "functional_form":       vec.get("functional_form"),
            "source_magnitude_id":   src_mag.get("id"),
            "target_magnitude_id":   tgt_mag.get("id"),
            "source_magnitude_desc": src_mag.get("description"),
            "target_magnitude_desc": tgt_mag.get("description"),
            "magnitude_axis":        vec.get("magnitude_axis"),
        })
    df = pd.DataFrame(rows)
    assert df["concept_id"].is_unique
    assert df["vector_id"].is_unique
    return df


def write_metadata_jsonl(concept_index: pd.DataFrame, out_path: Path):
    """Write AxBench-format metadata.jsonl. One line per concept_id."""
    with open(out_path, "w") as f:
        for _, row in concept_index.iterrows():
            entry = {
                "concept_id":          int(row["concept_id"]),
                "concept":             row["vector_id"],
                "ref":                 None,
                "concept_genres_map":  {row["vector_id"]: ["text"]},
            }
            f.write(json.dumps(entry) + "\n")


def stream_train_data(outputs_path: Path, vec_to_cid: dict[str, int]) -> pd.DataFrame:
    """Stream outputs.jsonl into a long DataFrame with AxBench columns.

    Schema mapping (per row):
        input            <- row["prompt"]
        output           <- row["completion"]
        output_concept   <- row["vector_id"] if positive else "EEEEE"
        concept_genre    <- "text"
        category         <- row["polarity"]
        dataset_category <- "instruction"
        concept_id       <- vec_to_cid[row["vector_id"]]
    """
    inputs, outputs, output_concepts = [], [], []
    categories, concept_ids = [], []
    n_pos_per_vec = Counter()
    n_neg_per_vec = Counter()

    with open(outputs_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            vid = r["vector_id"]
            cid = vec_to_cid.get(vid)
            if cid is None:
                # vector in outputs.jsonl but not in minimal_test.jsonl — shouldn't happen
                continue
            polarity = r["polarity"]
            inputs.append(r["prompt"])
            outputs.append(r["completion"])
            # EEEEE PROVENANCE — see module docstring.
            output_concepts.append(vid if polarity == "positive" else EMPTY_CONCEPT)
            categories.append(polarity)
            concept_ids.append(cid)
            if polarity == "positive":
                n_pos_per_vec[vid] += 1
            else:
                n_neg_per_vec[vid] += 1

    df = pd.DataFrame({
        "input":            inputs,
        "output":           outputs,
        "output_concept":   output_concepts,
        "concept_genre":    "text",
        "category":         categories,
        "dataset_category": "instruction",
        "concept_id":       concept_ids,
    })
    return df, n_pos_per_vec, n_neg_per_vec


def stream_heldout(heldout_path: Path, vec_to_cid: dict[str, int]) -> pd.DataFrame:
    """Stream heldout_prompts.jsonl into AxBench-ish eval dataframe."""
    concept_ids, input_ids, prompts = [], [], []
    n_per_vec = Counter()
    with open(heldout_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            vid = r["vector_id"]
            cid = vec_to_cid.get(vid)
            if cid is None:
                continue
            concept_ids.append(cid)
            input_ids.append(r["heldout_prompt_idx"])
            prompts.append(r["prompt"])
            n_per_vec[vid] += 1
    df = pd.DataFrame({
        "concept_id":      concept_ids,
        "input_id":        input_ids,
        "original_prompt": prompts,
        "input":           prompts,
    })
    return df, n_per_vec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vectors", required=True, type=Path,
                    help="data/inputs/<ds>_vectors.jsonl (defines stable concept_id ordering)")
    ap.add_argument("--raw-dir", required=True, type=Path,
                    help="data/raw/<ds> (contains outputs.jsonl + heldout_prompts.jsonl)")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="data/axbench/<ds> (will be created)")
    ap.add_argument("--cid-source", type=Path, default=None,
                    help="Optional concept_index.parquet to inherit concept_ids from. "
                         "If set, each vector_id in --vectors must appear in this index, and "
                         "its concept_id is taken from there (positional assignment is bypassed). "
                         "Not used for the paper's datasets.")
    args = ap.parse_args()

    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Vectors → concept_index → vector_id ↔ concept_id mapping.
    vectors = load_vectors(args.vectors)
    concept_index = build_concept_index(vectors)
    if args.cid_source is not None:
        # Override positional cids with cids from an external concept_index.
        src = pd.read_parquet(args.cid_source)
        src_map = dict(zip(src["vector_id"], src["concept_id"].astype(int)))
        missing = [vid for vid in concept_index["vector_id"] if vid not in src_map]
        if missing:
            raise ValueError(f"[convert] --cid-source missing {len(missing)} vector_ids; "
                             f"first 3: {missing[:3]}")
        concept_index["concept_id"] = concept_index["vector_id"].map(src_map).astype(int)
        print(f"[convert] inherited concept_ids from {args.cid_source} "
              f"(range {concept_index['concept_id'].min()}..{concept_index['concept_id'].max()})",
              flush=True)
    vec_to_cid = dict(zip(concept_index["vector_id"], concept_index["concept_id"]))
    print(f"[convert] {len(vectors)} vectors", flush=True)

    # 2) Stream outputs.jsonl → train_data.parquet (optional — quick track has no retrain).
    outputs_path = args.raw_dir / "outputs.jsonl"
    if outputs_path.exists():
        train_df, n_pos, n_neg = stream_train_data(outputs_path, vec_to_cid)
        print(f"[convert] train rows: {len(train_df)} "
              f"(pos={sum(n_pos.values())}, neg={sum(n_neg.values())})", flush=True)
    else:
        print(f"[convert] {outputs_path} not found — skipping train_data.parquet "
              "(quick track: reusing pre-trained vectors)", flush=True)
        train_df = pd.DataFrame(columns=["input", "output", "output_concept",
                                         "concept_genre", "category",
                                         "dataset_category", "concept_id"])
        n_pos = Counter()
        n_neg = Counter()

    # 3) Stream heldout_prompts.jsonl → heldout_eval.parquet (if file present)
    heldout_path = args.raw_dir / "heldout_prompts.jsonl"
    if heldout_path.exists():
        heldout_df, n_held = stream_heldout(heldout_path, vec_to_cid)
        print(f"[convert] heldout rows: {len(heldout_df)} across {len(n_held)} vectors", flush=True)
    else:
        print(f"[convert] WARNING: {heldout_path} not found — skipping heldout_eval.parquet", flush=True)
        heldout_df = pd.DataFrame(columns=["concept_id", "input_id", "original_prompt", "input"])
        n_held = Counter()

    # 4) Write all outputs.
    if len(train_df) > 0:
        train_df.to_parquet(args.out_dir / "train_data.parquet", engine="pyarrow", compression="snappy")
    if len(heldout_df) > 0:
        heldout_df.to_parquet(args.out_dir / "heldout_eval.parquet", engine="pyarrow", compression="snappy")
    concept_index.to_parquet(args.out_dir / "concept_index.parquet", engine="pyarrow", compression="snappy")
    write_metadata_jsonl(concept_index, args.out_dir / "metadata.jsonl")

    # 5) Validation + report
    expected_per_vec = 72  # from meta.json's n_prompts_default
    weak = []
    for vec in vectors:
        vid = vec["vector_id"]
        p, n = n_pos.get(vid, 0), n_neg.get(vid, 0)
        if p < expected_per_vec * 0.5 or n < expected_per_vec * 0.5:
            weak.append({"vector_id": vid, "n_pos": p, "n_neg": n})

    # 5b) Heldout uniqueness audit. Catches the original parse_numbered/dedup
    # bug: heldout vectors with fewer unique prompts than the eval expects.
    heldout_uniq_warnings = []
    if len(heldout_df) > 0:
        per_vec = heldout_df.groupby("concept_id").agg(
            n_total=("input", "size"),
            n_unique=("input", "nunique"),
        ).reset_index()
        # Map back to vector_id for legibility
        cid_to_vid = dict(zip(concept_index["concept_id"], concept_index["vector_id"]))
        for _, row in per_vec.iterrows():
            if row["n_unique"] < row["n_total"]:
                heldout_uniq_warnings.append({
                    "concept_id": int(row["concept_id"]),
                    "vector_id":  cid_to_vid.get(int(row["concept_id"]), "?"),
                    "n_total":    int(row["n_total"]),
                    "n_unique":   int(row["n_unique"]),
                })
        if heldout_uniq_warnings:
            print(f"[convert] WARNING: {len(heldout_uniq_warnings)} vectors have duplicate "
                  f"heldout prompts. Top 5:", flush=True)
            for w in heldout_uniq_warnings[:5]:
                print(f"    {w['vector_id']}: unique={w['n_unique']}/{w['n_total']}", flush=True)

    report = {
        "n_vectors":            len(vectors),
        "n_train_rows":         len(train_df),
        "n_train_positives":    int((train_df["category"] == "positive").sum()),
        "n_train_negatives":    int((train_df["category"] == "negative").sum()),
        "n_heldout_rows":       len(heldout_df),
        "n_heldout_vectors":    len(n_held),
        "expected_per_vector":  expected_per_vec,
        "weak_vectors":         weak,   # < 50% of expected; consider regenerating
        "heldout_duplicate_vectors": heldout_uniq_warnings,
        "elapsed_sec":          round(time.time() - t0, 1),
        "ts":                   time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (args.out_dir / "conversion_report.json").write_text(json.dumps(report, indent=2))
    print(f"[convert] DONE in {report['elapsed_sec']}s. "
          f"Weak vectors: {len(weak)}. "
          f"Heldout-dup vectors: {len(heldout_uniq_warnings)}", flush=True)


if __name__ == "__main__":
    main()
