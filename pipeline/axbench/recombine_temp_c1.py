"""Stitch the 4 temp_c1 chunks (a/b/c/d) into one temperature-ordered 1 C sweep.

The 1 C sweep (-20..110 C, 131 levels) was trained as 4 SLURM chunks purely for
parallelism. Each chunk restarts concept_id at 0 and carries a chunk-specific vector_id
prefix (temp_c1a__steer__<t> .. temp_c1d__steer__<t>). Logically it is ONE sweep. This
script reassembles it WITHOUT touching the chunk dirs, writing a new unified dataset:

  data/axbench/temp_c1/diffmean_lN/
     train/DiffMean_weight.pt    stitched, rows ordered by temperature  (n_temp, d)
     train/DiffMean_bias.pt      stitched, same order
     eval/results.jsonl          pooled, re-keyed to the unified temp-ordered concept_id
     eval/per_row_scores.parquet pooled, with concept_id remapped + temp_c column
  data/axbench/temp_c1/concept_index.parquet  unified, temp-ordered, contiguous concept_id

For the headline curvature plot: load train/DiffMean_weight.pt (already temp-ordered) and
plot ||v_{T+1} - v_T|| vs T -> spikes expected at 0 C and 100 C.

Idempotent + non-destructive: only writes the new temp_c1/ tree. If a chunk/layer has not
finished yet it reports the gap and skips that layer (safe to re-run after the jobs land).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

CHUNKS = ["temp_c1a", "temp_c1b", "temp_c1c", "temp_c1d"]
EXPECT_TOTAL = 131  # -20..110 inclusive


def slug_to_temp(slug: str) -> int:
    """vector_id slug -> signed temperature. 'neg20' -> -20, '12' -> 12."""
    return -int(slug[3:]) if slug.startswith("neg") else int(slug)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path(__file__).resolve().parents[2] / "data/axbench")
    ap.add_argument("--layers", nargs="+", type=int, default=[5, 10, 15, 20])
    ap.add_argument("--method-dir", default="diffmean")
    ap.add_argument("--out-name", default="temp_c1")
    args = ap.parse_args()

    import torch  # local import so the script is usable for eval-only checks without torch

    # ---- 1. unified, temperature-ordered concept_index (chunk-independent) ----
    rows = []
    for ds in CHUNKS:
        ci = args.base / ds / "concept_index.parquet"
        if not ci.exists():
            print(f"!! {ds}: concept_index missing -> cannot build unified index; aborting")
            return
        df = pd.read_parquet(ci)
        for _, r in df.iterrows():
            slug = str(r["vector_id"]).split("__")[-1]
            rows.append({
                "chunk": ds,
                "chunk_concept_id": int(r["concept_id"]),
                "temp_c": slug_to_temp(slug),
                "slug": slug,
                "vector_id": f"{args.out_name}__steer__{slug}",
                "concept_group": str(r["concept_group"]),
                "concept_description": r.get("concept_description", ""),
                "perspective": r.get("perspective", "steer"),
                "magnitude_axis": r.get("magnitude_axis", "temperature_c"),
                "target_magnitude_desc": r.get("target_magnitude_desc", f"{slug_to_temp(slug)}C"),
            })
    uidx = pd.DataFrame(rows).sort_values("temp_c").reset_index(drop=True)
    uidx["concept_id"] = uidx.index  # new contiguous, temp-ordered
    if len(uidx) != EXPECT_TOTAL:
        print(f"!! unified index has {len(uidx)} vectors, expected {EXPECT_TOTAL}")
    # (chunk, chunk_concept_id) -> unified concept_id, for re-keying eval rows
    remap = {(r.chunk, r.chunk_concept_id): int(r.concept_id) for r in uidx.itertuples()}

    out_root = args.base / args.out_name
    out_root.mkdir(parents=True, exist_ok=True)
    uidx.drop(columns=["chunk", "chunk_concept_id"]).to_parquet(out_root / "concept_index.parquet", index=False)
    print(f"unified concept_index: {len(uidx)} vectors, temp {uidx.temp_c.min()}..{uidx.temp_c.max()} C -> {out_root/'concept_index.parquet'}")

    # ---- 2. per layer: stitch weights + pool eval ----
    for L in args.layers:
        gaps = []
        # 2a. weights (one row per unified concept_id, temp-ordered)
        W = {}
        B = {}
        d = None
        for ds in CHUNKS:
            wp = args.base / ds / f"{args.method_dir}_l{L}" / "train" / "DiffMean_weight.pt"
            bp = args.base / ds / f"{args.method_dir}_l{L}" / "train" / "DiffMean_bias.pt"
            if not wp.exists():
                gaps.append(f"{ds} L{L}: train weight missing"); continue
            w = torch.load(wp, map_location="cpu")
            b = torch.load(bp, map_location="cpu") if bp.exists() else None
            d = w.shape[1]
            for ccid in range(w.shape[0]):
                ucid = remap.get((ds, ccid))
                if ucid is None:
                    gaps.append(f"{ds} L{L}: chunk cid {ccid} not in remap"); continue
                W[ucid] = w[ccid]
                if b is not None:
                    B[ucid] = b[ccid]
        if d is not None and len(W) == len(uidx):
            wt = torch.stack([W[i] for i in range(len(uidx))])  # (n_temp, d), temp-ordered
            tr = out_root / f"{args.method_dir}_l{L}" / "train"
            tr.mkdir(parents=True, exist_ok=True)
            torch.save(wt, tr / "DiffMean_weight.pt")
            if len(B) == len(uidx):
                torch.save(torch.stack([B[i] for i in range(len(uidx))]), tr / "DiffMean_bias.pt")
            print(f"  L{L}: stitched weight {tuple(wt.shape)} (temp-ordered) -> {tr/'DiffMean_weight.pt'}")
        else:
            print(f"  L{L}: weights NOT stitched (have {len(W)}/{len(uidx)} rows)")

        # 2b. eval pooling
        prs, results = [], []
        for ds in CHUNKS:
            ci = pd.read_parquet(args.base / ds / "concept_index.parquet")
            eval_dir = args.base / ds / f"{args.method_dir}_l{L}" / "eval"
            pr, rj = eval_dir / "per_row_scores.parquet", eval_dir / "results.jsonl"
            if not pr.exists() or not rj.exists():
                gaps.append(f"{ds} L{L}: eval output missing"); continue
            df = pd.read_parquet(pr)
            df["concept_id"] = df["concept_id"].map(lambda c: remap.get((ds, int(c))))
            prs.append(df)
            for line in open(rj):
                line = line.strip()
                if not line: continue
                r = json.loads(line)
                ucid = remap.get((ds, int(r["concept_id"])))
                r["concept_id"] = ucid
                r["temp_c"] = int(uidx.loc[uidx.concept_id == ucid, "temp_c"].iloc[0]) if ucid is not None else None
                r["vector_id"] = uidx.loc[uidx.concept_id == ucid, "vector_id"].iloc[0] if ucid is not None else None
                results.append(r)
        if prs:
            ev = out_root / f"{args.method_dir}_l{L}" / "eval"
            ev.mkdir(parents=True, exist_ok=True)
            merged = pd.concat(prs, ignore_index=True).merge(
                uidx[["concept_id", "temp_c", "vector_id"]], on="concept_id", how="left")
            merged.to_parquet(ev / "per_row_scores.parquet", index=False)
            results.sort(key=lambda r: (r.get("temp_c") if r.get("temp_c") is not None else 1e9))
            (ev / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n")
            print(f"  L{L}: pooled eval {len(results)} vectors, {len(merged)} rows -> {ev}")
        else:
            print(f"  L{L}: NO eval rows pooled (chunks not finished?)")

        if gaps:
            print(f"  L{L} gaps:")
            for g in gaps:
                print("    -", g)


if __name__ == "__main__":
    main()
