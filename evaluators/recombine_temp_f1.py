"""Stitch the temp_f1 chunks (a..f) into one temperature-ordered 1°F sweep (-10..240°F, 251 levels).

Mirror of recombine_temp_c1.py for the dense FAHRENHEIT twin. Each chunk restarts concept_id at 0 with
a chunk-specific vector_id prefix (temp_f1a__steer__<t> .. temp_f1f__steer__<t>); logically it is ONE
sweep. Reassembles WITHOUT touching the chunk dirs, writing data/axbench/temp_f1/:
  diffmean_lN/train/DiffMean_weight.pt   stitched, F-ordered (n_f, d)
  diffmean_lN/eval/{results.jsonl,per_row_scores.parquet}  pooled, re-keyed
  concept_index.parquet                  unified, F-ordered, with temp_f + temp_c (c_twin) columns

Headline confound test: load train/DiffMean_weight.pt (F-ordered), plot ‖v_{F+1}-v_F‖ vs F -> spikes
expected at 32°F & 212°F (PHYSICAL boundaries), NOT at round 0/50/100/150/200°F.

Idempotent + non-destructive; skips a layer if a chunk hasn't finished (safe to re-run).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

CHUNKS = ["temp_f1a", "temp_f1b", "temp_f1c", "temp_f1d", "temp_f1e", "temp_f1f"]
EXPECT_TOTAL = 251  # -10..240 inclusive

def slug_to_f(slug: str) -> int:
    return -int(slug[3:]) if slug.startswith("neg") else int(slug)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path("data/axbench"))
    ap.add_argument("--layers", nargs="+", type=int, default=[5, 10, 15, 20])
    ap.add_argument("--method-dir", default="diffmean")
    ap.add_argument("--out-name", default="temp_f1")
    args = ap.parse_args()
    import torch

    # ---- 1. unified, F-ordered concept_index ----
    rows = []
    for ds in CHUNKS:
        ci = args.base / ds / "concept_index.parquet"
        if not ci.exists():
            print(f"!! {ds}: concept_index missing -> aborting"); return
        df = pd.read_parquet(ci)
        for _, r in df.iterrows():
            slug = str(r["vector_id"]).split("__")[-1]; f = slug_to_f(slug)
            rows.append({"chunk": ds, "chunk_concept_id": int(r["concept_id"]),
                         "temp_f": f, "temp_c": round((f - 32) * 5 / 9), "slug": slug,
                         "vector_id": f"{args.out_name}__steer__{slug}",
                         "concept_group": str(r["concept_group"]),
                         "concept_description": r.get("concept_description", ""),
                         "perspective": r.get("perspective", "steer"),
                         "magnitude_axis": r.get("magnitude_axis", "temperature_f"),
                         "target_magnitude_desc": r.get("target_magnitude_desc", f"{f}F")})
    uidx = pd.DataFrame(rows).sort_values("temp_f").reset_index(drop=True)
    uidx["concept_id"] = uidx.index
    if len(uidx) != EXPECT_TOTAL:
        print(f"!! unified index has {len(uidx)} vectors, expected {EXPECT_TOTAL}")
    remap = {(r.chunk, r.chunk_concept_id): int(r.concept_id) for r in uidx.itertuples()}

    out_root = args.base / args.out_name; out_root.mkdir(parents=True, exist_ok=True)
    uidx.drop(columns=["chunk", "chunk_concept_id"]).to_parquet(out_root / "concept_index.parquet", index=False)
    print(f"unified concept_index: {len(uidx)} vectors, {uidx.temp_f.min()}..{uidx.temp_f.max()}°F -> {out_root/'concept_index.parquet'}")

    # ---- 2. per layer: stitch weights + pool eval ----
    for L in args.layers:
        gaps = []; W = {}; B = {}; d = None
        for ds in CHUNKS:
            wp = args.base / ds / f"{args.method_dir}_l{L}" / "train" / "DiffMean_weight.pt"
            bp = args.base / ds / f"{args.method_dir}_l{L}" / "train" / "DiffMean_bias.pt"
            if not wp.exists():
                gaps.append(f"{ds} L{L}: train weight missing"); continue
            w = torch.load(wp, map_location="cpu"); b = torch.load(bp, map_location="cpu") if bp.exists() else None
            d = w.shape[1]
            for ccid in range(w.shape[0]):
                ucid = remap.get((ds, ccid))
                if ucid is None: gaps.append(f"{ds} L{L}: cid {ccid} unmapped"); continue
                W[ucid] = w[ccid]
                if b is not None: B[ucid] = b[ccid]
        if d is not None and len(W) == len(uidx):
            wt = torch.stack([W[i] for i in range(len(uidx))])
            tr = out_root / f"{args.method_dir}_l{L}" / "train"; tr.mkdir(parents=True, exist_ok=True)
            torch.save(wt, tr / "DiffMean_weight.pt")
            if len(B) == len(uidx): torch.save(torch.stack([B[i] for i in range(len(uidx))]), tr / "DiffMean_bias.pt")
            print(f"  L{L}: stitched weight {tuple(wt.shape)} (F-ordered) -> {tr/'DiffMean_weight.pt'}")
        else:
            print(f"  L{L}: weights NOT stitched (have {len(W)}/{len(uidx)} rows)")

        prs, results = [], []
        for ds in CHUNKS:
            eval_dir = args.base / ds / f"{args.method_dir}_l{L}" / "eval"
            pr, rj = eval_dir / "per_row_scores.parquet", eval_dir / "results.jsonl"
            if not pr.exists() or not rj.exists():
                gaps.append(f"{ds} L{L}: eval output missing"); continue
            df = pd.read_parquet(pr); df["concept_id"] = df["concept_id"].map(lambda c: remap.get((ds, int(c)))); prs.append(df)
            for line in open(rj):
                line = line.strip()
                if not line: continue
                r = json.loads(line); ucid = remap.get((ds, int(r["concept_id"]))); r["concept_id"] = ucid
                r["temp_f"] = int(uidx.loc[uidx.concept_id == ucid, "temp_f"].iloc[0]) if ucid is not None else None
                r["vector_id"] = uidx.loc[uidx.concept_id == ucid, "vector_id"].iloc[0] if ucid is not None else None
                results.append(r)
        if prs:
            ev = out_root / f"{args.method_dir}_l{L}" / "eval"; ev.mkdir(parents=True, exist_ok=True)
            merged = pd.concat(prs, ignore_index=True).merge(uidx[["concept_id", "temp_f", "vector_id"]], on="concept_id", how="left")
            merged.to_parquet(ev / "per_row_scores.parquet", index=False)
            results.sort(key=lambda r: (r.get("temp_f") if r.get("temp_f") is not None else 1e9))
            (ev / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n")
            print(f"  L{L}: pooled eval {len(results)} vectors -> {ev}")
        if gaps:
            print(f"  L{L} gaps:"); [print("    -", g) for g in gaps]

if __name__ == "__main__":
    main()
