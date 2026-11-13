"""lt_model.py -- one switch for running the paper's analyses on another model's vectors.

    LT_MODEL unset   -> every function returns exactly the paths and layers the paper used
                        (Gemma-2-2B-it, AxBench vectors). Nothing changes.
    LT_MODEL=<tag>   -> steering vectors are read from models/<tag>/axbench/
                        (written by bigmodel/extract.py, mirroring data/axbench/); Gemma layer
                        indices are mapped to the same relative depth; outputs go to <dir>/<tag>/
                        so the Gemma tables and figures are never overwritten.
    LT_SPAN=content  -> (with LT_MODEL) use the prompt..response span (train_content/) instead of
                        the AxBench span (train/), as a robustness check.

Only WEIGHTS move. Concept indices, orderings, labels and move definitions keep coming from the
canonical data/axbench/ and inputs/ trees -- they are model-agnostic, and several analyses read
metadata of datasets that have no vectors in the model trees (e.g. the h2o ladder specs).
"""
import os
import sys
from pathlib import Path

REPO = Path(".")
TAG = os.environ.get("LT_MODEL") or None
SPAN = os.environ.get("LT_SPAN", "axb")
GEMMA_N_LAYERS = 26

if TAG is not None:
    sys.path.insert(0, str(REPO / "bigmodel"))
    from models import MODELS  # noqa: E402
    if TAG not in MODELS:
        raise SystemExit(f"LT_MODEL={TAG!r} is not in bigmodel/models.py")
if SPAN not in ("axb", "content"):
    raise SystemExit(f"LT_SPAN={SPAN!r}: use 'axb' or 'content'")
if TAG is None and SPAN != "axb":
    raise SystemExit("LT_SPAN=content needs LT_MODEL (the paper's Gemma vectors are AxBench-span only)")


def n_layers():
    return GEMMA_N_LAYERS if TAG is None else MODELS[TAG]["n_layers"]


def layer(L):
    """A Gemma layer index -> this model's layer at the same relative depth (identity for Gemma)."""
    return L if TAG is None else round(L * n_layers() / GEMMA_N_LAYERS)


def layers(Ls):
    out = [layer(L) for L in Ls]
    assert len(set(out)) == len(out), f"layer map collides: {Ls} -> {out}"
    return out


def weight_base():
    """Root to read DiffMean weights from; None means the canonical data/axbench tree."""
    return None if TAG is None else str(REPO / "models" / TAG / "axbench")


def weight_subdir():
    return "train" if SPAN == "axb" else "train_content"


def vec_dir(ds, L):
    """Folder holding DiffMean_weight.pt for dataset ds at (already mapped) layer L."""
    if TAG is None:
        return REPO / "data/axbench" / ds / f"diffmean_l{L}" / "train"
    return REPO / "models" / TAG / "axbench" / ds / f"diffmean_l{L}" / weight_subdir()


def eval_path(ds, L):
    """results.jsonl of the steering evaluation for ds at (already mapped) layer L.

    Only the AxBench span was steered, so this is an error under LT_SPAN=content: scoring
    content-span geometry by axb-span steering would mix two vector sets.
    """
    if SPAN != "axb":
        raise SystemExit("steering was evaluated on the AxBench span only; unset LT_SPAN")
    if TAG is None:
        return REPO / "data/axbench" / ds / f"diffmean_l{L}" / "eval/results.jsonl"
    return REPO / "models" / TAG / "axbench" / ds / f"diffmean_l{L}" / "eval/results.jsonl"


def out_dir(base):
    """Where an analysis writes. Unchanged for Gemma; <base>/<tag>[_content]/ otherwise."""
    p = Path(base)
    if TAG is not None:
        p = p / (TAG if SPAN == "axb" else f"{TAG}_content")
    p.mkdir(parents=True, exist_ok=True)
    return p


def label():
    return "google/gemma-2-2b-it" if TAG is None else MODELS[TAG]["hf_id"]
