"""model_switch.py -- one switch for running the paper's analyses on another model's vectors.

    RELGEOM_MODEL unset  -> every function returns exactly the paths and layers the paper used
                            (Gemma-2-2B-it, AxBench vectors in data/axbench/).
    RELGEOM_MODEL=<tag>  -> steering vectors are read from data/models/<tag>/axbench/ (written by
                            pipeline/bigmodel/extract.py, mirroring data/axbench/); Gemma layer
                            indices are mapped to the same relative depth; outputs go to
                            <dir>/<tag>/ so the Gemma tables are never overwritten.
                            Tags: olmo31_32b, qwen3_32b.

Only WEIGHTS move. Concept indices, orderings, labels and move definitions keep coming from the
canonical data/axbench/ and data/inputs/ trees -- they are model-agnostic.
"""
import os
from pathlib import Path

from relgeom.models import MODELS
from relgeom.paths import AXBENCH, MODELS as MODELS_DIR

TAG = os.environ.get("RELGEOM_MODEL") or None
GEMMA_N_LAYERS = 26

if TAG is not None and TAG not in MODELS:
    raise SystemExit(f"RELGEOM_MODEL={TAG!r} is not in relgeom/models.py")
if TAG is not None:
    from relgeom.data import need
    need(f"models/{TAG}/*")


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
    return None if TAG is None else str(MODELS_DIR / TAG / "axbench")


def vec_dir(ds, L):
    """Folder holding DiffMean_weight.pt for dataset ds at (already mapped) layer L."""
    base = AXBENCH if TAG is None else MODELS_DIR / TAG / "axbench"
    return base / ds / f"diffmean_l{L}" / "train"


def eval_path(ds, L):
    """results.jsonl of the steering evaluation for ds at (already mapped) layer L."""
    base = AXBENCH if TAG is None else MODELS_DIR / TAG / "axbench"
    return base / ds / f"diffmean_l{L}" / "eval/results.jsonl"


def out_dir(base):
    """Where an analysis writes. Unchanged for Gemma; <base>/<tag>/ otherwise."""
    p = Path(base)
    if TAG is not None:
        p = p / TAG
    p.mkdir(parents=True, exist_ok=True)
    return p


def label():
    return "google/gemma-2-2b-it" if TAG is None else MODELS[TAG]["hf_id"]
