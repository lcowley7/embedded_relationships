"""The full data tree is on Hugging Face; the git repository carries only the core.

    https://huggingface.co/datasets/octopus123/embedded_relationships

In git (enough for the Gemma geometry: Table 2, Figures 3 and 7, the sweeps):
    data/inputs/, data/raw/, and in data/axbench/ the concept indices, DiffMean vectors and
    per-concept judge results.
Only on Hugging Face, downloaded on first use:
    steered generations, per-row judge scores, the OLMo/Qwen 32B vectors and scores, and the
    SAE activations (see HF_ONLY).

    python -m relgeom.data          # download everything (about 0.9 GB) into data/
"""
from pathlib import Path

from relgeom.paths import DATA

HF_REPO = "octopus123/embedded_relationships"
HF_ONLY = ["axbench/*/diffmean_l*/inference/steering_data.parquet",
           "axbench/*/diffmean_l*/eval/per_row_scores.parquet",
           "models/*",
           "sae_acts/*"]


def fetch(patterns=None):
    """Download files from the Hugging Face dataset into data/ (all of them if patterns is None)."""
    from huggingface_hub import snapshot_download
    print(f"downloading {patterns or 'everything'} from huggingface.co/datasets/{HF_REPO} ...", flush=True)
    snapshot_download(HF_REPO, repo_type="dataset", local_dir=DATA, allow_patterns=patterns)


def need(*patterns):
    """Make sure files matching each pattern (relative to data/) exist, downloading them if not."""
    missing = [p for p in patterns if next(Path(DATA).glob(p), None) is None]
    if missing:
        fetch(missing)


if __name__ == "__main__":
    fetch()
