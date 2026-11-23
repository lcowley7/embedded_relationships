"""Every path the code uses, relative to the repository root (so scripts run from any directory)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
AXBENCH = DATA / "axbench"      # Gemma-2-2B-it datasets: concept_index, DiffMean vectors, steering + judge outputs
# parts of data/ are downloaded from Hugging Face on first use (relgeom/data.py)
RAW = DATA / "raw"              # generated training text and held-out prompts
INPUTS = DATA / "inputs"        # one row per steering vector (concept spec)
MODELS = DATA / "models"        # OLMo-3.1-32B / Qwen3-32B vectors and judge scores (data/models/<tag>/axbench/...)
SAE_ACTS = DATA / "sae_acts"    # GemmaScope activations (pipeline/sae/extract_sae_acts.py)

OUT = ROOT / "outputs"
TABLES = OUT / "tables"         # every CSV an analysis writes; other models in TABLES/<tag>/
FIGURES = OUT / "figures"       # every paper figure


def ensure(p):
    """Create directory p (and parents) and return it."""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
