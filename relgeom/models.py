"""Model and dataset registry for the replication on OLMo-3.1-32B and Qwen3-32B.

Every model is pinned to the Hugging Face commit fetched on 2026-09-15. Headline layers are
depth-matched to Gemma's 5/10/15/20 of 26 (round(L * n_layers / 26)); extract.py writes every
layer regardless.
"""

MODELS = {
    # the port check (gate G1): must reproduce the paper's AxBench vectors
    "gemma2_2b_port": dict(hf_id="google/gemma-2-2b-it",
                           revision="299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
                           n_layers=26, hidden=2304, chat_kwargs={},
                           headline=[5, 10, 15, 20], steer_layer=10),
    "olmo31_32b": dict(hf_id="allenai/Olmo-3.1-32B-Instruct",
                       revision="ac0587e4a7744a551c059d8cd17ba220bc940dae",
                       n_layers=64, hidden=5120, chat_kwargs={},
                       headline=[12, 25, 37, 49], steer_layer=25),
    "qwen3_32b": dict(hf_id="Qwen/Qwen3-32B",
                      revision="9216db5781bf21249d130ec9da846c4624c16137",
                      n_layers=64, hidden=5120, chat_kwargs={"enable_thinking": False},
                      headline=[12, 25, 37, 49], steer_layer=25),
}

EXPERIMENTS = {
    "temperature": ["temp_c1a", "temp_c1b", "temp_c1c", "temp_c1d",
                    "temp_f1a", "temp_f1b", "temp_f1c", "temp_f1d", "temp_f1e", "temp_f1f",
                    "h2o_v1"],
    "analogies": ["size_imp_v1", "war_arg_v1", "food_ideas_v1", "frames_v1"],
}
DATASETS = [d for ds in EXPERIMENTS.values() for d in ds]          # the 15 folders

# gate G1: Gemma re-extraction compared against the paper's AxBench vectors
PORT_DATASETS = ["size_imp_v1", "h2o_v1"]
PORT_LAYERS = [5, 10, 15, 20]
INPUT_FILES = ["train_data.parquet", "heldout_eval.parquet", "concept_index.parquet",
               "metadata.jsonl"]
