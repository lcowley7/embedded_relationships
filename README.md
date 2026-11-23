# How Relations Impose Embedding-Space Structure: code and data

Code and data for the paper *How Relations Impose Embedding-Space Structure*
by Lachlan Cowley (Monash University), Jey Han Lau, Yulia Otmakhova and Eduard Hovy
(The University of Melbourne).

The repository holds the code behind every table and figure in the paper, the generated training text,
and the difference-of-means (DiffMean) steering vectors. The full data, including the steered generations,
judge scores, the 32B models' vectors and the SAE activations, is on Hugging Face:
**[huggingface.co/datasets/octopus123/embedded_relationships](https://huggingface.co/datasets/octopus123/embedded_relationships)**.
Every table and figure regenerates on a CPU from the released data, without an API key or GPU.

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook notebooks/reproduce_from_data.ipynb
```

`notebooks/reproduce_from_data.ipynb` regenerates the paper's tables and figures from `data/` into
`outputs/`, section by section, and shows each next to its paper number. Set `SCOPE = "main"` for the
main-text results only, or `"all"` to include the appendix. Files that are only on Hugging Face are
downloaded into `data/` the first time a script needs them; `python -m relgeom.data` fetches everything
at once (about 0.9 GB).

`notebooks/rerun_pipeline.ipynb` rebuilds the data itself: Gemini generation, AxBench training and
steering on a SLURM cluster, the judge, the 32B models and the SAE activations. It prints every command
and only submits jobs when you set `DRY_RUN = False`.

Every script can also be run directly from any directory, e.g. `python analysis/temperature/moves_rsa.py`.

## Layout

```
relgeom/      shared library: paths, data download, vector loading, ladders, analogy mappings, model switch
analysis/     one script per analysis; each writes CSVs to outputs/tables/
  temperature/  water-word vs numeric temperature: moves and positions RSA
  analogies/    the three metaphors and the professional frames
  steering/     steering efficacy (judge scores, judge-free parse check)
  sensitivity/  the efficacy sweep (drop the weakest vectors and recompute)
  models/       OLMo-3.1-32B and Qwen3-32B beside Gemma-2-2B
  sae/          the GemmaScope feature-overlap analysis
  figures/      every paper figure, from the CSVs
pipeline/     how the data were made
  generate/     Gemini training-text generation (templates in *_templates.py)
  axbench/      conversion to AxBench format, the AxBench patch and config, max_act calibration
  judge/        the Gemini steering judge and its rubrics
  bigmodel/     DiffMean extraction and steering for the 32B models
  sae/          GemmaScope activation extraction
  slurm/        SLURM wrappers
data/         inputs, raw text, vectors, judge scores (see below)
prompts/      every prompt sent to an API model, verbatim (paper Figures 8-14)
docs/         the analogy mappings
notebooks/    the two notebooks above
```

## Paper results and the scripts that make them

| paper | script | output |
|---|---|---|
| Table 1 (water words) | `relgeom/harness.py` (`H2O_TEMP`), `relgeom/ladders.py` | |
| Table 2; Tables 6-7; Figure 15 | `analysis/temperature/moves_rsa.py`; figure `analysis/figures/appendix_figures.py` | `move_direction_stats_*.csv` |
| Table 12 | `analysis/temperature/positions_rsa.py` | `node_rsa_by_layer_unit.csv` |
| Tables 3-4 (mappings) | `relgeom/analogies.py`, `docs/analogy_mappings.md` | |
| Figure 3 | `analysis/analogies/analogy_rsa.py`; figure `analysis/figures/main_rsa_figures.py` | `analogy_rsa.csv` |
| Table 8 | `analysis/analogies/direction_stats.py`, `analysis/analogies/frames_pairs.py` | `analogy_direction_stats.csv` |
| Table 9 | `analysis/analogies/frames_pairs.py` | printed LaTeX rows |
| Figure 5 | `analysis/steering/temperature_judge_scores.py` | `temp_steering_eval.png` |
| Figure 6 | `analysis/steering/temperature_parse_check.py` | `temp_steering_parse.png` |
| Figure 7 | `analysis/steering/analogy_judge_scores.py`; figure `analysis/figures/analogy_steering_figure.py` | `T2_steering_efficacy_per_vector.csv` |
| Table 5 | `analysis/models/cross_model_steering.py` | `steering_comparison.csv` |
| Figure 4; Tables 10, 13 | the temperature and analogy scripts with `RELGEOM_MODEL=olmo31_32b` / `qwen3_32b`, then `analysis/models/cross_model_rsa.py` | `cross_model_summary.csv` |
| Figures 16-17 | `analysis/sensitivity/build_scores.py`, `sweep_temperature.py`, `sweep_analogies.py`; figures `appendix_figures.py` | `temp_sweep.csv`, `analogy_sweep*.csv` |
| Table 11 | the sweeps for each model, then `analysis/sensitivity/cross_model_table.py` | `cross_model_sweep.csv` |
| Figure 18 | `analysis/sae/activation_view.py`; figure `analysis/figures/sae_figure.py` | `act_*.csv` |
| Figures 8-14 | `prompts/` | |

Figures 1 and 2 are illustrations. Tables go to `outputs/tables/` (other models in
`outputs/tables/<tag>/`) and figures to `outputs/figures/`.

**Other models.** Setting `RELGEOM_MODEL=olmo31_32b` or `RELGEOM_MODEL=qwen3_32b` points every analysis at that
model's vectors (`data/models/<tag>/`), at the same relative depth (Gemma layer L maps to round(64L/26)),
and writes to `outputs/tables/<tag>/`. See `relgeom/model_switch.py`.

## Data

Files marked HF are only in the Hugging Face dataset (`relgeom/data.py` downloads them); everything else
is in both.

| path | contents |
|---|---|
| `data/inputs/<ds>_vectors.jsonl` | one row per steering vector (concept specification) |
| `data/raw/<ds>/` | generated training text (`outputs.jsonl`), held-out prompts, generation metadata |
| `data/axbench/<ds>/concept_index.parquet` | concept metadata in vector-row order |
| `data/axbench/<ds>/diffmean_l<L>/train/DiffMean_weight.pt` | Gemma-2-2B-it DiffMean vectors (rows = concepts) |
| `data/axbench/<ds>/diffmean_l<L>/eval/results.jsonl` | judge scores per concept and factor |
| `data/axbench/<ds>/diffmean_l<L>/inference/steering_data.parquet` | HF: steered generations (held-out prompts x factors) |
| `data/axbench/<ds>/diffmean_l<L>/eval/per_row_scores.parquet` | HF: judge scores and judge text per generation and rubric |
| `data/models/<tag>/axbench/<ds>/diffmean_l<L>/` | HF: OLMo-3.1-32B / Qwen3-32B vectors (the nine depth-matched layers) and judge scores |
| `data/sae_acts/` | HF: GemmaScope 16k activations (layers 10, 20) and the feature descriptions |

Datasets: `temp_c1` (131 Celsius levels, -20 to 110) and `temp_f1` (251 Fahrenheit levels, -10 to 240),
each trained in chunks (`temp_c1a..d`, `temp_f1a..f`) and merged; `h2o_v1` (18 water words); and the
analogies `size_imp_v1`, `war_arg_v1`, `food_ideas_v1`, `frames_v1` (94 vectors). Gemma vectors exist at
layers 0-5, 10, 15, 20 for temperature and 5, 10, 15, 20 for the analogies.

## Models and settings

| role | model | settings |
|---|---|---|
| data generation | `gemini-3.5-flash` (Google GenAI API) | `thinking_budget=0`; prompt pools at temperature 1.0 in batches of 24; continuations at temperature 0.7, `max_output_tokens=400` |
| steering judge | `gemini-3.1-flash-lite` | three 0/1/2 rubrics, harmonic mean with zero gate, max over factors |
| steered models | `google/gemma-2-2b-it` (main), `allenai/Olmo-3.1-32B-Instruct`, `Qwen/Qwen3-32B` | |

Steering follows AxBench (upstream commit `41c8332` plus `pipeline/axbench/axbench.patch`): DiffMean
vectors, added at every token, at 15 factors from 0 to 5 times each vector's calibrated `max_act`,
on the first 12 of each dataset's 30 held-out prompts. The 32B models were steered on a reduced grid
(9 factors, the first 6 prompts; `pipeline/bigmodel/run_steer.sh`), and the cross-model table re-scores
Gemma on the same grid.

**Gemini.** Generation and judging used the Gemini API (`GEMINI_API_KEY`). Eventually these models may no longer be
offered, and generation is stochastic. We provided the training text, steered generations and judge outputs
as an original record of our work; the paper's numbers can be reproduced from them.

## Citation

See `CITATION.cff`.

## Licence

Code: MIT (`LICENSE`). Data: CC BY 4.0, with third-party material noted in `LICENSE-DATA.md`.
