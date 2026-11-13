# How Relations Impose Embedding-Space Structure — code and data

Code and data release for the paper *How Relations Impose Embedding-Space Structure*
by Lachlan Cowley (Monash University), Jey Han Lau, Yulia Otmakhova and Eduard Hovy
(The University of Melbourne). It contains:

- the **prompts** we sent to API models (data generation and the steering judge), in `prompts/`;
- the **code** that generates the training text, builds difference-of-means steering vectors,
  judges steering, and runs every analysis in the paper (move RSA against a permutation null);
- the **generated training text** and the **Gemma-2-2B-it steering vectors** the analyses read,
  so the main results can be recomputed on a CPU without any API or GPU access.

This is research code and is still being tidied. Paths are relative to the repository root:
**run every script from the root**, e.g. `python method_audit_2026-09-16/ladders_18/move_stats.py proposed`.

## Models queried

| role | model | settings |
|---|---|---|
| data generation (prompt pools, continuations) | `gemini-3.5-flash` (Google GenAI API) | `thinking_budget=0`; pools at temperature 1.0 in batches of 24; continuations at temperature 0.7, `max_output_tokens=400` |
| steering judge | `gemini-3.1-flash-lite` | three 0/1/2 rubrics, harmonic mean with zero gate, max over 15 factors |
| steered models | `google/gemma-2-2b-it` (main), `allenai/Olmo-3.1-32B-Instruct`, `Qwen/Qwen3-32B` | |

Set `GEMINI_API_KEY` in the environment for anything that calls Gemini.

## Prompts (`prompts/`)

| file | contents |
|---|---|
| `temperature_generation.md` | numeric-temperature prompt pool + templated continuation frames; water-word pool, positive and shared negative continuation, the 18 members |
| `analogy_generation.md` | the shared positive/negative continuation templates, the four prompt pools, every per-cloud and per-concept slot fill (IMPORTANCE IS SIZE, ARGUMENT IS WAR, IDEAS ARE FOOD, professional frames) |
| `analogy_rendered_examples.md` | one fully rendered positive and negative call per analogy experiment |
| `judge_rubrics.md` | the fluency, instruction-following and concept-relevance rubrics |

These files are extracted from the code. The code sources are `hypothesis_test/temperature_v1.ipynb`
(temperature templates, executed by the generators), `fractal_analysis/extension/wave1_workshop.ipynb`
and `wave1_families.py` (analogy templates and concepts), and `evaluators/gemma4_judge.py` (rubrics).

## Pipeline

1. **Generate training text** (Gemini). Writes `inputs/<ds>_vectors.jsonl` and `data/raw/<ds>/`.
   - `hypothesis_test/generate_temp_numeric_v1.py`: °C prompt pool + `temp_c1a..d` (1 °C steps, −20..110)
   - `hypothesis_test/generate_temp_f1.py`: `temp_f1a..f` (1 °F steps, −10..240); reuses the `temp_c1` pool, no API calls
   - `hypothesis_test/generate_h2o_v1.py`: the 18 water words
   - `fractal_analysis/extension/generate_wave1.py`: `size_imp_v1`, `war_arg_v1`, `food_ideas_v1`, `frames_v1`
   - shared helpers (prompt filter, trigram de-duplication): `pipeline.py`
2. **Convert to AxBench format**: `convert_to_axbench.py` (prompt-paired negatives; see its docstring).
3. **Train / steer with AxBench** (DiffMean, all-token addition). We use upstream
   [AxBench](https://github.com/stanfordnlp/axbench) at commit `41c8332` with the small patch in
   `axbench_patches/axbench.patch`: per-concept paired negatives, truncation logging, lazy LM loading.
   Clone it into `ax_bench/axbench` and `git apply` the patch. Configs come from `hypothesis_test/make_configs.py`,
   which writes them to `ax_bench/axbench/axbench/sweep/frh/` where the SLURM wrappers look for them
   (template: `axbench_patches/diffmean_h2o_v1_g2-2b_l10.yaml`). Steering is evaluated on the first 12 of each
   dataset's 30 held-out prompts (`steering_num_of_examples: 12`). SLURM wrappers are in `slurm/ds_{train,latent,infer}.slurm`.
   The chunked temperature datasets are merged with `evaluators/recombine_temp_{c1,f1}.py`.
4. **Judge steering**: `olmo/gemini_judge.py` (wrapper: `slurm/ds_eval_gemini.slurm`).
5. **Other models** (OLMo-3.1-32B, Qwen3-32B): `bigmodel/extract.py` (DiffMean vectors from the same
   training text), `bigmodel/steer.py`, `bigmodel/models.py`. `bigmodel/run_steer.sh` reproduces the 32B
   steering runs, which used a reduced grid (9 of the 15 factors, the first 6 of the 12 prompts); `steer.py`'s
   defaults are Gemma's full grid. `LT_MODEL=<tag>` (`lt_model.py`) points
   every analysis at that model's vectors, at the same relative depth.

## Paper results → scripts

| paper | script (outputs) |
|---|---|
| Table 1 water-word temperatures | `H2O_TEMP` in `fractal_analysis/analysis_harness.py`; ladders in `method_audit_2026-09-16/ladders_18/ladders.py` (`SPECS["proposed"]`) |
| Table 2, Tables 6–7 (temperature moves, all layers) | `method_audit_2026-09-16/ladders_18/move_stats.py proposed` → `.../out/proposed/` |
| Figure 8 | `method_audit_2026-09-16/ladders_18/build_appendix_spec.py proposed`; moves-only version `fractal_analysis/extension/paper/build_appendix_moves_only.py` |
| Table 12 (temperature positions) | `fractal_analysis/make_node_rsa_tables_sweep.py` |
| Figure 3, Gemma columns of Tables 10/13 (analogy moves/positions) | `fractal_analysis/extension/wave1_rsa.py` → `tables/wave1_rsa.csv`; plot `fractal_analysis/extension/paper/build_paper_figures_moves_only.py` |
| Table 8 (analogy directions) | `fractal_analysis/extension/wave1_direction_stats.py` |
| Table 9 (profession pairs) | `fractal_analysis/extension/frames_diag/make_frames_tables.py` (also `wave1_frames_pooled.py`, `wave1_frames_deep.py`) |
| Figures 5–6 (temperature steering) | `fractal_analysis/make_temp_steering_eval.py`, `fractal_analysis/make_temp_steering_parse.py` |
| Figure 7 (analogy steering) | `fractal_analysis/extension/paper/build_figures.py` then `build_paper_figures_analogy.py` |
| Figures 9–10, Table 11 (efficacy sensitivity sweep) | `filtered_analysis/sweep/src/build_scores.py`; temperature `method_audit_2026-09-16/ladders_18/sweep_spec.py proposed` + `replot_spec.py proposed`; analogies `filtered_analysis/sweep/src/sweep_analogy.py` + `plot_analogy_sweep.py`; across models `compare_model_sweeps.py` |
| Figure 4, Tables 5 and 10 (other models) | `bigmodel/compare_models.py`, `bigmodel/compare_steering.py`, `fractal_analysis/extension/paper/build_paper_models_nowordnet.py` |
| Figure 11 (SAE features) | `fractal_analysis/extension/sae_diag/extract_sae_acts.py` (GPU, GemmaScope 16k), `activation_view.py`, `make_figure_overlap.py` |

Quick check (CPU, about a minute, reproduces Table 2 at layer 10):

```bash
LAYERS=10 python method_audit_2026-09-16/ladders_18/move_stats.py proposed
python fractal_analysis/extension/wave1_rsa.py      # analogy RSA, Figure 3 numbers
```

## Data included

| path | contents |
|---|---|
| `inputs/<ds>_vectors.jsonl` | one row per steering vector (concept spec) |
| `data/raw/<ds>/` | generated training text (`outputs.jsonl`), held-out prompts, generation metadata |
| `data/axbench/<ds>/concept_index.parquet` | concept metadata in vector-row order |
| `data/axbench/<ds>/diffmean_l<L>/train/DiffMean_weight.pt` | Gemma-2-2B-it DiffMean vectors (rows = concepts), for `temp_c1`, `temp_f1`, `h2o_v1` (layers 0–5, 10, 15, 20) and the four analogy datasets (5, 10, 15, 20) |
| `data/axbench/<ds>/diffmean_l<L>/eval/results.jsonl` | per-concept judge scores by steering factor |

The 32B-model vectors and the GemmaScope activations are not included because of their size.
The scripts that produce them are listed above.

## Environment

Python 3.10. See `requirements.txt`. The analyses need only numpy, pandas, scipy, torch (to read
`.pt` files) and matplotlib. Generation needs `google-genai`; vector extraction needs `transformers` and a GPU.
