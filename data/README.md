---
license: cc-by-4.0
language:
- en
pretty_name: How Relations Impose Embedding-Space Structure (data)
tags:
- interpretability
- steering-vectors
- representational-similarity-analysis
size_categories:
- 100K<n<1M
---

# How Relations Impose Embedding-Space Structure: data

Data for the paper *How Relations Impose Embedding-Space Structure* by Lachlan Cowley (Monash University),
Jey Han Lau, Yulia Otmakhova and Eduard Hovy (The University of Melbourne).

Code: [github.com/lcowley7/embedded_relationships](https://github.com/lcowley7/embedded_relationships).
The code repository carries a core subset of this tree; this dataset is the complete `data/` folder, and the
code downloads it into place (`python -m relgeom.data`).

| path | contents |
|---|---|
| `inputs/<ds>_vectors.jsonl` | one row per steering vector (concept specification) |
| `raw/<ds>/` | generated training text (`outputs.jsonl`), held-out prompts, generation metadata |
| `axbench/<ds>/concept_index.parquet` | concept metadata in vector-row order |
| `axbench/<ds>/diffmean_l<L>/train/DiffMean_weight.pt` | Gemma-2-2B-it DiffMean steering vectors (rows = concepts) |
| `axbench/<ds>/diffmean_l<L>/inference/steering_data.parquet` | steered generations (held-out prompts x factors) |
| `axbench/<ds>/diffmean_l<L>/eval/` | judge scores per generation and rubric, with judge text (`per_row_scores.parquet`), and per concept (`results.jsonl`) |
| `models/<tag>/axbench/<ds>/diffmean_l<L>/` | OLMo-3.1-32B-Instruct / Qwen3-32B vectors (nine depth-matched layers) and judge scores |
| `sae_acts/` | GemmaScope 16k residual SAE activations (Gemma-2-2B, layers 10 and 20) and Neuronpedia feature descriptions |

Datasets: `temp_c1` (131 Celsius levels; trained as chunks `temp_c1a..d`), `temp_f1` (251 Fahrenheit levels;
`temp_f1a..f`), `h2o_v1` (18 water words), and the analogies `size_imp_v1`, `war_arg_v1`, `food_ideas_v1`,
`frames_v1`.

Training text was generated with `gemini-3.5-flash` and steered outputs were judged by `gemini-3.1-flash-lite`.
Licence: CC BY 4.0. Third-party material (model outputs, GemmaScope/Neuronpedia descriptions) is subject to its
providers' terms; see `LICENSE-DATA.md` in the code repository.
