# Data licence

Everything under `data/` and `prompts/` written for this project (generated training text, held-out
prompts, vector specifications, steering vectors, steered generations, judge scores, SAE activations)
is released under the Creative Commons Attribution 4.0 International licence (CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/). Please cite the paper if you use it.

Third-party material, under its own terms:

- `data/sae_acts/gemma-2-2b_{10,20}-gemmascope-res-16k.json`: Neuronpedia auto-interpretability
  descriptions of the GemmaScope 16k residual SAE features, as distributed with AxBench.
- `pipeline/axbench/axbench.patch` modifies AxBench (https://github.com/stanfordnlp/axbench); it is
  distributed under AxBench's licence.
- The steering vectors and generations derive from `google/gemma-2-2b-it` (Gemma Terms of Use),
  `allenai/Olmo-3.1-32B-Instruct` and `Qwen/Qwen3-32B` (their respective licences).
