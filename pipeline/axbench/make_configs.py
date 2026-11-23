"""Write per-layer DiffMean YAMLs for one or more datasets by templating an existing config.

Each dataset gets diffmean_<ds>_g2-2b_l{5,10,15,20}.yaml pointing at
data/axbench/<ds>. Reuses one L10 config as the template
(same proforma/steer shape: factors, steering_num_of_examples, max_num_of_examples).
The template is pipeline/axbench/config_template.yaml (the h2o_v1 layer-10 config that was run).
Configs are written to ax_bench/axbench/axbench/sweep/frh/, where pipeline/slurm/ looks for them.
The temperature datasets were also trained at layers 0-4: LAYERS="0 1 2 3 4 5 10 15 20".

Usage: python pipeline/axbench/make_configs.py temp_c1a temp_c1b temp_c1c temp_c1d h2o_v1
"""
import os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]   # repository root
SWEEP = REPO / "ax_bench/axbench/axbench/sweep/frh"
# LAYERS env override (default unchanged) so a dataset can add the layer-0
# embedding baseline without editing this file: LAYERS="0 5 10 15 20" python ...
LAYERS = [int(x) for x in os.environ.get("LAYERS", "5 10 15 20").split()]
SRC_DS = "h2o_v1"
base = (Path(__file__).resolve().parent / "config_template.yaml").read_text()
SWEEP.mkdir(parents=True, exist_ok=True)

if len(sys.argv) < 2:
    sys.exit("usage: python make_configs.py <dataset> [<dataset> ...]")

for ds in sys.argv[1:]:
    for L in LAYERS:
        cfg = (base.replace(SRC_DS, ds)
                   .replace("diffmean_l10", f"diffmean_l{L}")
                   .replace("layer: 10", f"layer: {L}")
                   .replace("steering_layers: [10]", f"steering_layers: [{L}]"))
        out = SWEEP / f"diffmean_{ds}_g2-2b_l{L}.yaml"
        out.write_text(cfg)
        print(f"wrote {out.name}")
