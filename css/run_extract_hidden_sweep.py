import subprocess
import yaml

cfg_path = "configs/contrastive_pku.yaml"

with open(cfg_path, "r", encoding="utf-8") as f:
    base_cfg = yaml.safe_load(f)

for layer in range(22, 23):
    cfg = dict(base_cfg)
    cfg_model = dict(cfg["model"])
    cfg_model["layer"] = layer
    cfg["model"] = cfg_model
    cfg["output_dir"] = f"./outputs_llama2/pku_run_layer{layer}"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    cmd = [
        "python",
        "extract_hidden.py",
        "--cfg",
        cfg_path,
        "--split",
        "train",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
