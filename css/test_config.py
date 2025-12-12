from src.utils.config import load_config

cfg = load_config("configs/contrastive_pku.yaml")
print(cfg.seed)
print(cfg.model.path)
print(cfg.contrastive.proj_dim)
