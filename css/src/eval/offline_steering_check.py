import os
import json
import argparse

import torch

from src.utils.config import load_config
from src.models.steering import (
    load_projection_and_subspace,
    compute_pinv_W,
    risk_scores,
    steer_h,
)


def load_activations(base_dir: str, split: str):
    h_path = os.path.join(base_dir, f"{split}_h.pt")
    meta_path = os.path.join(base_dir, f"{split}_meta.json")
    h = torch.load(h_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return h, meta


def main(cfg_path: str, alpha: float, threshold: float):
    cfg = load_config(cfg_path)
    act_dir = os.path.join(cfg.output_dir, "activations")
    h, meta = load_activations(act_dir, "val")
    model, U, device = load_projection_and_subspace(cfg_path, device=None)
    W_pinv = compute_pinv_W(model).to(device)
    h = h.to(device)
    batch_size = 256
    all_before = []
    all_after = []
    n = h.size(0)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            hb = h[i : i + batch_size]
            s_before = risk_scores(hb, model, U)
            hb_new, _ = steer_h(hb, model, U, W_pinv, alpha, threshold)
            s_after = risk_scores(hb_new, model, U)
            all_before.append(s_before.cpu())
            all_after.append(s_after.cpu())
    s_before = torch.cat(all_before, dim=0)
    s_after = torch.cat(all_after, dim=0)
    print("n_samples", s_before.numel())
    print("mean_before", float(s_before.mean().item()))
    print("mean_after", float(s_after.mean().item()))
    print("pct_after_lt_before", float((s_after < s_before).float().mean().item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()
    main(args.cfg, args.alpha, args.threshold)
