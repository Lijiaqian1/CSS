import os
import json
import argparse

import torch

from src.utils.config import load_config
from src.models.steering import load_projection_and_subspace, risk_scores


def load_activations(base_dir: str, split: str):
    h_path = os.path.join(base_dir, f"{split}_h.pt")
    meta_path = os.path.join(base_dir, f"{split}_meta.json")
    h = torch.load(h_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    labels_safe = torch.tensor([int(m["safe"]) for m in meta], dtype=torch.long)
    labels_unsafe = 1 - labels_safe
    return h, labels_unsafe


def compute_auc(scores: torch.Tensor, labels: torch.Tensor):
    scores_np = scores.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    import numpy as np
    pos = scores_np[labels_np == 1]
    neg = scores_np[labels_np == 0]
    n_pos = pos.shape[0]
    n_neg = neg.shape[0]
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    all_scores = np.concatenate([pos, neg], axis=0)
    order = all_scores.argsort()
    ranks = order.argsort() + 1
    ranks_pos = ranks[:n_pos]
    auc = (ranks_pos.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def eval_risk(cfg_path: str):
    cfg = load_config(cfg_path)
    act_dir = os.path.join(cfg.output_dir, "activations")
    h, labels_unsafe = load_activations(act_dir, "val")
    model, U, device = load_projection_and_subspace(cfg_path, device=None)
    h = h.to(device)
    batch_size = 256
    all_scores = []
    n = h.size(0)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            hb = h[i : i + batch_size]
            s = risk_scores(hb, model, U)
            all_scores.append(s.cpu())
    scores = torch.cat(all_scores, dim=0)
    auc = compute_auc(scores, labels_unsafe)
    thr = float(scores.median().item())
    preds_unsafe = (scores >= thr).long()
    acc = (preds_unsafe == labels_unsafe).float().mean().item()
    print("n_samples", int(scores.numel()))
    print("auc_unsafe_high_score", auc)
    print("median_threshold", thr)
    print("acc_if_predict_unsafe_when_score_high", acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()
    eval_risk(args.cfg)
