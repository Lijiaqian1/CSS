import os
from typing import Tuple

import torch

from src.utils.config import load_config
from src.models.projection_head import ProjectionHead


def load_projection_and_subspace(cfg_path: str, device: torch.device = None):
    cfg = load_config(cfg_path)
    proj_dir = os.path.join(cfg.output_dir, "projection")
    sub_dir = os.path.join(cfg.output_dir, "subspace")
    ckpt_path = os.path.join(proj_dir, "projection_head.pt")
    sub_path = os.path.join(sub_dir, "unsafe_subspace.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sub = torch.load(sub_path, map_location="cpu")
    d_in = int(ckpt["d_in"])
    d_out = int(ckpt["d_out"])
    model = ProjectionHead(d_in, d_out)
    model.load_state_dict(ckpt["state_dict"])
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    U = sub["U"].to(device)
    return model, U, device


def compute_pinv_W(model: ProjectionHead, lam: float = 1e-4):
    W = model.W.weight
    dtype = W.dtype
    device = W.device
    WWt = torch.matmul(W, W.t())
    m = WWt.size(0)
    I = torch.eye(m, device=device, dtype=dtype)
    inv = torch.linalg.inv(WWt + lam * I)
    W_pinv = torch.matmul(W.t(), inv)
    return W_pinv


def risk_scores(h: torch.Tensor, proj_head: ProjectionHead, U: torch.Tensor):
    z = proj_head(h)
    v = torch.matmul(z, U)
    s = torch.norm(v, dim=-1)
    return s


def steering_delta(
    h: torch.Tensor,
    proj_head: ProjectionHead,
    U: torch.Tensor,
    W_pinv: torch.Tensor,
    alpha: float,
    threshold: float = None,
):
    z = proj_head(h)
    v = torch.matmul(z, U)
    s = torch.norm(v, dim=-1)
    proj = torch.matmul(v, U.t())
    proj_t = proj.t()
    delta_t = torch.matmul(W_pinv, proj_t)
    delta = delta_t.t()
    if threshold is not None:
        mask = (s >= threshold).float().unsqueeze(-1)
        delta = delta * mask
    delta = alpha * delta
    return delta, s


def steer_h(
    h: torch.Tensor,
    proj_head: ProjectionHead,
    U: torch.Tensor,
    W_pinv: torch.Tensor,
    alpha: float,
    threshold: float = None,
):
    delta, s = steering_delta(h, proj_head, U, W_pinv, alpha, threshold)
    h_cast = h.to(delta.device).to(delta.dtype)
    h_new = h_cast - delta
    return h_new, s
