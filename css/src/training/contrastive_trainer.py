import os
import json
import argparse
import random
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

from src.utils.config import load_config
from src.models.projection_head import ProjectionHead
from src.training.utils_loss import supcon_loss, ortho_reg


class ActivationDataset(Dataset):
    def __init__(self, h_tensor: torch.Tensor, meta: List[Dict[str, Any]]) -> None:
        self.h = h_tensor
        self.meta = meta

    def __len__(self) -> int:
        return self.h.size(0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = {
            "h": self.h[idx],
            "label": int(self.meta[idx]["safe"]),
            "pair_id": int(self.meta[idx]["pair_id"]),
            "safer": self.meta[idx]["safer"],
        }
        return x


def collate_activations(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    h = torch.stack([b["h"] for b in batch], dim=0)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    pair_ids = torch.tensor([b["pair_id"] for b in batch], dtype=torch.long)
    safer = []
    for b in batch:
        v = b["safer"]
        if v is None:
            safer.append(-1)
        else:
            safer.append(int(v))
    safer = torch.tensor(safer, dtype=torch.long)
    return {"h": h, "labels": labels, "pair_ids": pair_ids, "safer": safer}


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def load_activations(base_dir: str, split: str) -> ActivationDataset:
    h_path = os.path.join(base_dir, f"{split}_h.pt")
    meta_path = os.path.join(base_dir, f"{split}_meta.json")
    h = torch.load(h_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return ActivationDataset(h, meta)


def train_contrastive(cfg_path: str):
    cfg = load_config(cfg_path)
    set_seed(cfg.seed)
    act_dir = os.path.join(cfg.output_dir, "activations")
    train_ds = load_activations(act_dir, "train")
    batch_size = int(cfg.contrastive.batch_size)
    dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_activations,
    )
    d_in = train_ds.h.size(1)
    d_out = int(cfg.contrastive.proj_dim)
    model = ProjectionHead(d_in, d_out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    lr = float(cfg.contrastive.lr)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    temperature = float(cfg.contrastive.temperature)
    lambda_ortho = float(getattr(cfg.contrastive, "ortho_reg", 0.0) or 0.0)
    epochs = int(getattr(cfg.contrastive, "epochs", 20) or 20)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_steps = 0
        for batch in dl:
            h = batch["h"].to(device)
            labels = batch["labels"].to(device)
            z = model(h)
            loss_sup = supcon_loss(z, labels, temperature)
            W = model.W.weight
            loss_ortho = lambda_ortho * ortho_reg(W)
            loss = loss_sup + loss_ortho
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_steps += 1
        avg_loss = total_loss / max(1, n_steps)
        print(f"epoch {epoch+1}/{epochs} loss {avg_loss:.4f}")
    out_dir = os.path.join(cfg.output_dir, "projection")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "d_in": d_in,
            "d_out": d_out,
            "cfg_path": cfg_path,
        },
        os.path.join(out_dir, "projection_head.pt"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()
    train_contrastive(args.cfg)
