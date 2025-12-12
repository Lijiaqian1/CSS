import os
import json
import argparse
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.config import load_config
from src.models.projection_head import ProjectionHead


class HiddenOnlyDataset(Dataset):
    def __init__(self, h_tensor: torch.Tensor) -> None:
        self.h = h_tensor

    def __len__(self) -> int:
        return self.h.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.h[idx]


def load_activations(base_dir: str, split: str):
    h_path = os.path.join(base_dir, f"{split}_h.pt")
    meta_path = os.path.join(base_dir, f"{split}_meta.json")
    h = torch.load(h_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return h, meta


def build_safe_unsafe_pairs(meta: List[Dict[str, Any]]):
    by_pair: Dict[int, List[Dict[str, Any]]] = {}
    for idx, m in enumerate(meta):
        pid = int(m["pair_id"])
        if pid not in by_pair:
            by_pair[pid] = []
        by_pair[pid].append(
            {
                "idx": idx,
                "safe": int(m["safe"]),
            }
        )
    unsafe_indices = []
    safe_indices = []
    for pid, items in by_pair.items():
        if len(items) < 2:
            continue
        safe_list = [x for x in items if x["safe"] == 1]
        unsafe_list = [x for x in items if x["safe"] == 0]
        if len(safe_list) == 0 or len(unsafe_list) == 0:
            continue
        unsafe_indices.append(unsafe_list[0]["idx"])
        safe_indices.append(safe_list[0]["idx"])
    return torch.tensor(unsafe_indices, dtype=torch.long), torch.tensor(safe_indices, dtype=torch.long)


def extract_subspace(cfg_path: str):
    cfg = load_config(cfg_path)
    act_dir = os.path.join(cfg.output_dir, "activations")
    proj_dir = os.path.join(cfg.output_dir, "projection")
    h, meta = load_activations(act_dir, "train")
    unsafe_idx, safe_idx = build_safe_unsafe_pairs(meta)
    d_in = h.size(1)
    ckpt_path = os.path.join(proj_dir, "projection_head.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    d_out = ckpt["d_out"]
    model = ProjectionHead(d_in, d_out)
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    ds = HiddenOnlyDataset(h)
    batch_size = getattr(cfg.contrastive, "batch_size", 256)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    all_z = []
    with torch.no_grad():
        for batch in dl:
            batch = batch.to(device)
            z = model(batch)
            all_z.append(z.cpu())
    all_z = torch.cat(all_z, dim=0)
    z_unsafe = all_z[unsafe_idx]
    z_safe = all_z[safe_idx]
    D = z_unsafe - z_safe
    D = D - D.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    k = cfg.subspace.k
    V = Vh.transpose(0, 1)
    U_sub = V[:, :k]
    out_dir = os.path.join(cfg.output_dir, "subspace")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {
            "U": U_sub,
            "singular": S[:k],
            "num_pairs": int(D.size(0)),
            "cfg_path": cfg_path,
        },
        os.path.join(out_dir, "unsafe_subspace.pt"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()
    extract_subspace(args.cfg)
