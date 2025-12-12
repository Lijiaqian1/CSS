import os
import json
import argparse
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.config import load_config
from src.models.llama_loader import load_model_and_tokenizer
from data.pku_safe_rlhf import PkuSafeRLHF
from tqdm import tqdm


class HiddenDataset(Dataset):
    def __init__(self, pairs: List[Any]) -> None:
        self.samples = []
        for pid, p in enumerate(pairs):
            safer = p.safer
            if safer == 0:
                safer0, safer1 = 1, 0
            elif safer == 1:
                safer0, safer1 = 0, 1
            else:
                safer0, safer1 = -1, -1
            self.samples.append(
                {
                    "pair_id": pid,
                    "resp_id": 0,
                    "prompt": p.prompt,
                    "response": p.resp0,
                    "safe": int(p.safe0),
                    "safer": safer0,
                }
            )
            self.samples.append(
                {
                    "pair_id": pid,
                    "resp_id": 1,
                    "prompt": p.prompt,
                    "response": p.resp1,
                    "safe": int(p.safe1),
                    "safer": safer1,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def _to_list(x):
    if isinstance(x, torch.Tensor):
        return x.tolist()
    if isinstance(x, list):
        return x
    return [x]


def extract_split(cfg_path: str, split: str):
    cfg = load_config(cfg_path)
    if split == "train":
        split_name = cfg.data.split_train
        max_samples = cfg.data.max_samples_train
    else:
        split_name = cfg.data.split_val
        max_samples = cfg.data.max_samples_val
    dataset = PkuSafeRLHF(
        split=split_name,
        max_samples=max_samples,
    )
    ds = HiddenDataset(list(dataset.iter_pairs()))
    model, tokenizer = load_model_and_tokenizer(
        model_path=cfg.model.path,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layer_idx = cfg.model.layer
    max_length = getattr(cfg.model, "max_length", 512)
    batch_size = getattr(cfg.data, "extract_batch_size", 8)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
    )
    all_h = []
    all_meta: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in tqdm(dl):
            prompts = _to_list(batch["prompt"])
            responses = _to_list(batch["response"])
            texts = [a + "\n\n" + b for a, b in zip(prompts, responses)]
            enc = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True)
            hidden_states = out.hidden_states
            h_layer = hidden_states[layer_idx + 1]
            attn = enc["attention_mask"]
            idx_last = attn.sum(dim=1) - 1
            idx_last = idx_last.view(-1, 1, 1).expand(-1, 1, h_layer.size(-1))
            h_tok = h_layer.gather(1, idx_last).squeeze(1)
            all_h.append(h_tok.cpu())
            pair_ids = _to_list(batch["pair_id"])
            resp_ids = _to_list(batch["resp_id"])
            safes = _to_list(batch["safe"])
            safers_raw = _to_list(batch["safer"])
            for pid, rid, s, sr in zip(pair_ids, resp_ids, safes, safers_raw):
                if sr == -1:
                    safer = None
                else:
                    safer = int(sr)
                all_meta.append(
                    {
                        "pair_id": int(pid),
                        "resp_id": int(rid),
                        "safe": int(s),
                        "safer": safer,
                    }
                )
    all_h_tensor = torch.cat(all_h, dim=0)
    out_dir = os.path.join(cfg.output_dir, "activations")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(all_h_tensor, os.path.join(out_dir, f"{split}_h.pt"))
    with open(os.path.join(out_dir, f"{split}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "val"], default="train")
    args = parser.parse_args()
    extract_split(args.cfg, args.split)
