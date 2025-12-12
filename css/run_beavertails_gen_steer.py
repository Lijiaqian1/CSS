import argparse
import json
import os
import random
import torch
from datasets import load_dataset
from transformers.utils import logging as hf_logging
from tqdm import tqdm

hf_logging.set_verbosity_error()

from src.utils.config import load_config
from src.models.llama_loader import load_model_and_tokenizer
from src.models.steering import (
    load_projection_and_subspace,
    compute_pinv_W,
    steer_h,
)


def _get(obj, key, default=None):
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict) and key in obj:
        return obj[key]
    return default


def make_gen_hook(proj_head, U, W_pinv, valmax, max_new, threshold, direction):
    state = {"step": 0}

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h, *rest = output
        else:
            h, rest = output, []
        if not isinstance(h, torch.Tensor) or h.dim() != 3:
            return output

        if h.size(1) > 1:
            alpha = valmax
        else:
            state["step"] += 1
            step = state["step"]
            if max_new <= 1:
                alpha = valmax
            elif step > max_new:
                alpha = 0.0
            else:
                alpha = valmax * (1 - (step - 1) / (max_new - 1))

        alpha = alpha * direction
        if alpha == 0.0:
            return output

        dev = h.device
        h_last = h[:, -1, :]
        ph = proj_head.to(dev)
        U_ = U.to(dev)
        Wp = W_pinv.to(dev)

        with torch.no_grad():
            h_new, _ = steer_h(
                h_last,
                ph,
                U_,
                Wp,
                alpha=alpha,
                threshold=threshold,
            )

        h_mod = h.clone()
        h_mod[:, -1, :] = h_new.to(h.dtype)

        if isinstance(output, tuple):
            return (h_mod, *rest)
        else:
            return h_mod

    return hook


def load_gen_samples(cfg):
    hf_name = _get(cfg.data, "hf_name")
    split = _get(cfg.data, "split_eval")
    max_per = int(_get(cfg.data, "max_eval_per_category"))
    cats = _get(cfg.data, "categories")
    seed = int(_get(cfg, "seed", 42))

    ds = load_dataset(hf_name, split=split)
    rng = random.Random(seed)

    out = {}
    print(f"[LOAD] split={split}, total={len(ds)}")

    all_keys = set()
    for i in range(min(len(ds), 2000)):
        row = ds[i]
        for k in row["category"].keys():
            all_keys.add(k)
    print("[LOAD] available category keys:")
    for k in sorted(all_keys):
        print("  -", k)

    for cat in cats:
        name = cat["name"]
        key = cat["key"]

        idxs = []
        for i in range(len(ds)):
            row = ds[i]
            if bool(row["category"].get(key, False)):
                idxs.append(i)

        rng.shuffle(idxs)
        idxs = idxs[:max_per]

        samples = []
        for idx in idxs:
            r = ds[idx]
            samples.append(
                {
                    "ds_idx": idx,
                    "prompt": r["prompt"],
                    "is_safe": bool(r["is_safe"]),
                }
            )

        out[name] = samples
        print(f"[LOAD] category={name}, key={key}, num_samples={len(samples)}")

    return out


def eval_beavertails_gen(cfg_path, cfg, valmax, direction, max_new, threshold, out_path, n_print):
    model_path = cfg.model.path
    layer_idx = int(cfg.model.layer)
    max_len = int(_get(cfg.model, "max_length", 512))

    print(f"[MODEL] path={model_path}, layer={layer_idx}")
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"[MODEL] device={device}")

    proj_head, U, _ = load_projection_and_subspace(cfg_path, device=device)
    W_pinv = compute_pinv_W(proj_head).to(device)
    print("[STEER] projection/subspace loaded")

    block = model.model.layers[layer_idx]

    samples_by_cat = load_gen_samples(cfg)

    results = []

    for cat_name, samples in samples_by_cat.items():
        print(f"[EVAL] category={cat_name}, num_samples={len(samples)}")
        printed = 0

        for ex in tqdm(samples, desc=f"[EVAL] {cat_name}"):
            prompt = ex["prompt"]
            y_safe = ex["is_safe"]
            ds_idx = ex["ds_idx"]

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
            ).to(device)
            in_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                out_base = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,
                )
            gen_ids_base = out_base[0][in_len:]
            text_base = tokenizer.decode(gen_ids_base, skip_special_tokens=True)

            hook = make_gen_hook(
                proj_head,
                U,
                W_pinv,
                valmax=valmax,
                max_new=max_new,
                threshold=threshold,
                direction=direction,
            )
            handle = block.register_forward_hook(hook)
            with torch.no_grad():
                out_steer = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,
                )
            handle.remove()
            gen_ids_steer = out_steer[0][in_len:]
            text_steer = tokenizer.decode(gen_ids_steer, skip_special_tokens=True)

            rec = {
                "category": cat_name,
                "ds_idx": ds_idx,
                "prompt": prompt,
                "is_safe_label": y_safe,
                "base_answer": text_base,
                "steer_answer": text_steer,
            }
            results.append(rec)

            if printed < n_print:
                printed += 1
                print("=" * 80)
                print(f"[SAMPLE] category={cat_name}, ds_idx={ds_idx}, is_safe={y_safe}")
                print("[PROMPT]")
                print(prompt)
                print("\n[BASE ANSWER]")
                print(text_base)
                print("\n[STEER ANSWER]")
                print(text_steer)
                print()

    if out_path is None or out_path == "":
        out_path = f"beavertails_gen_valmax{valmax}_dir{direction}.json"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] saved {len(results)} records to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--valmax", type=float, default=100.0)
    parser.add_argument("--direction", type=float, default=-1.0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--print_per_category", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    eval_beavertails_gen(
        args.cfg,
        cfg,
        args.valmax,
        args.direction,
        args.max_new_tokens,
        args.threshold,
        args.out,
        args.print_per_category,
    )
