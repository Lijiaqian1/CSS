import argparse
import json
import os
import random
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
from src.utils.config import load_config
from src.models.llama_loader import load_model_and_tokenizer
from src.models.steering import (
    load_projection_and_subspace,
    compute_pinv_W,
    steer_h,
)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict) and key in obj:
        return obj[key]
    return default


def make_gen_hook(proj_head, U, W_pinv, valmax, max_new, threshold, direction):
    state = {"step": 0, "call_id": 0}

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h, *rest = output
        else:
            h, rest = output, []

        if not isinstance(h, torch.Tensor) or h.dim() != 3:
            return output

        state["call_id"] += 1
        if h.size(1) > 1:
            alpha_base = valmax
        else:
            state["step"] += 1
            step = state["step"]
            if max_new <= 1:
                alpha_base = valmax
            elif step > max_new:
                alpha_base = 0.0
            else:
                alpha_base = valmax * (1 - (step - 1) / (max_new - 1))

        alpha = alpha_base * direction
        if alpha == 0.0:
            return output

        dev = h.device
        h_last = h[:, -1, :]

        proj_head_local = proj_head.to(dev)
        U_local = U.to(dev)
        W_pinv_local = W_pinv.to(dev)

        with torch.no_grad():
            h_new_last, _ = steer_h(
                h_last,
                proj_head_local,
                U_local,
                W_pinv_local,
                alpha=alpha,
                threshold=threshold,
            )

        h_mod = h.clone()
        h_mod[:, -1, :] = h_new_last.to(h.dtype)

        if isinstance(output, tuple):
            return (h_mod, *rest)
        else:
            return h_mod

    return hook


def build_catqa_samples(cfg, total_samples: int, subcats_from_cli: List[str], seed: int):
    hf_name = _get(cfg.data, "hf_name", "declare-lab/CategoricalHarmfulQA")
    split = _get(cfg.data, "split", "train")
    ds = load_dataset(hf_name, split=split)

    all_subcats = sorted({row["Subcategory"] for row in ds})

    if subcats_from_cli and not (len(subcats_from_cli) == 1 and subcats_from_cli[0].lower() == "all"):
        selected_subcats = subcats_from_cli
    else:
        cfg_subcats = _get(cfg.data, "subcategories", ["all"])
        if isinstance(cfg_subcats, list) and len(cfg_subcats) == 1 and str(cfg_subcats[0]).lower() == "all":
            selected_subcats = all_subcats
        else:
            selected_subcats = cfg_subcats

    selected_subcats = [s for s in selected_subcats if s in all_subcats]
    if not selected_subcats:
        raise ValueError("No valid subcategories selected.")

    rng = random.Random(seed)
    buckets: Dict[str, List[int]] = {s: [] for s in selected_subcats}

    for idx, row in enumerate(ds):
        sub = row["Subcategory"]
        if sub in buckets:
            buckets[sub].append(idx)

    k = total_samples
    n_sub = len(selected_subcats)
    base_per = k // n_sub
    extra = k % n_sub

    samples: List[Dict[str, Any]] = []
    for i, sub in enumerate(selected_subcats):
        idxs = buckets[sub]
        if not idxs:
            continue
        need = base_per + (1 if i < extra else 0)
        need = min(need, len(idxs))
        chosen = rng.sample(idxs, need)
        for ds_idx in chosen:
            row = ds[ds_idx]
            samples.append(
                {
                    "ds_idx": ds_idx,
                    "Category": row["Category"],
                    "Subcategory": row["Subcategory"],
                    "Question": row["Question"],
                }
            )

    return samples, all_subcats, selected_subcats


def eval_catqa_gen(
    cfg_path: str,
    cfg,
    valmax: float,
    direction: float,
    max_new: int,
    threshold: float,
    total_samples: int,
    subcats_from_cli: List[str],
    out_path: str,
    seed: int,
    print_per_subcat: int,
):
    random.seed(seed)

    model_path = cfg.model.path
    layer_idx = int(cfg.model.layer)
    max_len = int(_get(cfg.model, "max_length", 512))

    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device

    proj_head, U, _ = load_projection_and_subspace(cfg_path, device=device)
    W_pinv = compute_pinv_W(proj_head).to(device)

    block = model.model.layers[layer_idx]

    samples, all_subcats, selected_subcats = build_catqa_samples(
        cfg=cfg,
        total_samples=total_samples,
        subcats_from_cli=subcats_from_cli,
        seed=seed,
    )

    by_subcat: Dict[str, List[Dict[str, Any]]] = {}
    for ex in samples:
        sub = ex["Subcategory"]
        by_subcat.setdefault(sub, []).append(ex)

    results: List[Dict[str, Any]] = []

    for sub in selected_subcats:
        sub_samples = by_subcat.get(sub, [])
        if not sub_samples:
            continue
        printed = 0

        for ex in tqdm(sub_samples, desc=f"[EVAL] {sub}"):
            q = ex["Question"]
            cat = ex["Category"]
            ds_idx = ex["ds_idx"]

            prompt = q

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
                proj_head=proj_head,
                U=U,
                W_pinv=W_pinv,
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
                "ds_idx": ds_idx,
                "category": cat,
                "subcategory": sub,
                "question": q,
                "base_answer": text_base,
                "steer_answer": text_steer,
            }
            results.append(rec)

            if printed < print_per_subcat:
                printed += 1
                print("=" * 80)
                print(f"[SAMPLE] subcategory={sub}, ds_idx={ds_idx}")
                print("[QUESTION]")
                print(q)
                print("\n[BASE ANSWER]")
                print(text_base)
                print("\n[STEER ANSWER]")
                print(text_steer)
                print()

    if not out_path:
        out_path = f"catqa_gen_valmax{valmax}_dir{direction}_N{len(results)}.json"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Total selected subcategories: {len(selected_subcats)}")
    print(f"Total generated samples: {len(results)}")
    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--valmax", type=float, default=50.0)
    parser.add_argument("--direction", type=float, default=-1.0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--total_samples", type=int, default=None)
    parser.add_argument("--subcats", nargs="*", default=None)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_per_subcat", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    total_samples = args.total_samples
    if total_samples is None:
        total_samples = int(_get(cfg.data, "total_samples", 200))

    subcats_from_cli = args.subcats if args.subcats is not None else []

    eval_catqa_gen(
        cfg_path=args.cfg,
        cfg=cfg,
        valmax=args.valmax,
        direction=args.direction,
        max_new=args.max_new_tokens,
        threshold=args.threshold,
        total_samples=total_samples,
        subcats_from_cli=subcats_from_cli,
        out_path=args.out,
        seed=args.seed,
        print_per_subcat=args.print_per_subcat,
    )


if __name__ == "__main__":
    main()
