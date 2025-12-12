import argparse
import torch
from src.utils.config import load_config
from src.models.llama_loader import load_model_and_tokenizer
from src.models.steering import (
    load_projection_and_subspace,
    compute_pinv_W,
    steer_h,
)
from data.pku_safe_rlhf import PkuSafeRLHF


def make_dim_hook(
    proj_head,
    U,
    W_pinv,
    valmax,
    M,
    threshold,
    debug_steps=5,
    direction=1.0,
    sanity_random=False,
):
    state = {"step": 0, "call_id": 0}

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h, *rest = output
        else:
            h, rest = output, []

        if not isinstance(h, torch.Tensor) or h.dim() != 3:
            return output

        batch, seq_len, hidden = h.shape
        dev = h.device
        state["call_id"] += 1
        call_id = state["call_id"]

        if seq_len > 1:
            alpha_base = valmax
            step = 0
        else:
            state["step"] += 1
            step = state["step"]
            if M <= 1:
                alpha_base = valmax
            elif step > M:
                alpha_base = 0.0
            else:
                alpha_base = valmax * (1 - (step - 1) / (M - 1))

        alpha = alpha_base * direction
        if alpha == 0.0:
            return output

        h_last = h[:, -1, :]

        if sanity_random:
            with torch.no_grad():
                noise = torch.randn_like(h_last) * alpha * 0.5
                h_new_last = h_last + noise
        else:
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

        delta = h_new_last - h_last
        delta_norm = delta.norm(dim=-1).mean().item()

        h_new_last = h_new_last.to(h.dtype)
        h_mod = h.clone()
        h_mod[:, -1, :] = h_new_last

        if isinstance(output, tuple):
            return (h_mod, *rest)
        else:
            return h_mod

    return hook


def run_demo(cfg_path, num_samples, valmax, max_new_tokens, threshold, direction):
    cfg = load_config(cfg_path)
    model_path = cfg.model.path

    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device

    proj_head, U, _ = load_projection_and_subspace(cfg_path, device=device)
    W_pinv = compute_pinv_W(proj_head).to(device)

    ds = PkuSafeRLHF(split="test", max_samples=num_samples)
    layer_idx = int(cfg.model.layer)
    block = model.model.layers[layer_idx]

    for i, pair in enumerate(ds.iter_pairs()):
        prompt = pair.prompt

        print("=" * 80)
        print(f"Sample {i}")
        print("PROMPT:")
        print(prompt)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=getattr(cfg.model, "max_length", 512),
        ).to(device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out_base = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        base_ids = out_base[0][input_len:]
        base_text = tokenizer.decode(base_ids, skip_special_tokens=True)
        print("\n[BASE]")
        print(base_text)

        hook = make_dim_hook(
            proj_head=proj_head,
            U=U,
            W_pinv=W_pinv,
            valmax=valmax,
            M=max_new_tokens,
            threshold=threshold,
            debug_steps=5,
            direction=direction,
            sanity_random=False,
        )
        handle = block.register_forward_hook(hook)

        with torch.no_grad():
            out_steer = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        handle.remove()

        steer_ids = out_steer[0][input_len:]
        steer_text = tokenizer.decode(steer_ids, skip_special_tokens=True)
        print("\n[STEERED]")
        print(steer_text)

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--valmax", type=float, default=5.0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--direction", type=float, default=1.0)
    args = parser.parse_args()

    run_demo(
        cfg_path=args.cfg,
        num_samples=args.num_samples,
        valmax=args.valmax,
        max_new_tokens=args.max_new_tokens,
        threshold=args.threshold,
        direction=args.direction,
    )
