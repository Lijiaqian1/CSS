import os
from typing import Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_tokenizer(
    model_path: str,
    padding_side: str = "left",
    use_fast: bool = True,
) -> AutoTokenizer:
    model_path = os.path.expanduser(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=use_fast,
        padding_side=padding_side,
        truncation_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_llama_causal_lm(
    model_path: str,
    dtype: torch.dtype = torch.float16,
    device_map: str = "auto",
    cache_dir: Optional[str] = None,
) -> AutoModelForCausalLM:
    model_path = os.path.expanduser(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    return model


def load_model_and_tokenizer(
    model_path: str,
    dtype: torch.dtype = torch.float16,
    device_map: str = "auto",
    cache_dir: Optional[str] = None,
    padding_side: str = "left",
    use_fast: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = load_tokenizer(
        model_path=model_path,
        padding_side=padding_side,
        use_fast=use_fast,
    )
    model = load_llama_causal_lm(
        model_path=model_path,
        dtype=dtype,
        device_map=device_map,
        cache_dir=cache_dir,
    )
    return model, tokenizer
