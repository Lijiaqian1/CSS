'''
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


model_name = "meta-llama/Llama-3.1-8B"
local_path = "./model_weights/llama3.1_8b" 

print(f"Loading LLaMA-3.1-8B from {model_name} ...")


tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_path)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,   
    device_map="auto",
    cache_dir=local_path
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("LLaMA-3.1-8B model and tokenizer loaded successfully.")
'''
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


model_name = "meta-llama/Llama-2-7B-hf"
local_path = "./model_weights/llama2_7b" 

print(f"Loading LLaMA-2-7B from {model_name} ...")


tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_path)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,   
    device_map="auto",
    cache_dir=local_path
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

