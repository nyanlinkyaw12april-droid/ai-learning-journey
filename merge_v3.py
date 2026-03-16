import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "meta-llama/Llama-3.2-3B-Instruct"
LORA = r"C:\Users\Asus\ARAKKHA\arakkha_v3_lora"
OUT  = r"C:\Users\Asus\ARAKKHA\arakkha_v3_merged"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(BASE)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA)

print("Merging...")
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained(OUT)
tokenizer.save_pretrained(OUT)
print("Done!", OUT)
