from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_id = "meta-llama/Llama-3.2-3B-Instruct"
adapter_path = r"C:\Users\Asus\ARAKKHA\arakkha-model"
output_path = r"C:\Users\Asus\ARAKKHA\arakkha-merged"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="cpu"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, adapter_path)

print("Merging...")
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained(output_path)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.save_pretrained(output_path)

print("Done!")
