import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
TRAINING_FILE = r"C:\Users\Asus\ARAKKHA\arakkha_v3_training.jsonl"
OUTPUT_DIR = r"C:\Users\Asus\ARAKKHA\arakkha_v3_lora"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer OK")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto"
)
model.gradient_checkpointing_enable()
print("Model OK")

print("Setting up LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("Loading dataset...")
dataset = load_dataset("json", data_files=TRAINING_FILE, split="train")

def format_prompt(example):
    messages = example["messages"]
    text = ""
    for msg in messages:
        if msg["role"] == "user":
            text += f"<|user|>\n{msg['content']}\n"
        elif msg["role"] == "assistant":
            text += f"<|assistant|>\n{msg['content']}\n"
    return {"text": text}

dataset = dataset.map(format_prompt)
print(f"Dataset size: {len(dataset)} samples")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    report_to="none",
    gradient_checkpointing=True,
    optim="adafactor"
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer
)

print("Starting training...")
trainer.train()

print("Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done! Model saved to", OUTPUT_DIR)
