import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import datasets

BASE = "meta-llama/Llama-3.2-3B-Instruct"
DATA = r"C:\Users\Asus\ARAKKHA\arakkha_v4_training.jsonl"
OUT  = r"C:\Users\Asus\ARAKKHA\arakkha_v4_lora"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(BASE)
tokenizer.pad_token = tokenizer.eos_token

print("Loading dataset...")
dataset = datasets.load_dataset("json", data_files=DATA, split="train")
print(f"Dataset size: {len(dataset)} pairs")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","v_proj","k_proj","o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = SFTConfig(
    output_dir=OUT,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    warmup_steps=100,
    lr_scheduler_type="cosine",
    report_to="none",
    max_grad_norm=0.3,
    dataloader_pin_memory=False,
    max_length=512,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

print("Starting ARAKKHA v4 training...")
trainer.train()
print("Saving model...")
model.save_pretrained(OUT)
tokenizer.save_pretrained(OUT)
print("Done! Model saved to", OUT)
