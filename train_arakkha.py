import torch
import json
import os
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
TRAINING_FILE = "arakkha_training_2026-03.jsonl"
OUTPUT_DIR = "./arakkha-output"
SAVE_DIR = "./arakkha-model"
NUM_EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
MAX_LENGTH = 512
# ──────────────────────────────────────────────────────────

def main():
    print("="*50)
    print("ARAKKHA Local Training")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*50)

    # Check CUDA
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("WARNING: No GPU found. Training will be very slow.")

    # Load dataset
    print(f"\nLoading training data: {TRAINING_FILE}")
    examples = []
    with open(TRAINING_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    dataset = Dataset.from_list(examples)
    print(f"Loaded {len(dataset)} training examples")

    # Load tokenizer
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print(f"Model loaded: {model.num_parameters()/1e6:.0f}M parameters")

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Train
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=OUTPUT_DIR,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=4,
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=50,
            save_steps=500,
            warmup_steps=100,
            lr_scheduler_type="cosine",
            report_to="none",
            max_length=MAX_LENGTH,
            dataset_text_field="text"
        ),
        train_dataset=dataset,
        processing_class=tokenizer
    )

    print(f"\nStarting ARAKKHA training...")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Examples: {len(dataset)}")
    print("This will take 1-2 hours on RTX 4060...\n")

    trainer.train()

    # Save model
    print(f"\nSaving model to {SAVE_DIR}...")
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    print("\n" + "="*50)
    print("ARAKKHA training complete!")
    print(f"Model saved to: {SAVE_DIR}")
    print("="*50)

if __name__ == '__main__':
    main()
