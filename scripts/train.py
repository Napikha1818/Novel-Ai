"""
train.py
========
Fine-tune Qwen2.5-7B (base) dengan QLoRA.
Langsung pakai PEFT + TRL, tanpa Axolotl.

Usage:
  accelerate launch scripts/train.py
  # atau
  python scripts/train.py

GPU: 1x NVIDIA L4 (24GB)
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# === CONFIG ===
BASE_MODEL = "Qwen/Qwen2.5-7B"
DATASET_PATH = "data/dataset/train_axolotl.jsonl"
EVAL_PATH = "data/dataset/eval.jsonl"
OUTPUT_DIR = "output/eka-qwen2.5-7b-qlora"

# QLoRA
LORA_R = 128
LORA_ALPHA = 256
LORA_DROPOUT = 0.05

# Training
NUM_EPOCHS = 4
BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 1.5e-4
MAX_SEQ_LEN = 8192
WARMUP_RATIO = 0.1
SAVE_STEPS = 100
LOGGING_STEPS = 10


def load_dataset_from_jsonl(path):
    """Load dataset dari JSONL. Support format sharegpt dan instruction."""
    conversations = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if "conversations" in data:
                conversations.append(data["conversations"])
            else:
                # Format instruction: convert ke conversations
                conv = [
                    {"from": "system", "value": data.get("system", "")},
                    {"from": "human", "value": data["instruction"]},
                    {"from": "gpt", "value": data["output"]},
                ]
                conversations.append(conv)
    return Dataset.from_dict({"conversations": conversations})


def format_conversation(example, tokenizer):
    """Format conversations ke chat template."""
    messages = []
    for turn in example["conversations"]:
        role = turn["from"]
        content = turn["value"]
        if role == "system":
            messages.append({"role": "system", "content": content})
        elif role == "human":
            messages.append({"role": "user", "content": content})
        elif role == "gpt":
            messages.append({"role": "assistant", "content": content})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def main():
    print("=" * 60)
    print("Fine-tuning Qwen2.5-7B — Gaya Eka Kurniawan")
    print("=" * 60)

    # === Load tokenizer ===
    print("\n🔄 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set chat template untuk base model (ChatML)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"
        )

    # === Load dataset ===
    print("📚 Loading dataset...")
    train_dataset = load_dataset_from_jsonl(DATASET_PATH)
    eval_dataset = load_dataset_from_jsonl(EVAL_PATH)

    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Eval : {len(eval_dataset)} samples")

    # Format ke text
    train_dataset = train_dataset.map(
        lambda x: format_conversation(x, tokenizer),
        remove_columns=["conversations"],
    )
    eval_dataset = eval_dataset.map(
        lambda x: format_conversation(x, tokenizer),
        remove_columns=["conversations"],
    )

    # === Load model ===
    print("🔄 Loading model (4-bit quantized)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model = prepare_model_for_kbit_training(model)

    # === LoRA config ===
    print("🔧 Applying LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # === Training config ===
    print("⚙️  Setting up training...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="text",
        seed=42,
        report_to="none",
    )

    # === Trainer ===
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        max_seq_length=MAX_SEQ_LEN,
        packing=True,
    )

    # === Train ===
    print("\n🚀 Starting training...")
    print(f"   Model     : {BASE_MODEL}")
    print(f"   LoRA r    : {LORA_R}")
    print(f"   Epochs    : {NUM_EPOCHS}")
    print(f"   Batch     : {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"   LR        : {LEARNING_RATE}")
    print(f"   Seq len   : {MAX_SEQ_LEN}")
    print(f"   Output    : {OUTPUT_DIR}")
    print()

    trainer.train()

    # === Save ===
    print("\n💾 Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n🎉 Training selesai! Model tersimpan di {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
