import os
import json
from typing import Dict, Any

# Force Transformers to use PyTorch-only backends
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_KERAS", "1")

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# 1) Base model: switch to a smaller open model to fit memory
# Default can be overridden via BASE_MODEL env var.
BASE_MODEL = os.environ.get("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# 2) Data and output paths (relative to repo root or run dir)
TRAIN_PATH = "lora/data/train.jsonl"
VALID_PATH = "lora/data/valid.jsonl"
OUT_DIR = "lora/output/adapter"

def describe_jsonl_line(example: Dict[str, Any]) -> None:
    print("One dataset item (as loaded):")
    if "messages" in example:
        for turn in example["messages"]:
            role = turn.get("role")
            content = turn.get("content")
            print(f"  - role={role!r}, content={content!r}")
    else:
        print("  This item has no 'messages' key. Expected a chat-like list.")

def add_text_column(example: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    # Use the model's chat template to convert messages -> a single training string
    if "messages" not in example:
        example["text"] = ""
        return example
    example["text"] = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,  # we train on full dialogs including assistant
    )
    return example

def main():
    # 3) Load dataset
    ds = load_dataset("json", data_files={"train": TRAIN_PATH, "validation": VALID_PATH})
    print("Loaded dataset:")
    print("  train size:", len(ds["train"]))
    print("  validation size:", len(ds["validation"]))

    if len(ds["train"]) > 0:
        describe_jsonl_line(ds["train"][0])

    # 4) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # avoid overflow issues in half precision

    # 5) Turn chat messages into a plain "text" field for SFTTrainer
    ds = ds.map(lambda ex: add_text_column(ex, tokenizer), remove_columns=[c for c in ds["train"].column_names if c != "messages"])

    # 6) Load base model WITHOUT device_map/zero-init to avoid meta tensors
    # Select dtype based on hardware
    if torch.cuda.is_available():
        bf16_ok = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16_ok else torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,  # ensure real tensors, not meta
    )

    # For gradient checkpointing, disable cache
    model.config.use_cache = False

    # 7) Attach LoRA
    lora_config = LoraConfig(
        r=16,                    # rank of the LoRA matrices (capacity of adapter)
        lora_alpha=32,           # scaling for updates
        lora_dropout=0.05,       # dropout on LoRA layers
        bias="none",             # do not train/add biases
        task_type="CAUSAL_LM",   # language modeling
        target_modules=[
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj"
        ],                       # common Mistral projections
    )
    model = get_peft_model(model, lora_config)

    # 8) Training args
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=OUT_DIR,                     # where adapter files will be saved
        per_device_train_batch_size=1,          # tokens per GPU per step
        gradient_accumulation_steps=8,          # effective batch size = 1*8
        learning_rate=2e-4,                     # base LR
        num_train_epochs=2,                     # epochs
        logging_steps=10,                       # log frequency
        save_steps=200,                         # checkpoint frequency
        eval_steps=200,                         # eval frequency
        evaluation_strategy="steps",            # run eval by steps
        bf16=use_bf16,                          # bfloat16 if supported
        fp16=(torch.cuda.is_available() and not use_bf16),
        gradient_checkpointing=True,            # save VRAM
        report_to="none",                       # no external logging
        remove_unused_columns=False,            # important for TRL/SFT datasets
    )

    # 9) SFTTrainer on the "text" field
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        dataset_text_field="text",
        args=args,
        max_seq_length=1024,
    )

    # 10) Train
    print("Starting training...")
    trainer.train()
    print("Training complete.")

    # 11) Save LoRA adapter for Ollama
    os.makedirs(OUT_DIR, exist_ok=True)
    trainer.model.save_pretrained(OUT_DIR)  # writes adapter_model.safetensors + adapter_config.json
    print(f"Adapter saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()