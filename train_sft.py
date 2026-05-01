"""
Stage 1 — Supervised Fine-Tuning, persona-aware.

Reads PERSONA from environment (default 'dev'). Outputs:
  ./{persona}-sft-merged   (full HF checkpoint, ready for RM/PPO/DPO)

Tested with: trl>=0.20, transformers>=4.56, peft>=0.18, unsloth>=2026.4.
"""

# Unsloth must be imported before trl/transformers/peft.
import unsloth                                  # noqa: F401
from unsloth import FastLanguageModel

import os
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

PERSONA    = os.environ.get("PERSONA", "dev")
BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
DATA_PATH  = f"{PERSONA}_sft.jsonl"
OUT_DIR    = f"{PERSONA}-sft"
MAX_SEQ    = 2048

print(f"[train_sft] persona={PERSONA}  data={DATA_PATH}  out={OUT_DIR}")

model, tokenizer = FastLanguageModel.from_pretrained(
    BASE_MODEL,
    max_seq_length=MAX_SEQ,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=32,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

ds = load_dataset("json", data_files=DATA_PATH)["train"]


def fmt(ex):
    return {"text": tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False)}


ds = ds.map(fmt, remove_columns=ds.column_names)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=ds,
    args=SFTConfig(
        output_dir=OUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=10,          # bumped further — small data, more passes
        learning_rate=2e-4,
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        dataset_text_field="text",
    ),
)

trainer.train()

model.save_pretrained_merged(f"{OUT_DIR}-merged", tokenizer,
                              save_method="merged_16bit")
print(f"SFT model saved -> {OUT_DIR}-merged")
