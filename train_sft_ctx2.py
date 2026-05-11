"""
Stage 1B - Supervised Fine-Tuning with previous-2-turn context.

Reads PERSONA from environment (default 'dev'). Inputs:
  ./{persona}_ctx2_sft.jsonl

Outputs:
  ./{persona}-ctx2-sft-merged

This is intentionally separate from train_sft.py so the current-utterance
baseline and the context model have visibly different training entrypoints.
"""

# Unsloth must be imported before trl/transformers/peft.
import unsloth                                  # noqa: F401
from unsloth import FastLanguageModel

import os
import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

PERSONA    = os.environ.get("PERSONA", "dev")
BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
DATA_PATH  = f"{PERSONA}_ctx2_sft.jsonl"
OUT_DIR    = f"{PERSONA}-ctx2-sft"
MAX_SEQ    = 2048

print(f"[train_sft_ctx2] persona={PERSONA}  data={DATA_PATH}  out={OUT_DIR}")

USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
USE_FP16 = torch.cuda.is_available() and not USE_BF16
print(f"[train_sft_ctx2] precision: bf16={USE_BF16} fp16={USE_FP16}")

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
        num_train_epochs=10,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        bf16=USE_BF16,
        fp16=USE_FP16,
        logging_steps=10,
        save_strategy="epoch",
        dataset_text_field="text",
    ),
)

trainer.train()

model.save_pretrained_merged(f"{OUT_DIR}-merged", tokenizer,
                              save_method="merged_16bit")
print(f"Context SFT model saved -> {OUT_DIR}-merged")
