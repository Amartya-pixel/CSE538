"""
Stage 2A (RLHF only) — Train a reward model on the preference pairs.

Reads PERSONA from environment (default 'dev'). Outputs:
  ./{persona}-rm-final
"""

import os
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer)
from trl import RewardConfig, RewardTrainer

PERSONA   = os.environ.get("PERSONA", "dev")
SFT_MODEL = f"{PERSONA}-sft-merged"
DATA_PATH = f"{PERSONA}_dpo.jsonl"
OUT_DIR   = f"{PERSONA}-rm"

print(f"[train_rm] persona={PERSONA}  base={SFT_MODEL}  out={OUT_DIR}")

tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    SFT_MODEL,
    num_labels=1,
    dtype="bfloat16",
)
model.config.pad_token_id = tokenizer.pad_token_id

ds = load_dataset("json", data_files=DATA_PATH)["train"]


def merge_prompt(ex):
    return {
        "chosen":   f"{ex['prompt']}\n\nResponse: {ex['chosen']}",
        "rejected": f"{ex['prompt']}\n\nResponse: {ex['rejected']}",
    }


ds = ds.map(merge_prompt, remove_columns=ds.column_names)

trainer = RewardTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=ds,
    args=RewardConfig(
        output_dir=OUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=6,           # bumped further
        learning_rate=5e-6,
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        max_length=1024,
    ),
)

trainer.train()
trainer.save_model(f"{OUT_DIR}-final")
tokenizer.save_pretrained(f"{OUT_DIR}-final")
print(f"Reward model saved -> {OUT_DIR}-final")
