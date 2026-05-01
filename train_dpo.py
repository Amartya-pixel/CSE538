"""
Stage 2 (DPO branch) — Direct Preference Optimization. Persona-aware.

Output: ./{persona}-dpo-merged
"""

import unsloth                                  # noqa: F401
from unsloth import FastLanguageModel

import os
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

PERSONA   = os.environ.get("PERSONA", "dev")
SFT_MODEL = f"{PERSONA}-sft-merged"
DATA_PATH = f"{PERSONA}_dpo.jsonl"
OUT_DIR   = f"{PERSONA}-dpo"
MAX_SEQ   = 2048

print(f"[train_dpo] persona={PERSONA}  base={SFT_MODEL}  out={OUT_DIR}")

model, tokenizer = FastLanguageModel.from_pretrained(
    SFT_MODEL,
    max_seq_length=MAX_SEQ,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

ds = load_dataset("json", data_files=DATA_PATH)["train"]

trainer = DPOTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=ds,
    args=DPOConfig(
        output_dir=OUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=12,          # bumped further — DPO is stable, more passes help
        learning_rate=5e-6,
        beta=0.1,
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        max_length=MAX_SEQ,
        max_prompt_length=1024,
    ),
)

trainer.train()
model.save_pretrained_merged(f"{OUT_DIR}-merged", tokenizer,
                              save_method="merged_16bit")
print(f"DPO model saved -> {OUT_DIR}-merged")
