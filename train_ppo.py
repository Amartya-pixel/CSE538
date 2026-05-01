"""
Stage 2B (RLHF only) — PPO using the reward model. Persona-aware.

Memory-conscious config for A100 40GB with a 1.5B base. With KL pulled tighter
than last run (0.15 vs 0.05) so the policy doesn't drift off the JSON format.

Output: ./{persona}-ppo-final
"""

import os
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification,
                          AutoTokenizer,
                          BitsAndBytesConfig)
from trl import PPOConfig, PPOTrainer

PERSONA   = os.environ.get("PERSONA", "dev")
SFT_MODEL = f"{PERSONA}-sft-merged"
RM_MODEL  = f"{PERSONA}-rm-final"
OUT_DIR   = f"{PERSONA}-ppo-final"

print(f"[train_ppo] persona={PERSONA}  base={SFT_MODEL}  rm={RM_MODEL}")

tok = AutoTokenizer.from_pretrained(SFT_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Policy: trainable, bf16. Do NOT enable gradient checkpointing (TRL wrapper bug).
policy = AutoModelForCausalLM.from_pretrained(
    SFT_MODEL, dtype=torch.bfloat16)

# Reference: frozen, 4-bit
ref = AutoModelForCausalLM.from_pretrained(
    SFT_MODEL, quantization_config=bnb)

# Reward: frozen, 4-bit
reward = AutoModelForSequenceClassification.from_pretrained(
    RM_MODEL, num_labels=1, quantization_config=bnb)

# Value: trainable, bf16
value = AutoModelForSequenceClassification.from_pretrained(
    RM_MODEL, num_labels=1, dtype=torch.bfloat16)

ds = load_dataset("json", data_files=f"{PERSONA}_sft.jsonl")["train"]


def to_prompt(ex):
    text = tok.apply_chat_template(
        ex["messages"][:2], tokenize=False, add_generation_prompt=True)
    enc = tok(text, truncation=True, max_length=384, padding=False)
    return {"input_ids": enc["input_ids"]}


ds = ds.map(to_prompt, remove_columns=ds.column_names)

EVAL_SIZE = min(8, max(2, len(ds) // 20))
ds_train  = ds.select(range(len(ds) - EVAL_SIZE))
ds_eval   = ds.select(range(len(ds) - EVAL_SIZE, len(ds)))

NUM_PASSES = 1
config = PPOConfig(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1.4e-5,
    num_ppo_epochs=1,            # was 2 — reduce overfitting on noisy reward
    num_mini_batches=1,
    response_length=48,
    kl_coef=0.15,                # was 0.05 — strong pull toward SFT distribution
    total_episodes=len(ds_train) * NUM_PASSES,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to=[],
    optim="paged_adamw_8bit",
)

trainer = PPOTrainer(
    args=config,
    processing_class=tok,
    model=policy,
    ref_model=ref,
    reward_model=reward,
    value_model=value,
    train_dataset=ds_train,
    eval_dataset=ds_eval,
)

trainer.train()
trainer.save_model(OUT_DIR)
print(f"PPO model saved -> {OUT_DIR}")
