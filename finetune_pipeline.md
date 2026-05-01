# Fine-Tuning Pipeline — RLHF vs DPO Comparison

## What you're comparing

```
                                  ┌──────────────────────┐
                                  │  base Qwen2.5-3B     │
                                  └──────────┬───────────┘
                                             │
                                       train_sft.py
                                             ▼
                                  ┌──────────────────────┐
                                  │   SFT-only model     │  ← baseline
                                  └─────┬─────────┬──────┘
                                        │         │
                       train_reward_model.py    train_dpo.py
                                        │         │
                                        ▼         ▼
                          ┌────────────────┐  ┌──────────────────┐
                          │  reward model  │  │  SFT + DPO model │
                          └───────┬────────┘  └──────────────────┘
                                  │
                            train_ppo.py
                                  ▼
                          ┌────────────────┐
                          │ SFT + RM + PPO │  ← classical RLHF
                          └────────────────┘

                                eval.py scores all four on the held-out conversation
```

You end up with four models scored on the same held-out conversation #10:
**base**, **sft**, **dpo**, **ppo**. Same data, same eval, apples to apples.

## Why both, conceptually

| | Classical RLHF (PPO) | DPO |
|--|--|--|
| Stages       | SFT → reward model → PPO | SFT → DPO |
| Reward model | yes — separate ~hours of training, can overfit | none |
| Optimizer    | reinforcement learning loop with KL penalty | plain supervised loss derived from preferences |
| Stability    | finicky: reward hacking, KL blow-ups, batch sensitivity | very stable |
| Compute      | ~5–10× DPO | baseline |
| Code length  | hundreds of lines, careful tuning | one trainer call |
| Result       | comparable on most benchmarks | comparable |
| Used by      | InstructGPT, original ChatGPT | Llama 3, Qwen2/3, Mistral |

For a structured-prediction task like yours (predict importance + tool),
expect the gap between **sft → dpo** to be small and the gap between
**dpo → ppo** to be smaller still. RLHF/DPO shine on open-ended generation
where preferences are about style; for a task with crisp ground truth, SFT
is doing most of the work. But the comparison itself is the contribution.

## Step-by-step

### 0. Generate the data splits

```bash
cd "/Users/amartya/Documents/CSE534 project"
source .venv/bin/activate
pip install python-docx
python parse_dataset.py \
    "/path/to/arjun_10_conversations_evolving_tool_calls_labeled.docx" \
    --holdout 10
```

Produces `sft.jsonl` (~165 train), `dpo.jsonl` (~165 preference pairs),
`test.jsonl` (~17 utterances from conversation 10), `stats.json`.

**Augment first.** Use Claude/GPT-4o to paraphrase each utterance 5–10×
keeping the label fixed. Also synthesize ~30 Alarm examples — your data
only has 1, the model will never learn that class otherwise.

### 1. SFT (shared by both branches)

```bash
pip install unsloth trl==0.11.4 transformers==4.45.2 datasets accelerate
python train_sft.py
```

Output: `qwen-arjun-sft-merged/` — full FP16 weights, ready for both branches.

### 2A. RLHF branch — reward model + PPO

```bash
python train_reward_model.py    # ~10 min on a Colab T4
python train_ppo.py             # ~30–60 min depending on epochs
```

Outputs: `qwen-arjun-rm-final/` and `qwen-arjun-ppo-final/`.

**What PPO is actually doing**, in one paragraph: the policy generates a
response to each prompt, the reward model scores it, and the policy is
nudged to produce higher-scoring responses — but constrained by a KL
penalty against the frozen SFT model so it doesn't drift into incoherent
text just to maximize reward. The KL coefficient (`init_kl_coef=0.1`) is
the most important knob; too low and the policy collapses, too high and
nothing changes.

### 2B. DPO branch

```bash
python train_dpo.py             # ~15 min on a Colab T4
```

Output: `qwen-arjun-dpo-merged/`.

**What DPO is doing**, also in one paragraph: it skips the reward model
entirely. For each preference pair `(x, chosen, rejected)`, the loss
directly increases the policy's probability of `chosen` relative to
`rejected`, *while* implicitly regularizing against the reference policy
(controlled by `beta`). It's mathematically equivalent to running PPO
against an optimal reward model — but you never have to train one.

### 3. Evaluate all four

```bash
python eval.py \
    --base Qwen/Qwen2.5-3B-Instruct \
    --sft  ./qwen-arjun-sft-merged \
    --dpo  ./qwen-arjun-dpo-merged \
    --ppo  ./qwen-arjun-ppo-final \
    --test ./test.jsonl
```

Prints a table like:

```
model  tool_acc  imp_acc   prec     rec      f1     parse
base       0.41    0.59   0.55    0.82   0.66    0.74
sft        0.82    0.76   0.91    0.88   0.89    1.00
dpo        0.88    0.82   0.93    0.89   0.91    1.00
ppo        0.85    0.81   0.94    0.85   0.89    0.98
```

(Numbers are illustrative — your actual numbers depend on dataset size
and augmentation.) Saves full metrics to `eval_results.json` for plots.

### 4. Use the winner in the listening agent

Pick whichever model performs best, convert to GGUF, register with Ollama,
and point `listening_agent.py` at it:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
python convert_hf_to_gguf.py /path/to/qwen-arjun-dpo-merged \
    --outfile qwen-arjun.gguf --outtype q4_k_m

cat > Modelfile <<'EOF'
FROM ./qwen-arjun.gguf
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
EOF

ollama create qwen-arjun -f Modelfile
```

In `listening_agent.py`:
- `MODEL_NAME = "qwen-arjun"`
- swap `mcp_server.py` for `arjun_mcp_server.py` in `SERVER_SCRIPT`
- update the agent to parse the JSON output (`{"importance": ..., "tool": ..., "detail": ...}`)
  instead of relying on Ollama's `tool_calls` format

## Hardware reality check

| Stage | Mac (MLX) | Cloud GPU |
|--|--|--|
| SFT  | yes — `mlx_lm.lora` works  | yes — Colab T4 free |
| RM   | painful — no good MLX path | yes — needed       |
| PPO  | not realistic              | yes — needed       |
| DPO  | yes — `mlx_lm.dpo` exists   | yes                |

Translation: the **DPO branch** is the only one you can realistically run
end-to-end on your MacBook. The **PPO branch** requires Colab/Lambda/RunPod.
A free Colab T4 session is enough for both branches at 3B scale.

## What goes in your CSE534 writeup

Three specific things that will read well to a grader:

1. **The pipeline diagram + side-by-side metrics** — RLHF and DPO trained
   on the *same SFT base* with the *same preference data*, scored on the
   *same held-out conversation*.
2. **Wall-clock and code-complexity comparison** — total training time
   per branch, lines of code, GPU-hours. RLHF will lose decisively here.
3. **Per-class behavior** — does PPO over-fire low-importance utterances?
   Does DPO collapse the rare Alarm class? Per-tool F1 in `eval.py`
   surfaces this.
