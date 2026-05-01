"""
Score every model variant on the same persona's held-out test set.
Persona-aware: defaults all paths from the PERSONA env var.

Usage:
    PERSONA=dev python eval.py
    # or override individual paths:
    python eval.py --base Qwen/Qwen2.5-1.5B-Instruct \
                   --sft  ./dev-sft-merged \
                   --dpo  ./dev-dpo-merged \
                   --ppo  ./dev-ppo-final \
                   --test ./dev_test.jsonl
"""

import argparse
import json
import os
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PERSONA = os.environ.get("PERSONA", "dev")

TOOLS = ["create_reminder", "add_note", "create_calendar_event",
         "create_alarm", None]


def load_test(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def parse_pred(text):
    text = text.strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def generate(model_path, examples):
    print(f"\n[load] {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    preds = []
    for ex in examples:
        msgs = [
            {"role": "system", "content": ex["system"]},
            {"role": "user",   "content": ex["user"]},
        ]
        prompt = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inp, max_new_tokens=80, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        gen = tok.decode(out[0, inp["input_ids"].shape[1]:],
                         skip_special_tokens=True)
        preds.append(gen)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds


def score(examples, preds):
    n            = len(examples)
    parsed_ok    = 0
    tool_correct = 0
    imp_correct  = 0
    tp = fp = fn = tn = 0
    per_tool_tp = Counter(); per_tool_fp = Counter(); per_tool_fn = Counter()

    for ex, raw in zip(examples, preds):
        gold = ex["gold"]
        pred = parse_pred(raw)
        if pred is None:
            pred = {"importance": "Low", "tool": None, "detail": ""}
        else:
            parsed_ok += 1

        if pred.get("tool") == gold["tool"]:
            tool_correct += 1
        if pred.get("importance") == gold["importance"]:
            imp_correct += 1

        gold_pos = gold["tool"] is not None
        pred_pos = pred.get("tool") is not None
        if gold_pos and pred_pos:
            tp += 1
        elif not gold_pos and pred_pos:
            fp += 1
        elif gold_pos and not pred_pos:
            fn += 1
        else:
            tn += 1

        if gold["tool"] is not None:
            if pred.get("tool") == gold["tool"]:
                per_tool_tp[gold["tool"]] += 1
            else:
                per_tool_fn[gold["tool"]] += 1
        if pred.get("tool") is not None and pred.get("tool") != gold["tool"]:
            per_tool_fp[pred["tool"]] += 1

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    per_tool_f1 = {}
    for t in TOOLS:
        if t is None:
            continue
        ttp, tfp, tfn = per_tool_tp[t], per_tool_fp[t], per_tool_fn[t]
        p = ttp / (ttp + tfp) if (ttp + tfp) else 0.0
        r = ttp / (ttp + tfn) if (ttp + tfn) else 0.0
        per_tool_f1[t] = (2 * p * r / (p + r)) if (p + r) else 0.0

    return {
        "n":                  n,
        "json_parse_rate":    parsed_ok / n,
        "tool_accuracy":      tool_correct / n,
        "importance_accuracy": imp_correct / n,
        "action_precision":   prec,
        "action_recall":      rec,
        "action_f1":          f1,
        "per_tool_f1":        per_tool_f1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--sft",  default=f"./{PERSONA}-sft-merged")
    ap.add_argument("--dpo",  default=f"./{PERSONA}-dpo-merged")
    ap.add_argument("--ppo",  default=f"./{PERSONA}-ppo-final")
    ap.add_argument("--test", default=f"./{PERSONA}_test.jsonl")
    args = ap.parse_args()

    examples = load_test(args.test)
    print(f"[{PERSONA}] loaded {len(examples)} test utterances.")

    runs = [("base", args.base), ("sft", args.sft),
            ("dpo", args.dpo),   ("ppo", args.ppo)]
    results = {}
    for name, path in runs:
        if not path or not (os.path.isdir(path) or "/" in path):
            print(f"[skip] {name} ({path})")
            continue
        try:
            preds = generate(path, examples)
            results[name] = score(examples, preds)
        except Exception as e:
            print(f"[error] {name}: {e}")

    print("\n" + "=" * 88)
    print(f"{'model':<6} {'tool_acc':>9} {'imp_acc':>9} {'prec':>7} "
          f"{'rec':>7} {'f1':>7} {'parse':>7}")
    print("-" * 88)
    for name, m in results.items():
        print(f"{name:<6} {m['tool_accuracy']:>9.3f} "
              f"{m['importance_accuracy']:>9.3f} "
              f"{m['action_precision']:>7.3f} {m['action_recall']:>7.3f} "
              f"{m['action_f1']:>7.3f} {m['json_parse_rate']:>7.3f}")
    print("=" * 88)
    for name, m in results.items():
        print(f"  {name}: per_tool_f1={m['per_tool_f1']}")

    out = f"{PERSONA}_eval_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull metrics saved to {out}")


if __name__ == "__main__":
    main()
