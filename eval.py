"""
Score every model variant on the same persona's held-out test set.
Persona-aware: defaults all paths from the PERSONA env var.

Usage:
    PERSONA=dev python eval.py
    # or override individual paths:
    python eval.py --base Qwen/Qwen2.5-1.5B-Instruct \
                   --sft  ./dev-sft-merged \
                   --ctx-sft ./dev-ctx2-sft-merged \
                   --state-ctx-sft ./dev-state-ctx2-sft-merged \
                   --test ./dev_test.jsonl
"""

import argparse
import json
import os
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from json_decision import parse_decision

PERSONA = os.environ.get("PERSONA", "dev")

TOOLS = [
    "create_reminder", "update_reminder", "delete_reminder",
    "add_note", "update_note", "delete_note",
    "create_calendar_event", "update_calendar_event",
    "delete_calendar_event",
    "create_alarm", "update_alarm", "delete_alarm",
    "ask_clarification",
    None,
]
IMPORTANCE_LABELS = ["Low", "Medium", "High"]
WEAK_CLASSES = [
    "create_alarm",
    "update_alarm",
    "delete_alarm",
    "update_reminder",
    "delete_reminder",
    "update_note",
    "delete_note",
    "update_calendar_event",
    "delete_calendar_event",
    "ask_clarification",
    "Medium",
    None,
]


def label(value):
    return "None" if value is None else str(value)


def safe_div(num, den):
    return num / den if den else 0.0


def normalize_detail(text):
    text = "" if text is None else str(text)
    return " ".join(text.lower().strip().split())


def token_f1(pred_detail, gold_detail):
    pred_tokens = normalize_detail(pred_detail).split()
    gold_tokens = normalize_detail(gold_detail).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())
    precision = safe_div(overlap, len(pred_tokens))
    recall = safe_div(overlap, len(gold_tokens))
    return safe_div(2 * precision * recall, precision + recall)


def load_test(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def parse_pred(text):
    return parse_decision(text)


def generate(model_path, examples):
    print(f"\n[load] {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float32
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[eval] dtype={dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype,
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
    exact_correct = 0
    json_exact_correct = 0
    detail_exact_correct = 0
    detail_presence_correct = 0
    detail_token_f1_sum = 0.0
    detail_exact_when_tool_correct = 0
    detail_tool_correct_support = 0
    tp = fp = fn = tn = 0
    per_tool_tp = Counter(); per_tool_fp = Counter(); per_tool_fn = Counter()
    tool_confusion = {label(t): Counter() for t in TOOLS}
    importance_confusion = {imp: Counter() for imp in IMPORTANCE_LABELS}
    weak_stats = {
        label(c): {"support": 0, "correct": 0}
        for c in WEAK_CLASSES
    }

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
        if (pred.get("tool") == gold["tool"] and
                pred.get("importance") == gold["importance"]):
            exact_correct += 1
        if (pred.get("tool") == gold["tool"] and
                pred.get("importance") == gold["importance"] and
                normalize_detail(pred.get("detail")) == normalize_detail(gold.get("detail"))):
            json_exact_correct += 1

        gold_detail = gold.get("detail", "")
        pred_detail = pred.get("detail", "")
        detail_match = normalize_detail(pred_detail) == normalize_detail(gold_detail)
        if detail_match:
            detail_exact_correct += 1
        if bool(normalize_detail(pred_detail)) == bool(normalize_detail(gold_detail)):
            detail_presence_correct += 1
        detail_token_f1_sum += token_f1(pred_detail, gold_detail)
        if pred.get("tool") == gold["tool"]:
            detail_tool_correct_support += 1
            if detail_match:
                detail_exact_when_tool_correct += 1

        gold_tool = gold["tool"]
        pred_tool = pred.get("tool")
        gold_imp = gold["importance"]
        pred_imp = pred.get("importance")
        tool_confusion.setdefault(label(gold_tool), Counter())[label(pred_tool)] += 1
        importance_confusion.setdefault(gold_imp, Counter())[pred_imp] += 1

        gold_pos = gold_tool is not None
        pred_pos = pred_tool is not None
        if gold_pos and pred_pos:
            tp += 1
        elif not gold_pos and pred_pos:
            fp += 1
        elif gold_pos and not pred_pos:
            fn += 1
        else:
            tn += 1

        if gold_tool is not None:
            if pred_tool == gold_tool:
                per_tool_tp[gold_tool] += 1
            else:
                per_tool_fn[gold_tool] += 1
        if pred_tool is not None and pred_tool != gold_tool:
            per_tool_fp[pred_tool] += 1

        weak_tool_key = label(gold_tool)
        if weak_tool_key in weak_stats:
            weak_stats[weak_tool_key]["support"] += 1
            if pred_tool == gold_tool:
                weak_stats[weak_tool_key]["correct"] += 1
        if gold_imp == "Medium":
            weak_stats["Medium"]["support"] += 1
            if pred_imp == gold_imp:
                weak_stats["Medium"]["correct"] += 1

    prec = safe_div(tp, tp + fp)
    rec  = safe_div(tp, tp + fn)
    f1   = safe_div(2 * prec * rec, prec + rec)

    per_tool_precision_recall_f1 = {}
    for t in TOOLS:
        if t is None:
            continue
        ttp, tfp, tfn = per_tool_tp[t], per_tool_fp[t], per_tool_fn[t]
        p = safe_div(ttp, ttp + tfp)
        r = safe_div(ttp, ttp + tfn)
        per_tool_precision_recall_f1[t] = {
            "precision": p,
            "recall": r,
            "f1": safe_div(2 * p * r, p + r),
            "support": ttp + tfn,
            "tp": ttp,
            "fp": tfp,
            "fn": tfn,
        }

    weak_class_recall = {}
    for cls, vals in weak_stats.items():
        weak_class_recall[cls] = {
            "support": vals["support"],
            "correct": vals["correct"],
            "recall": safe_div(vals["correct"], vals["support"]),
        }

    action_vs_no_action_confusion = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "matrix": {
            "Action": {"Action": tp, "NoAction": fn},
            "NoAction": {"Action": fp, "NoAction": tn},
        },
    }
    per_tool_f1 = {
        t: vals["f1"]
        for t, vals in per_tool_precision_recall_f1.items()
    }

    return {
        "n":                  n,
        "json_parse_rate":    safe_div(parsed_ok, n),
        "tool_accuracy":      safe_div(tool_correct, n),
        "importance_accuracy": safe_div(imp_correct, n),
        "exact_match_accuracy": safe_div(exact_correct, n),
        "json_exact_match_accuracy": safe_div(json_exact_correct, n),
        "detail_exact_match_accuracy": safe_div(detail_exact_correct, n),
        "detail_presence_accuracy": safe_div(detail_presence_correct, n),
        "detail_token_f1": safe_div(detail_token_f1_sum, n),
        "detail_exact_when_tool_correct": safe_div(
            detail_exact_when_tool_correct, detail_tool_correct_support),
        "action_precision":   prec,
        "action_recall":      rec,
        "action_f1":          f1,
        "per_tool_f1":        per_tool_f1,
        "tool_confusion_matrix": {
            gold: dict(preds)
            for gold, preds in tool_confusion.items()
        },
        "importance_confusion_matrix": {
            gold: dict(preds)
            for gold, preds in importance_confusion.items()
        },
        "action_vs_no_action_confusion": action_vs_no_action_confusion,
        "per_tool_precision_recall_f1": per_tool_precision_recall_f1,
        "weak_class_recall": weak_class_recall,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--sft",  default=f"./{PERSONA}-sft-merged")
    ap.add_argument("--ctx-sft", default="",
                    help="Optional context-window SFT model path.")
    ap.add_argument("--state-ctx-sft", default="",
                    help="Optional state plus context-window SFT model path.")
    ap.add_argument("--test", default=f"./{PERSONA}_test.jsonl")
    args = ap.parse_args()

    examples = load_test(args.test)
    print(f"[{PERSONA}] loaded {len(examples)} test utterances.")

    runs = [("base", args.base), ("sft", args.sft),
            ("ctx_sft", args.ctx_sft),
            ("state_ctx_sft", args.state_ctx_sft)]
    results = {}
    for name, path in runs:
        if not path:
            print(f"[skip] {name} ({path})")
            continue
        if path.startswith((".", "/")) and not os.path.isdir(path):
            print(f"[skip] {name} missing local model folder: {path}")
            continue
        if not (os.path.isdir(path) or "/" in path):
            print(f"[skip] {name} ({path})")
            continue
        try:
            preds = generate(path, examples)
            results[name] = score(examples, preds)
        except Exception as e:
            print(f"[error] {name}: {e}")

    print("\n" + "=" * 88)
    print(f"{'model':<14} {'tool_acc':>9} {'imp_acc':>9} {'prec':>7} "
          f"{'rec':>7} {'f1':>7} {'parse':>7} {'exact':>7} {'detail_f1':>9}")
    print("-" * 88)
    for name, m in results.items():
        print(f"{name:<14} {m['tool_accuracy']:>9.3f} "
              f"{m['importance_accuracy']:>9.3f} "
              f"{m['action_precision']:>7.3f} {m['action_recall']:>7.3f} "
              f"{m['action_f1']:>7.3f} {m['json_parse_rate']:>7.3f} "
              f"{m['exact_match_accuracy']:>7.3f} "
              f"{m['detail_token_f1']:>9.3f}")
    print("=" * 88)
    for name, m in results.items():
        print(f"  {name}: per_tool_f1={m['per_tool_f1']}")

    out = f"{PERSONA}_eval_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull metrics saved to {out}")


if __name__ == "__main__":
    main()
