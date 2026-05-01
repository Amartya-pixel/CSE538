"""
Parse a persona JSON file (Arjun / Dev / Margaret / Neel format) into:
  - {PERSONA}_sft.jsonl   : SFT training data with persona-aware system prompt
  - {PERSONA}_dpo.jsonl   : preference pairs for DPO and reward-model training
  - {PERSONA}_test.jsonl  : held-out evaluation set (one whole conversation)
  - {PERSONA}_stats.json  : label distribution

The persona's name, role, and personality traits (from the JSON's `tool_rules`
and `assumptions` fields) are folded directly into the system prompt — that's
how the model learns Dev-style vs. Arjun-style decision making.

Usage:
    pip install python-docx        # not actually needed, but harmless
    python parse_json_dataset.py <persona_json> <persona_key> [--holdout 10]

Example:
    python parse_json_dataset.py dev_entrepreneur_*.json dev --holdout 10
"""

import argparse
import json
import os
import random
import re
from collections import Counter

# ---------- Tool taxonomy --------------------------------------------------
TOOL_MAP = {
    "Reminder":       "create_reminder",
    "Notes":          "add_note",
    "Calendar event": "create_calendar_event",
    "Alarm":          "create_alarm",
    "None":           None,
}


def normalize_tool_kind(raw: str) -> str:
    """Convert 'Calendar event - block 9-10' or 'Reminder — register' -> kind."""
    if raw == "None":
        return "None"
    head = re.split(r"\s*[-—]\s*", raw, maxsplit=1)[0].strip()
    return head if head in TOOL_MAP else "None"


def split_detail(raw: str) -> str:
    if raw == "None":
        return ""
    parts = re.split(r"\s*[-—]\s*", raw, maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


# ---------- System prompt construction -------------------------------------
def build_system_prompt(meta: dict) -> str:
    """Compose a persona-aware system prompt from JSON metadata."""
    rules = meta.get("tool_rules", "")
    # Pull "<Name> - <role>" out of "Person: <Name> — <role>. Tool rules ..."
    person_role = "the user"
    if "Person:" in rules:
        chunk = rules.split("Person:", 1)[1]
        chunk = re.split(r"\.\s*Tool rules", chunk, maxsplit=1)[0]
        person_role = chunk.strip().rstrip(".")
    traits = " ".join(meta.get("assumptions", [])).strip()
    if not traits:
        traits = ""

    return (
        f"You are the personal assistant for {person_role}. {traits}\n\n"
        f"You silently listen to their conversations and decide, for every "
        f"spoken line:\n"
        f"  - importance: Low | Medium | High\n"
        f"  - tool: one of create_reminder, add_note, create_calendar_event, "
        f"create_alarm, or null when no action is needed.\n"
        f"Decisions should reflect this person's specific habits and "
        f"priorities, not a generic assistant's. Respond ONLY with a single "
        f"JSON object on one line: "
        f'{{"importance": "...", "tool": "...", "detail": "..."}}'
    ).strip()


# ---------- Extraction -----------------------------------------------------
def extract_samples(meta: dict):
    samples = []
    for convo in meta["conversations"]:
        cnum = convo["number"]
        counterpart = convo.get("context", {}).get("counterpart", "Other")
        for d in convo["dialogues"]:
            tool_kind = normalize_tool_kind(d["tool"])
            samples.append({
                "convo":      cnum,
                "speaker":    d["speaker"],
                "counterpart": counterpart,
                "text":       d["utterance"],
                "importance": d["importance"],
                "tool_kind":  tool_kind,
                "detail":     split_detail(d["tool"]),
            })
    return samples


def to_target(s):
    return {
        "importance": s["importance"],
        "tool":       TOOL_MAP[s["tool_kind"]],
        "detail":     s["detail"],
    }


def fmt_user(s):
    """How the spoken line is presented to the model."""
    return f"{s['speaker']}: {s['text']}"


# ---------- Output writers -------------------------------------------------
def write_sft(samples, system_prompt, path):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps({
                "messages": [
                    {"role": "system",    "content": system_prompt},
                    {"role": "user",      "content": fmt_user(s)},
                    {"role": "assistant", "content": json.dumps(to_target(s))},
                ]
            }) + "\n")


def write_dpo(samples, system_prompt, path, seed=0):
    rng = random.Random(seed)
    importance_vals = ["Low", "Medium", "High"]
    tool_vals = list({TOOL_MAP[k] for k in TOOL_MAP})

    with open(path, "w") as f:
        for s in samples:
            chosen = to_target(s)
            r = rng.random()
            rejected = dict(chosen)
            if r < 0.6 or r > 0.9:
                rejected["tool"] = rng.choice(
                    [t for t in tool_vals if t != chosen["tool"]])
            if r >= 0.6:
                rejected["importance"] = rng.choice(
                    [i for i in importance_vals if i != chosen["importance"]])
            f.write(json.dumps({
                "prompt":   f"{system_prompt}\n\nUtterance: {fmt_user(s)}",
                "chosen":   json.dumps(chosen),
                "rejected": json.dumps(rejected),
            }) + "\n")


def write_test(samples, system_prompt, path):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps({
                "system":    system_prompt,
                "user":      fmt_user(s),
                "gold":      to_target(s),
                "convo":     s["convo"],
                "tool_kind": s["tool_kind"],
            }) + "\n")


# ---------- Main -----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("persona_key", help="e.g. dev / arjun / margaret / neel")
    ap.add_argument("--holdout", type=int, default=10,
                    help="Conversation # held out for testing.")
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.json_path))
    persona = args.persona_key.lower()

    meta = json.load(open(args.json_path))
    sys_prompt = build_system_prompt(meta)

    samples = extract_samples(meta)
    train = [s for s in samples if s["convo"] != args.holdout]
    test  = [s for s in samples if s["convo"] == args.holdout]

    write_sft(train, sys_prompt, os.path.join(out_dir, f"{persona}_sft.jsonl"))
    write_dpo(train, sys_prompt, os.path.join(out_dir, f"{persona}_dpo.jsonl"))
    write_test(test, sys_prompt, os.path.join(out_dir, f"{persona}_test.jsonl"))

    stats = {
        "persona":         persona,
        "system_prompt":   sys_prompt,
        "total":           len(samples),
        "train":           len(train),
        "test":            len(test),
        "holdout_convo":   args.holdout,
        "by_importance":   dict(Counter(s["importance"] for s in samples)),
        "by_tool_kind":    dict(Counter(s["tool_kind"]  for s in samples)),
    }
    with open(os.path.join(out_dir, f"{persona}_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[{persona}] sft={len(train)}  dpo={len(train)}  test={len(test)}")
    print(json.dumps(stats, indent=2)[:1000])


if __name__ == "__main__":
    main()
