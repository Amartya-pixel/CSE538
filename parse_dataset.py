"""
Parse arjun_10_conversations_evolving_tool_calls_labeled.docx into:
  - sft.jsonl   : supervised fine-tuning data (Qwen chat format)
  - dpo.jsonl   : preference pairs for DPO and reward-model training
  - test.jsonl  : held-out evaluation set (one whole conversation)
  - stats.json  : label distribution for sanity checking

Usage:
    pip install python-docx
    python parse_dataset.py <docx_path> [--holdout 10]

Default holdout is conversation #10 — that conversation's utterances are
written ONLY to test.jsonl and never appear in sft.jsonl or dpo.jsonl.
"""

import argparse
import json
import os
import random
import re
from collections import Counter

from docx import Document

TOOL_MAP = {
    "Reminder":       "create_reminder",
    "Notes":          "add_note",
    "Calendar event": "create_calendar_event",
    "Alarm":          "create_alarm",
    "None":           None,
}

KNOWN_SPEAKERS = {
    "Arjun", "Buddy", "Manager", "Trainer", "Mom", "Sister",
    "Roommate", "Receptionist", "Dad", "Building Manager", "Neighbor",
}

SYSTEM_PROMPT = (
    "You are Arjun's personal assistant. You silently listen to his conversations. "
    "For every spoken line, decide:\n"
    "  - importance: Low | Medium | High\n"
    "  - tool: one of create_reminder, add_note, create_calendar_event, "
    "create_alarm, or null when no action is needed.\n"
    "Respond ONLY with a single JSON object on one line: "
    '{"importance": "...", "tool": "...", "detail": "..."}'
)

IMP_PAT  = re.compile(r"Importance:\s*(Low|Medium|High)\s*\|\s*Tool:\s*(.+)")
UTT_PAT  = re.compile(r"^([^:]+?):\s*(.+)$")
CONV_PAT = re.compile(r"Conversation\s+(\d+)")


def parse_doc(path):
    doc = Document(path)
    lines = [p.text for p in doc.paragraphs if p.text.strip()]

    samples = []
    pending = None
    convo_id = 0

    for line in lines:
        m_conv = CONV_PAT.match(line)
        if m_conv:
            convo_id = int(m_conv.group(1))
            pending = None
            continue

        m_imp = IMP_PAT.match(line)
        if m_imp and pending is not None:
            tool_raw = m_imp.group(2).strip()
            if tool_raw == "None":
                tool_kind, detail = "None", ""
            else:
                parts = re.split(r"\s*[—-]\s*", tool_raw, maxsplit=1)
                tool_kind = parts[0].strip()
                detail    = parts[1].strip() if len(parts) > 1 else ""
            if tool_kind in TOOL_MAP:
                samples.append({
                    "convo":      convo_id,
                    "speaker":    pending[0],
                    "text":       pending[1],
                    "importance": m_imp.group(1),
                    "tool_kind":  tool_kind,
                    "detail":     detail,
                })
            pending = None
            continue

        m_utt = UTT_PAT.match(line)
        if m_utt and m_utt.group(1).strip() in KNOWN_SPEAKERS:
            pending = (m_utt.group(1).strip(), m_utt.group(2).strip())

    return samples


def to_target(s):
    return {
        "importance": s["importance"],
        "tool":       TOOL_MAP[s["tool_kind"]],
        "detail":     s["detail"],
    }


def write_sft(samples, path):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps({
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": f"{s['speaker']}: {s['text']}"},
                    {"role": "assistant", "content": json.dumps(to_target(s))},
                ]
            }) + "\n")


def write_dpo(samples, path, seed=0):
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
                "prompt":   f"{SYSTEM_PROMPT}\n\nUtterance: {s['speaker']}: {s['text']}",
                "chosen":   json.dumps(chosen),
                "rejected": json.dumps(rejected),
            }) + "\n")


def write_test(samples, path):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps({
                "system":    SYSTEM_PROMPT,
                "user":      f"{s['speaker']}: {s['text']}",
                "gold":      to_target(s),
                "convo":     s["convo"],
                "tool_kind": s["tool_kind"],
            }) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("docx")
    ap.add_argument("--holdout", type=int, default=10,
                    help="Conversation # to hold out for testing (default 10).")
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.docx))
    samples = parse_doc(args.docx)

    train = [s for s in samples if s["convo"] != args.holdout]
    test  = [s for s in samples if s["convo"] == args.holdout]

    write_sft(train,  os.path.join(out_dir, "sft.jsonl"))
    write_dpo(train,  os.path.join(out_dir, "dpo.jsonl"))
    write_test(test,  os.path.join(out_dir, "test.jsonl"))

    stats = {
        "total":           len(samples),
        "train":           len(train),
        "test":            len(test),
        "holdout_convo":   args.holdout,
        "by_importance":   dict(Counter(s["importance"] for s in samples)),
        "by_tool_kind":    dict(Counter(s["tool_kind"]  for s in samples)),
    }
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote sft.jsonl ({len(train)})  dpo.jsonl ({len(train)})  "
          f"test.jsonl ({len(test)})")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
