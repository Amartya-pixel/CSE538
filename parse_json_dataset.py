"""
Parse a persona JSON file (Arjun / Dev / Margaret / Neel format) into:
  - {PERSONA}_sft.jsonl   : SFT training data with persona-aware system prompt
  - {PERSONA}_test.jsonl  : held-out evaluation set
  - {PERSONA}_stats.json  : label distribution

The persona's name, role, and personality traits (from the JSON's `tool_rules`
and `assumptions` fields) are folded directly into the system prompt — that's
how the model learns Dev-style vs. Arjun-style decision making.

Usage:
    pip install python-docx        # not actually needed, but harmless
    python parse_json_dataset.py <persona_json> <persona_key> [--holdout 10]

Example:
    python parse_json_dataset.py dev_entrepreneur_*.json dev --holdout 10
    python parse_json_dataset.py dev_entrepreneur_*.json dev --holdout 10 16 17
"""

import argparse
import json
import os
import re
from collections import Counter

# ---------- Tool taxonomy --------------------------------------------------
TOOL_MAP = {
    "Reminder":              "create_reminder",
    "Reminder update":       "update_reminder",
    "Reminder delete":       "delete_reminder",
    "Notes":                 "add_note",
    "Notes update":          "update_note",
    "Notes delete":          "delete_note",
    "Calendar event":        "create_calendar_event",
    "Calendar event update": "update_calendar_event",
    "Calendar event delete": "delete_calendar_event",
    "Alarm":                 "create_alarm",
    "Alarm update":          "update_alarm",
    "Alarm delete":          "delete_alarm",
    "Clarification":         "ask_clarification",
    "None":                  None,
}

TOOL_NAMES = [v for v in TOOL_MAP.values() if v is not None]

TOOL_DEFINITIONS = (
    "Universal tool definitions, independent of persona:\n"
    "  - create_reminder: use when someone should not forget a future task.\n"
    "  - update_reminder: use when an existing reminder should change.\n"
    "  - delete_reminder: use when an existing reminder is obsolete or canceled.\n"
    "  - create_calendar_event: use when time should be reserved on a calendar.\n"
    "  - update_calendar_event: use when an existing calendar block should change.\n"
    "  - delete_calendar_event: use when an existing calendar block is canceled.\n"
    "  - create_alarm: use for an urgent prompt at a specific moment.\n"
    "  - update_alarm: use when an existing urgent prompt should change.\n"
    "  - delete_alarm: use when an existing urgent prompt is obsolete or canceled.\n"
    "  - add_note: use for reference material worth saving.\n"
    "  - update_note: use when saved reference material should change.\n"
    "  - delete_note: use when saved reference material should be removed.\n"
    "  - ask_clarification: use when the user intent is ambiguous, missing a target, or missing required timing/details.\n"
    "  - null: use when no action is needed."
)


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
        f"{TOOL_DEFINITIONS}\n\n"
        f"Decision order for every spoken line:\n"
        f"  1. Choose tool using only the universal tool definitions. Do not "
        f"change tool meaning based on persona.\n"
        f"  2. Choose importance using the person's habits, goals, risks, "
        f"deadlines, and preferences.\n"
        f"Allowed importance values: Low | Medium | High.\n"
        f"Allowed tool values: {', '.join(TOOL_NAMES)}, or null.\n"
        f"Respond ONLY with a single valid JSON object on one line.\n"
        f"Do not include markdown, code fences, comments, or explanations.\n"
        f"Use exactly these keys: importance, tool, detail.\n"
        f"When tool is null, detail must be an empty string.\n"
        f"Required JSON shape: "
        f'{{"importance": "Low|Medium|High", '
        f'"tool": "allowed tool name or null", "detail": "string"}}'
    ).strip()


# ---------- Extraction -----------------------------------------------------
def state_label(tool_kind: str) -> str:
    if tool_kind.startswith("Reminder"):
        return "Reminder"
    if tool_kind.startswith("Calendar event"):
        return "Calendar"
    if tool_kind.startswith("Alarm"):
        return "Alarm"
    if tool_kind.startswith("Notes"):
        return "Note"
    return ""


def apply_state_update(state: list[dict], tool_kind: str, detail: str) -> None:
    label = state_label(tool_kind)
    if not label or not detail:
        return
    if tool_kind.endswith("delete"):
        for i, item in enumerate(state):
            if item["type"] == label:
                del state[i]
                return
        return
    if tool_kind.endswith("update"):
        for item in reversed(state):
            if item["type"] == label:
                item["detail"] = detail
                return
    state.append({"type": label, "detail": detail})


def extract_samples(meta: dict, context_turns: int = 0, include_state: bool = False):
    samples = []
    for convo in meta["conversations"]:
        cnum = convo["number"]
        counterpart = convo.get("context", {}).get("counterpart", "Other")
        history = []
        state = []
        for d in convo["dialogues"]:
            tool_kind = normalize_tool_kind(d["tool"])
            detail = split_detail(d["tool"])
            samples.append({
                "convo":      cnum,
                "speaker":    d["speaker"],
                "counterpart": counterpart,
                "text":       d["utterance"],
                "previous_turns": history[-context_turns:] if context_turns else [],
                "state_items": [dict(item) for item in state] if include_state else [],
                "importance": d["importance"],
                "tool_kind":  tool_kind,
                "detail":     detail,
            })
            history.append({
                "speaker": d["speaker"],
                "text": d["utterance"],
            })
            apply_state_update(state, tool_kind, detail)
    return samples


def to_target(s):
    return {
        "importance": s["importance"],
        "tool":       TOOL_MAP[s["tool_kind"]],
        "detail":     s["detail"],
    }


def fmt_user(s):
    """How the spoken line is presented to the model."""
    previous = s.get("previous_turns") or []
    state_items = s.get("state_items") or []
    parts = []
    if previous:
        context_lines = "\n".join(
            f"{turn['speaker']}: {turn['text']}" for turn in previous
        )
        parts.append(f"Recent conversation:\n{context_lines}")
    if state_items:
        state_lines = "\n".join(
            f"- {item['type']}: {item['detail']}" for item in state_items
        )
        parts.append(f"Existing saved items:\n{state_lines}")
    parts.append(f"Current utterance:\n{s['speaker']}: {s['text']}")
    return "\n\n".join(parts) if (previous or state_items) else f"{s['speaker']}: {s['text']}"


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
    ap.add_argument("--holdout", type=int, nargs="+", default=[10],
                    help="Conversation number(s) held out for testing.")
    ap.add_argument("--context-turns", type=int, default=0,
                    help="Number of previous dialogue turns to include.")
    ap.add_argument("--include-state", action="store_true",
                    help="Include existing saved tool state before the current utterance.")
    ap.add_argument("--suffix", default="",
                    help="Optional output suffix, e.g. _ctx2.")
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.json_path))
    persona = args.persona_key.lower()

    meta = json.load(open(args.json_path))
    sys_prompt = build_system_prompt(meta)

    if args.context_turns < 0:
        raise ValueError("--context-turns must be >= 0")
    if args.suffix and not args.suffix.startswith("_"):
        args.suffix = "_" + args.suffix

    holdouts = set(args.holdout)
    samples = extract_samples(
        meta, context_turns=args.context_turns,
        include_state=args.include_state,
    )
    train = [s for s in samples if s["convo"] not in holdouts]
    test  = [s for s in samples if s["convo"] in holdouts]

    prefix = f"{persona}{args.suffix}"
    write_sft(train, sys_prompt, os.path.join(out_dir, f"{prefix}_sft.jsonl"))
    write_test(test, sys_prompt, os.path.join(out_dir, f"{prefix}_test.jsonl"))

    stats = {
        "persona":         persona,
        "system_prompt":   sys_prompt,
        "total":           len(samples),
        "train":           len(train),
        "test":            len(test),
        "holdout_convos":  sorted(holdouts),
        "context_turns":   args.context_turns,
        "include_state":   args.include_state,
        "suffix":          args.suffix,
        "by_importance":   dict(Counter(s["importance"] for s in samples)),
        "by_tool_kind":    dict(Counter(s["tool_kind"]  for s in samples)),
    }
    with open(os.path.join(out_dir, f"{prefix}_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[{prefix}] sft={len(train)}  test={len(test)}")
    print(json.dumps(stats, indent=2)[:1000])


if __name__ == "__main__":
    main()
