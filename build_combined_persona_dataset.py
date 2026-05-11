"""
Build one persona-aware training set from all generated persona JSONL files.

The combined model sees the same universal tool definitions across personas,
while each row's system prompt carries that persona's importance preferences.

Usage:
    python3 build_combined_persona_dataset.py
    python3 build_combined_persona_dataset.py --suffix ctx2

Outputs:
    all_sft.jsonl
    all_test.jsonl
    all_stats.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path


PERSONAS = ["arjun", "dev", "margaret", "neel"]
HERE = Path(__file__).resolve().parent


def read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def add_persona(row: dict, persona: str) -> dict:
    row = dict(row)
    row["persona"] = persona
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="",
                    help="Optional persona file suffix, e.g. ctx2.")
    args = ap.parse_args()
    suffix = args.suffix
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    out_prefix = f"all{suffix}"

    sft_rows = []
    test_rows = []

    for persona in PERSONAS:
        sft_path = HERE / f"{persona}{suffix}_sft.jsonl"
        test_path = HERE / f"{persona}{suffix}_test.jsonl"

        sft_rows.extend(add_persona(row, persona) for row in read_jsonl(sft_path))
        test_rows.extend(add_persona(row, persona) for row in read_jsonl(test_path))

    write_jsonl(HERE / f"{out_prefix}_sft.jsonl", sft_rows)
    write_jsonl(HERE / f"{out_prefix}_test.jsonl", test_rows)

    tool_counts = Counter()
    importance_counts = Counter()
    persona_counts = Counter()
    for row in test_rows + sft_rows:
        persona_counts[row["persona"]] += 1
        if "messages" in row:
            target = json.loads(row["messages"][2]["content"])
        else:
            target = row["gold"]
        tool_counts[target["tool"] or "None"] += 1
        importance_counts[target["importance"]] += 1

    stats = {
        "personas": PERSONAS,
        "suffix": suffix,
        "sft": len(sft_rows),
        "test": len(test_rows),
        "by_persona_total_sft_plus_test": dict(persona_counts),
        "by_tool_sft_plus_test": dict(tool_counts),
        "by_importance_sft_plus_test": dict(importance_counts),
    }
    with (HERE / f"{out_prefix}_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
