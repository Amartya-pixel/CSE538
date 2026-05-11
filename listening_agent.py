"""
Persona-aware listening agent:
  mic  ->  Whisper (transcribe)
       ->  YOUR fine-tuned Qwen via Ollama (decide using persona's style)
       ->  MCP tool call (Reminder / Note / Calendar / Alarm)
       ->  terminal notification/log only

Set PERSONA via env var or constants below. The agent loads:
  - persona-specific Ollama model (e.g. 'dev-assistant' you registered earlier)
  - persona's system prompt (read from <persona>_test.jsonl which the trainer
    saved alongside the model — every test record carries the same prompt)

Run:  PERSONA=dev python listening_agent.py
Stop: Ctrl+C
"""

import asyncio
import json
import os
import sys
from collections import deque
import numpy as np
import sounddevice as sd
import whisper
import ollama

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from json_decision import decide_with_json_guardrails

# ---------------- Config ----------------
PERSONA          = os.environ.get("PERSONA", "dev")
MODEL_NAME       = os.environ.get("OLLAMA_MODEL", f"{PERSONA}-assistant")
WHISPER_SIZE     = "base"
SAMPLE_RATE      = 16_000
CHUNK_SECONDS    = 4
SILENCE_THRESH   = 0.008

HERE             = os.path.dirname(os.path.abspath(__file__))
SERVER_SCRIPT    = os.environ.get(
    "MCP_SERVER_SCRIPT",
    os.path.join(HERE, "mac_actions_mcp_server.py"),
)
TEST_JSONL       = os.path.join(HERE, f"{PERSONA}_test.jsonl")


def load_system_prompt() -> str:
    """Read the persona's system prompt from the test JSONL the trainer wrote."""
    if os.path.exists(TEST_JSONL):
        with open(TEST_JSONL) as f:
            return json.loads(f.readline())["system"]
    # Fallback if the test file isn't here.
    return (
        f"You are {PERSONA}'s personal assistant. For each spoken line return "
        'exactly one valid JSON object: '
        '{"importance":"Low|Medium|High","tool":"allowed tool or null",'
        '"detail":"string"}.'
    )


def notify(text: str) -> None:
    if text:
        print(f"     notification: {text}")


# Map model's tool names to MCP tool names (they should match — keeping
# explicit so a future renamed tool doesn't silently fail).
TOOL_ROUTE = {
    "create_reminder":        "create_reminder",
    "update_reminder":        "update_reminder",
    "delete_reminder":        "delete_reminder",
    "add_note":               "add_note",
    "update_note":            "update_note",
    "delete_note":            "delete_note",
    "create_calendar_event":  "create_calendar_event",
    "update_calendar_event":  "update_calendar_event",
    "delete_calendar_event":  "delete_calendar_event",
    "create_alarm":           "create_alarm",
    "update_alarm":           "update_alarm",
    "delete_alarm":           "delete_alarm",
}


# ---------------- Audio + Whisper ----------------
print(f"[boot] persona={PERSONA}  ollama_model={MODEL_NAME}", file=sys.stderr)
print(f"[boot] loading Whisper '{WHISPER_SIZE}' ...", file=sys.stderr)
asr = whisper.load_model(WHISPER_SIZE)
SYSTEM_PROMPT = load_system_prompt()
RECENT_TURNS = deque(maxlen=2)
SAVED_ITEMS = []


def state_label(tool_name: str) -> str:
    if "reminder" in tool_name:
        return "Reminder"
    if "calendar_event" in tool_name:
        return "Calendar"
    if "alarm" in tool_name:
        return "Alarm"
    if "note" in tool_name:
        return "Note"
    return ""


def format_model_input(current_text: str) -> str:
    parts = []
    if RECENT_TURNS:
        context_lines = "\n".join(
            f"{turn['speaker']}: {turn['text']}" for turn in RECENT_TURNS
        )
        parts.append(f"Recent conversation:\n{context_lines}")
    if SAVED_ITEMS:
        state_lines = "\n".join(
            f"- {item['type']}: {item['detail']}" for item in SAVED_ITEMS
        )
        parts.append(f"Existing saved items:\n{state_lines}")
    speaker = PERSONA.capitalize()
    parts.append(f"Current utterance:\n{speaker}: {current_text}")
    return "\n\n".join(parts)


def remember_turn(text: str) -> None:
    RECENT_TURNS.append({"speaker": PERSONA.capitalize(), "text": text})


def apply_state_update(decision: dict) -> None:
    tool = decision.get("tool")
    detail = (decision.get("detail") or "").strip()
    if tool in (None, "null", "", "ask_clarification") or not detail:
        return
    label = state_label(tool)
    if not label:
        return
    if tool.startswith("delete_"):
        for i, item in enumerate(SAVED_ITEMS):
            if item["type"] == label:
                del SAVED_ITEMS[i]
                return
        return
    if tool.startswith("update_"):
        for item in reversed(SAVED_ITEMS):
            if item["type"] == label:
                item["detail"] = detail
                return
    SAVED_ITEMS.append({"type": label, "detail": detail})


def record_chunk() -> np.ndarray:
    audio = sd.rec(
        int(CHUNK_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()


def transcribe(audio: np.ndarray) -> str:
    if float(np.abs(audio).mean()) < SILENCE_THRESH:
        return ""
    result = asr.transcribe(audio, fp16=False, language="en")
    return result.get("text", "").strip()


# ---------------- LLM + MCP loop ----------------
async def main() -> None:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_SCRIPT],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            available = {t.name for t in tools_result.tools}
            print(f"[boot] MCP tools: {sorted(available)}", file=sys.stderr)
            print("[ready] listening. Speak naturally. Ctrl+C to quit.\n",
                  file=sys.stderr)

            while True:
                audio = record_chunk()
                text  = transcribe(audio)
                if not text:
                    continue
                print(f"heard: {text}")

                # The fine-tuned model returns a JSON decision object. We use
                # Ollama structured output when available, then validate/repair.
                model_input = format_model_input(text)
                decision, raw, parse_status = decide_with_json_guardrails(
                    ollama,
                    model=MODEL_NAME,
                    system_prompt=SYSTEM_PROMPT,
                    user_text=model_input,
                    options={"temperature": 0.0},
                )
                if parse_status != "schema":
                    print(f"  json_guardrail={parse_status}")

                tool_name = decision.get("tool")
                if tool_name in (None, "null", ""):
                    print(f"  importance={decision.get('importance')} tool=none — skipping")
                    remember_turn(text)
                    continue
                if tool_name == "ask_clarification":
                    detail = decision.get("detail", "") or "clarification needed"
                    notify(f"Clarification needed: {detail}")
                    remember_turn(text)
                    continue

                mcp_name = TOOL_ROUTE.get(tool_name)
                if mcp_name not in available:
                    print(f"  unknown tool {tool_name!r} — skipping")
                    remember_turn(text)
                    continue

                # Build minimal args for each MCP tool from the model's `detail`.
                detail = decision.get("detail", "") or ""
                priority = decision.get("importance", "Medium")
                args = _args_for(mcp_name, detail, priority)
                print(f"  -> MCP call: {mcp_name}({args})")
                result = await session.call_tool(mcp_name, args)
                out = "".join(getattr(c, "text", "") for c in result.content)
                print(f"     result: {out}")
                apply_state_update(decision)
                remember_turn(text)
                print(f"     state: {SAVED_ITEMS}")
                notify(out)


def _args_for(mcp_name: str, detail: str, priority: str) -> dict:
    """Adapt the model's free-text `detail` into each tool's arg shape."""
    if mcp_name == "create_reminder":
        return {"text": detail or "(unspecified)", "when": "later", "priority": priority}
    if mcp_name == "update_reminder":
        return {"target": detail or "(unspecified)", "new_text": detail, "priority": priority}
    if mcp_name == "delete_reminder":
        return {"target": detail or "(unspecified)"}
    if mcp_name == "add_note":
        return {"category": "Spoken", "content": detail or "(unspecified)", "priority": priority}
    if mcp_name == "update_note":
        return {"target": detail or "(unspecified)", "new_content": detail, "priority": priority}
    if mcp_name == "delete_note":
        return {"target": detail or "(unspecified)"}
    if mcp_name == "create_calendar_event":
        return {"title": detail or "Event", "time": "TBD", "priority": priority}
    if mcp_name == "update_calendar_event":
        return {"target": detail or "(unspecified)", "new_title": detail, "priority": priority}
    if mcp_name == "delete_calendar_event":
        return {"target": detail or "(unspecified)"}
    if mcp_name == "create_alarm":
        return {"time": "soon", "message": detail or "Reminder", "priority": priority}
    if mcp_name == "update_alarm":
        return {"target": detail or "(unspecified)", "new_message": detail, "priority": priority}
    if mcp_name == "delete_alarm":
        return {"target": detail or "(unspecified)"}
    return {}


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[exit] bye.")
