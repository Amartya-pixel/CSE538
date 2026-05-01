"""
Persona-aware listening agent:
  mic  ->  Whisper (transcribe)
       ->  YOUR fine-tuned Qwen via Ollama (decide using persona's style)
       ->  MCP tool call (Reminder / Note / Calendar / Alarm)
       ->  TTS confirmation back to you

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
import subprocess
import sys
import numpy as np
import sounddevice as sd
import whisper
import ollama

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ---------------- Config ----------------
PERSONA          = os.environ.get("PERSONA", "dev")
MODEL_NAME       = os.environ.get("OLLAMA_MODEL", f"{PERSONA}-assistant")
WHISPER_SIZE     = "base"
SAMPLE_RATE      = 16_000
CHUNK_SECONDS    = 4
SILENCE_THRESH   = 0.008
TTS_VOICE        = "Samantha"

HERE             = os.path.dirname(os.path.abspath(__file__))
SERVER_SCRIPT    = os.path.join(HERE, "arjun_mcp_server.py")  # the 4-tool server
TEST_JSONL       = os.path.join(HERE, f"{PERSONA}_test.jsonl")


def load_system_prompt() -> str:
    """Read the persona's system prompt from the test JSONL the trainer wrote."""
    if os.path.exists(TEST_JSONL):
        with open(TEST_JSONL) as f:
            return json.loads(f.readline())["system"]
    # Fallback if the test file isn't here.
    return (
        f"You are {PERSONA}'s personal assistant. For each spoken line return "
        'a single JSON object: {"importance":"...","tool":"...","detail":"..."}.'
    )


def speak(text: str) -> None:
    if not text:
        return
    subprocess.run(["say", "-v", TTS_VOICE, text], check=False)


# Map model's tool names to MCP tool names (they should match — keeping
# explicit so a future renamed tool doesn't silently fail).
TOOL_ROUTE = {
    "create_reminder":       "create_reminder",
    "add_note":              "add_note",
    "create_calendar_event": "create_calendar_event",
    "create_alarm":          "create_alarm",
}


# ---------------- Audio + Whisper ----------------
print(f"[boot] persona={PERSONA}  ollama_model={MODEL_NAME}", file=sys.stderr)
print(f"[boot] loading Whisper '{WHISPER_SIZE}' ...", file=sys.stderr)
asr = whisper.load_model(WHISPER_SIZE)
SYSTEM_PROMPT = load_system_prompt()


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


def parse_decision(raw: str):
    """Pull the first {...} JSON object out of the model's output."""
    raw = raw.strip()
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(raw[start:], start=start):
        if ch == "{":   depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:i + 1])
                except Exception:
                    return None
    return None


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

                # The fine-tuned model returns a JSON decision object.
                resp = ollama.chat(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": f"You: {text}"},
                    ],
                    options={"temperature": 0.0},
                )
                raw = resp["message"]["content"]
                decision = parse_decision(raw)
                if decision is None:
                    print(f"  (unparseable: {raw!r})")
                    continue

                tool_name = decision.get("tool")
                if tool_name in (None, "null", ""):
                    print(f"  importance={decision.get('importance')} tool=none — skipping")
                    continue

                mcp_name = TOOL_ROUTE.get(tool_name)
                if mcp_name not in available:
                    print(f"  unknown tool {tool_name!r} — skipping")
                    continue

                # Build minimal args for each MCP tool from the model's `detail`.
                detail = decision.get("detail", "") or ""
                priority = decision.get("importance", "Medium")
                args = _args_for(mcp_name, detail, priority)
                print(f"  -> MCP call: {mcp_name}({args})")
                result = await session.call_tool(mcp_name, args)
                out = "".join(getattr(c, "text", "") for c in result.content)
                print(f"     result: {out}")
                speak(out)


def _args_for(mcp_name: str, detail: str, priority: str) -> dict:
    """Adapt the model's free-text `detail` into each tool's arg shape."""
    if mcp_name == "create_reminder":
        return {"text": detail or "(unspecified)", "when": "later", "priority": priority}
    if mcp_name == "add_note":
        return {"category": "Spoken", "content": detail or "(unspecified)", "priority": priority}
    if mcp_name == "create_calendar_event":
        return {"title": detail or "Event", "time": "TBD", "priority": priority}
    if mcp_name == "create_alarm":
        return {"time": "soon", "message": detail or "Reminder", "priority": priority}
    return {}


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[exit] bye.")
