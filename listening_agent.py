"""
Continuous listening agent:
  mic  ->  Whisper (transcribe)  ->  Qwen via Ollama (decide)  ->  MCP tool call

The model only triggers a tool when it actually hears the user ask for one
of the two registered actions. Otherwise the audio chunk is ignored.

Run:  python listening_agent.py
Stop: Ctrl+C
"""

import asyncio
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
MODEL_NAME       = "qwen2.5:7b"      # or "qwen2.5:3b" for lighter
WHISPER_SIZE     = "base"            # tiny / base / small / medium
SAMPLE_RATE      = 16_000
CHUNK_SECONDS    = 4
SILENCE_THRESH   = 0.008             # rough RMS-ish gate; tune for your mic
TTS_VOICE        = "Samantha"        # try `say -v ?` in Terminal for the full list
HERE             = os.path.dirname(os.path.abspath(__file__))
SERVER_SCRIPT    = os.path.join(HERE, "mcp_server.py")


def speak(text: str) -> None:
    """Speak `text` aloud through macOS's built-in TTS."""
    if not text:
        return
    subprocess.run(["say", "-v", TTS_VOICE, text], check=False)

SYSTEM_PROMPT = (
    "You are a voice assistant running on macOS. You have exactly two tools: "
    "open_calculator and open_weather. "
    "Call a tool ONLY when the user clearly asks to open the calculator, do math, "
    "check the weather, forecast, or temperature. "
    "For anything else (small talk, background chatter, unrelated speech) reply with "
    "the single word: IGNORE. Do not invent tools. Do not chat."
)


# ---------------- Audio + Whisper ----------------
print(f"[boot] loading Whisper '{WHISPER_SIZE}' ...", file=sys.stderr)
asr = whisper.load_model(WHISPER_SIZE)


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
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.inputSchema or {
                            "type": "object", "properties": {}
                        },
                    },
                }
                for t in tools_result.tools
            ]
            print(f"[boot] MCP tools: {[t['function']['name'] for t in tools]}",
                  file=sys.stderr)
            print("[ready] listening. Speak naturally. Ctrl+C to quit.\n",
                  file=sys.stderr)

            while True:
                audio = record_chunk()
                text  = transcribe(audio)
                if not text:
                    continue
                print(f"heard: {text}")

                resp = ollama.chat(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": text},
                    ],
                    tools=tools,
                )
                msg = resp["message"]
                tool_calls = msg.get("tool_calls") or []

                if not tool_calls:
                    # model said IGNORE or just chatted — drop it
                    continue

                for tc in tool_calls:
                    name = tc["function"]["name"]
                    args = tc["function"].get("arguments") or {}
                    print(f"  -> MCP call: {name}({args})")
                    result = await session.call_tool(name, args)
                    out = "".join(
                        getattr(c, "text", "") for c in result.content
                    )
                    print(f"     result: {out}")
                    # Voice confirmation back to the user.
                    speak(out)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[exit] bye.")
