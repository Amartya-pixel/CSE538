"""
Web UI for the persona-aware listening agent — lightweight version.

Pipeline:
    mic  -> faster-whisper -> Ollama (your fine-tuned model)
         -> decide YES / NO  -> append to actions_log.txt + speak "yes" / "no"

No MCP, no AppleScript subprocess — purely log-and-confirm. Round trip is
just transcribe + generate, typically <1s end-to-end on a warm Ollama.

Run:
    pip install flask flask-socketio simple-websocket faster-whisper
    PERSONA=dev python voice_ui.py

Then open http://127.0.0.1:5050 and watch actions_log.txt:
    tail -f actions_log.txt
"""

import datetime as dt
import json
import os
import subprocess
import sys
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd
import ollama
from faster_whisper import WhisperModel
from flask import Flask
from flask_socketio import SocketIO

# ---------------- Config ----------------
PERSONA          = os.environ.get("PERSONA", "dev")
MODEL_NAME       = os.environ.get("OLLAMA_MODEL", f"{PERSONA}-assistant")
WHISPER_SIZE     = os.environ.get("WHISPER_SIZE", "base")   # tiny / base / small
SAMPLE_RATE      = 16_000
TTS_VOICE        = os.environ.get("TTS_VOICE", "Samantha")

CHUNK_DUR        = 0.1
SILENCE_DBFS     = float(os.environ.get("SILENCE_DBFS", "0.006"))
TRAIL_SILENCE    = float(os.environ.get("TRAIL_SILENCE", "1.0"))
MAX_UTTERANCE    = float(os.environ.get("MAX_UTTERANCE", "20.0"))

HERE             = os.path.dirname(os.path.abspath(__file__))
TEST_JSONL       = os.path.join(HERE, f"{PERSONA}_test.jsonl")
LOG_FILE         = os.path.join(HERE, "actions_log.txt")

# ---------------- Flask + SocketIO -----
app = Flask(__name__)
app.config["SECRET_KEY"] = "dev"
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")
_state = {"running": False}


# ---------------- System prompt --------
def load_system_prompt() -> str:
    if os.path.exists(TEST_JSONL):
        with open(TEST_JSONL) as f:
            return json.loads(f.readline())["system"]
    return (
        f"You are {PERSONA}'s personal assistant. For each spoken line return a "
        'single JSON object: {"importance":"...","tool":"...","detail":"..."}.'
    )


SYSTEM_PROMPT = load_system_prompt()


# ---------------- Helpers --------------
def speak(word: str) -> None:
    subprocess.Popen(["say", "-v", TTS_VOICE, word])


def parse_decision(raw: str) -> Optional[dict]:
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


def decide_yes_no(decision: dict) -> tuple[bool, str]:
    """Return (yes/no, human-readable reason). Pure rule-based, instant.
    Lenient: any concrete tool + non-empty detail counts as YES."""
    tool = decision.get("tool")
    imp  = decision.get("importance", "Low")
    detail = (decision.get("detail") or "").strip()

    if tool in (None, "null", ""):
        return False, "no actionable intent — model returned no tool"
    if not detail:
        return False, f"tool {tool} chosen but detail empty — nothing concrete to record"
    return True, f"importance {imp} with concrete {tool}: {detail}"


def log_action(transcript: str, raw_output: str, decision: dict, yes: bool,
               reason: str, t_asr_ms: int, t_llm_ms: int) -> None:
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    flag = "YES" if yes else "NO "
    block = (
        f"[{ts}]  {flag}  ({t_asr_ms+t_llm_ms} ms total: asr {t_asr_ms} + llm {t_llm_ms})\n"
        f"  heard:     {transcript!r}\n"
        f"  raw_model: {raw_output!r}\n"
        f"  parsed:    importance={decision.get('importance')}  "
                    f"tool={decision.get('tool')}  detail={decision.get('detail','')!r}\n"
        f"  reason:    {reason}\n\n"
    )
    with open(LOG_FILE, "a") as f:
        f.write(block)


# ---------------- Audio + Whisper -----
print(f"[boot] persona={PERSONA}  ollama={MODEL_NAME}", file=sys.stderr)
print(f"[boot] log file: {LOG_FILE}", file=sys.stderr)
print(f"[boot] loading faster-whisper '{WHISPER_SIZE}' (int8 CPU) ...", file=sys.stderr)
asr = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")


def record_utterance() -> Optional[np.ndarray]:
    chunk_samples         = int(CHUNK_DUR * SAMPLE_RATE)
    trail_chunks_needed   = int(TRAIL_SILENCE / CHUNK_DUR)
    max_chunks            = int(MAX_UTTERANCE / CHUNK_DUR)
    pre_speech_max_chunks = int(15.0 / CHUNK_DUR)

    buf = []
    speech_started = False
    silence_run = 0
    pre_silence = 0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                         blocksize=chunk_samples) as stream:
        for _ in range(max_chunks):
            if not _state["running"]:
                return None
            data, _ovf = stream.read(chunk_samples)
            chunk = data.flatten()
            energy = float(np.abs(chunk).mean())

            if energy > SILENCE_DBFS:
                if not speech_started:
                    speech_started = True
                    socketio.emit("status", {"state": "speaking"})
                silence_run = 0
                buf.append(chunk)
            elif speech_started:
                silence_run += 1
                buf.append(chunk)
                if silence_run >= trail_chunks_needed:
                    break
            else:
                pre_silence += 1
                if pre_silence >= pre_speech_max_chunks:
                    return None

    if not speech_started:
        return None
    return np.concatenate(buf)


def transcribe(audio: np.ndarray) -> str:
    segments, _ = asr.transcribe(
        audio, language="en", beam_size=1,
        vad_filter=False, without_timestamps=True,
    )
    return " ".join(s.text for s in segments).strip()


# ---------------- Main listen loop (plain thread, no asyncio) -----
def listen_loop():
    socketio.emit("status", {"state": "listening"})
    while _state["running"]:
        socketio.emit("status", {"state": "listening"})
        audio = record_utterance()
        if not _state["running"]:
            break
        if audio is None:
            continue

        socketio.emit("status", {"state": "thinking"})

        t0 = time.perf_counter()
        text = transcribe(audio)
        t_asr_ms = int((time.perf_counter() - t0) * 1000)
        if not text:
            continue
        socketio.emit("transcribed", {"text": text, "ms": t_asr_ms})

        t0 = time.perf_counter()
        resp = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"You: {text}"},
            ],
            options={"temperature": 0.0, "num_predict": 80, "num_ctx": 1024},
            keep_alive="30m",
        )
        t_llm_ms = int((time.perf_counter() - t0) * 1000)
        raw = resp["message"]["content"]
        decision = parse_decision(raw) or {"importance": "Low", "tool": None, "detail": ""}

        yes, reason = decide_yes_no(decision)
        log_action(text, raw, decision, yes, reason, t_asr_ms, t_llm_ms)

        decision["_timing"] = {"asr_ms": t_asr_ms, "llm_ms": t_llm_ms}
        decision["_raw"]    = raw
        socketio.emit("decision", decision)
        socketio.emit("verdict", {"yes": yes, "reason": reason})
        speak("yes" if yes else "no")

    socketio.emit("status", {"state": "stopped"})


# ---------------- Routes ---------------
@app.route("/")
def index():
    return INDEX_HTML.replace("__PERSONA__", PERSONA).replace("__MODEL__", MODEL_NAME)


@socketio.on("start")
def on_start():
    if _state["running"]:
        return
    _state["running"] = True
    threading.Thread(target=listen_loop, daemon=True).start()


@socketio.on("stop")
def on_stop():
    _state["running"] = False


# ---------------- HTML -----------------
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__PERSONA__'s Assistant</title>
<style>
:root {
  --bg: #0f0f12; --fg: #e7e7ea; --muted: #777; --accent: #4ade80;
  --accent2: #22c55e; --warn: #f59e0b; --bad: #ef4444; --card: #18181b;
  --info: #60a5fa;
}
* { box-sizing: border-box; }
body {
  margin: 0; font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg); color: var(--fg); display: flex; flex-direction: column;
  align-items: center; min-height: 100vh; padding: 30px 20px;
}
h1 { margin: 0 0 4px; font-size: 22px; font-weight: 600; }
.sub { color: var(--muted); font-size: 13px; margin-bottom: 36px; }

.stage { position: relative; width: 280px; height: 280px;
         display: flex; align-items: center; justify-content: center; }
.ring  { position: absolute; border-radius: 50%; opacity: 0; }
.ring1 { width: 200px; height: 200px; background: var(--accent); }
.ring2 { width: 240px; height: 240px; background: var(--accent2); }
.ring3 { width: 280px; height: 280px; background: var(--accent2); }
.circle { width: 160px; height: 160px; border-radius: 50%;
  background: radial-gradient(circle at 30% 30%, #6ee7b7, #16a34a 70%);
  box-shadow: 0 0 60px rgba(34,197,94,.5);
  display: flex; align-items: center; justify-content: center;
  font-size: 13px; color: rgba(255,255,255,.85); font-weight: 600;
  letter-spacing: 1px; transition: all .25s ease; z-index: 2; }

.listening .ring1 { animation: pulse 1.6s ease-out infinite; }
.listening .ring2 { animation: pulse 1.6s ease-out .3s infinite; }
.listening .ring3 { animation: pulse 1.6s ease-out .6s infinite; }
.listening .circle { animation: bob 2s ease-in-out infinite; }

.speaking .ring1 { animation: pulse 0.7s ease-out infinite; }
.speaking .ring2 { animation: pulse 0.7s ease-out .15s infinite; }
.speaking .ring3 { animation: pulse 0.7s ease-out .3s infinite; }
.speaking .circle { background: radial-gradient(circle at 30% 30%, #93c5fd, #2563eb 70%);
                    box-shadow: 0 0 80px rgba(37,99,235,.7); transform: scale(1.08); }

.thinking .circle { background: radial-gradient(circle at 30% 30%, #fde68a, #d97706 70%);
                    box-shadow: 0 0 60px rgba(245,158,11,.6); animation: spin 1.2s linear infinite; }
.idle .circle, .stopped .circle {
  background: radial-gradient(circle at 30% 30%, #555, #2a2a2a 70%); box-shadow: none; }

@keyframes pulse { 0% { transform: scale(.6); opacity: .5; } 100% { transform: scale(1.2); opacity: 0; } }
@keyframes bob   { 0%,100% { transform: scale(1); } 50% { transform: scale(1.04); } }
@keyframes spin  { to { transform: rotate(360deg); } }

.status { margin: 18px 0 26px; color: var(--muted); font-size: 13px;
          letter-spacing: 1px; text-transform: uppercase; }

.controls { display: flex; gap: 12px; margin-bottom: 28px; }
button { background: var(--accent2); color: white; border: none;
  padding: 11px 26px; border-radius: 999px; font-weight: 600;
  font-size: 14px; cursor: pointer; transition: background .15s; }
button:hover  { background: var(--accent); }
button.stop   { background: #444; }
button.stop:hover { background: var(--bad); }
button:disabled { opacity: .4; cursor: not-allowed; }

.feed { width: 100%; max-width: 700px; }
.entry { background: var(--card); border-radius: 10px; padding: 14px 16px;
         margin-bottom: 10px; border-left: 3px solid var(--muted); }
.entry .who   { color: var(--muted); font-size: 11px; text-transform: uppercase;
                letter-spacing: 1px; margin-bottom: 4px; }
.entry .what  { font-size: 14px; }
.entry.heard  { border-left-color: var(--info); }
.entry.decide { border-left-color: var(--warn); }
.entry.yes    { border-left-color: var(--accent); }
.entry.no     { border-left-color: var(--bad); }
.tag { display: inline-block; padding: 2px 8px; border-radius: 4px;
       font-size: 11px; font-weight: 600; margin-right: 6px; }
.tag.High   { background: rgba(239,68,68,.2);  color: #fca5a5; }
.tag.Medium { background: rgba(245,158,11,.2); color: #fcd34d; }
.tag.Low    { background: rgba(120,120,120,.2); color: #aaa; }
.tag.tool   { background: rgba(34,197,94,.2);  color: #86efac; }
.tag.null   { background: rgba(120,120,120,.2); color: #aaa; }
.verdict { font-size: 26px; font-weight: 700; letter-spacing: 1px; margin-bottom: 4px; }
.verdict.yes { color: var(--accent); }
.verdict.no  { color: var(--bad); }
</style>
</head>
<body>
  <h1>__PERSONA__'s personal assistant</h1>
  <div class="sub">model: <code>__MODEL__</code> · log: <code>actions_log.txt</code></div>

  <div class="stage idle" id="stage">
    <div class="ring ring1"></div>
    <div class="ring ring2"></div>
    <div class="ring ring3"></div>
    <div class="circle" id="circle">IDLE</div>
  </div>
  <div class="status" id="status">click start</div>

  <div class="controls">
    <button id="start">▶ Start listening</button>
    <button id="stop" class="stop" disabled>■ Stop</button>
  </div>

  <div class="feed" id="feed"></div>

<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
<script>
const stage = document.getElementById('stage');
const circle = document.getElementById('circle');
const statusEl = document.getElementById('status');
const feed = document.getElementById('feed');
const startBtn = document.getElementById('start');
const stopBtn  = document.getElementById('stop');
const socket = io();

const labels = {
  listening: ['LISTENING','speak naturally — i wait for you to finish'],
  speaking:  ['HEARING YOU','keep going — i stop when you pause'],
  thinking:  ['THINKING','transcribing & deciding'],
  idle:      ['IDLE','click start'],
  stopped:   ['STOPPED','stopped'],
};
function setState(s) {
  stage.className = 'stage ' + s;
  const [c, st] = labels[s] || [s.toUpperCase(), s];
  circle.textContent = c;
  statusEl.textContent = st;
}

function addEntry(cls, who, html) {
  const e = document.createElement('div');
  e.className = 'entry ' + cls;
  e.innerHTML = `<div class="who">${who}</div><div class="what">${html}</div>`;
  feed.prepend(e);
  while (feed.children.length > 50) feed.removeChild(feed.lastChild);
}

socket.on('status', d => setState(d.state));

socket.on('transcribed', d => {
  const ms = d.ms ? ` <span style="color:#888;font-size:11px">(asr ${d.ms}ms)</span>` : '';
  addEntry('heard', 'heard', `“${d.text}”${ms}`);
});

socket.on('decision', d => {
  const imp  = d.importance || 'Low';
  const tool = d.tool || 'null';
  const tcls = tool === 'null' ? 'null' : 'tool';
  const det  = d.detail ? ` — <em style="color:#aaa">${d.detail}</em>` : '';
  const t    = d._timing
    ? ` <span style="color:#888;font-size:11px">(asr ${d._timing.asr_ms}ms · llm ${d._timing.llm_ms}ms)</span>`
    : '';
  const raw  = d._raw
    ? `<div style="margin-top:6px;color:#666;font-size:11px;font-family:Menlo,monospace">raw: ${d._raw.replace(/</g,'&lt;').slice(0,200)}</div>`
    : '';
  addEntry('decide', 'decision',
    `<span class="tag ${imp}">${imp}</span><span class="tag ${tcls}">${tool}</span>${det}${t}${raw}`);
});

socket.on('verdict', d => {
  const cls   = d.yes ? 'yes' : 'no';
  const label = d.yes ? 'YES' : 'NO';
  addEntry(cls, 'verdict',
    `<div class="verdict ${cls}">${label}</div>` +
    `<div style="color:#bbb">reason: ${d.reason}</div>`);
});

startBtn.onclick = () => {
  socket.emit('start');
  startBtn.disabled = true; stopBtn.disabled = false;
  setState('listening');
};
stopBtn.onclick = () => {
  socket.emit('stop');
  startBtn.disabled = false; stopBtn.disabled = true;
};
</script>
</body>
</html>"""


if __name__ == "__main__":
    print(f"[ready] open http://127.0.0.1:5050")
    socketio.run(app, host="127.0.0.1", port=5050, allow_unsafe_werkzeug=True)
