#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PERSONA=arjun
export OLLAMA_MODEL=arjun-assistant
export XDG_CACHE_HOME="$PWD/.cache"
export HF_HOME="$PWD/.hf_cache"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x ".venv-voice/bin/python" ]; then
  PYTHON_BIN=".venv-voice/bin/python"
fi

"$PYTHON_BIN" listening_agent.py
