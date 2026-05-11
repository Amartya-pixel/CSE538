#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PERSONA=arjun
export OLLAMA_MODEL=arjun-assistant
export HF_TOKEN_PATH="/tmp/hf_dummy_token"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HOME="$PWD/.hf_cache"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x ".venv-voice/bin/python" ]; then
  PYTHON_BIN=".venv-voice/bin/python"
fi

"$PYTHON_BIN" voice_ui.py
