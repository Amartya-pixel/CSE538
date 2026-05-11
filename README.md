# Persona-Aware Voice Assistant

This project trains a persona-aware assistant that converts spoken or written dialogue into a strict JSON tool decision:

```json
{"importance": "Low|Medium|High", "tool": "allowed tool or null", "detail": "string"}
```

The assistant supports reminders, calendar events, alarms, notes, update/delete calls, clarification requests, and no-action decisions.

## Project Structure

Core training/evaluation files:

```text
parse_json_dataset.py              # Generate SFT/test/stats files from source persona JSON
build_combined_persona_dataset.py  # Combine all personas into all_* datasets
train_sft.py                       # Current-utterance SFT training
train_sft_ctx2.py                  # Previous-2-turn context SFT training
train_sft_state_ctx2.py            # State + previous-2-turn context SFT training
eval.py                            # Evaluation and diagnostic metrics
json_decision.py                   # JSON schema, parser, validator, repair, structured decoding
```

Colab notebooks:

```text
colab_retrain_one_persona.ipynb    # Train/evaluate one persona
colab_retrain_all_personas.ipynb   # Train/evaluate one combined all-persona model
```

Local voice/demo files:

```text
voice_ui.py                        # Safe web UI; logs decisions, does not call actual apps
listening_agent.py                 # MCP tool-calling agent; can call actual app tools
mac_actions_mcp_server.py          # Local MCP server for reminders/calendar/alarms/notes
run_arjun_voice_ui.sh              # Convenience launcher for voice_ui.py
run_arjun_listening_agent.sh       # Convenience launcher for listening_agent.py
```

Persona source data:

```text
arjun_10_conversations_evolving_tool_calls_labeled.json
dev_entrepreneur_10_conversations_evolving_tool_calls_labeled.json
margaret_10_conversations_evolving_tool_calls_labeled.json
neel_neuroscience_10_conversations_evolving_tool_calls_labeled.json
```

## Dataset Variants

For each persona, the pipeline can use three dataset formats:

```text
{persona}_sft.jsonl
{persona}_test.jsonl
{persona}_stats.json
```

Current utterance only.

```text
{persona}_ctx2_sft.jsonl
{persona}_ctx2_test.jsonl
{persona}_ctx2_stats.json
```

Previous 2 dialogue turns + current utterance.

```text
{persona}_state_ctx2_sft.jsonl
{persona}_state_ctx2_test.jsonl
{persona}_state_ctx2_stats.json
```

Previous 2 dialogue turns + existing saved items/state + current utterance.

For all-persona training:

```text
all_sft.jsonl
all_test.jsonl
all_stats.json

all_ctx2_sft.jsonl
all_ctx2_test.jsonl
all_ctx2_stats.json

all_state_ctx2_sft.jsonl
all_state_ctx2_test.jsonl
all_state_ctx2_stats.json
```

The project currently uses a 4-conversation holdout:

```text
held out conversations: 9, 10, 16, 17
training conversations: 14 per persona
test conversations: 4 per persona
```

Conversation 18 stays in training because it contains weak-class examples such as update, delete, alarm, no-action, and clarification.

## Generate JSONL/Stat Files Locally For Colab Option A

If you choose `REGENERATE_JSONL = False` in Colab, the notebook expects JSONL/stat files to already exist. Generate them locally first with these commands.

For one persona, for example Arjun:

```bash
python3 parse_json_dataset.py arjun_10_conversations_evolving_tool_calls_labeled.json arjun \
  --holdout 9 10 16 17

python3 parse_json_dataset.py arjun_10_conversations_evolving_tool_calls_labeled.json arjun \
  --holdout 9 10 16 17 --context-turns 2 --suffix ctx2

python3 parse_json_dataset.py arjun_10_conversations_evolving_tool_calls_labeled.json arjun \
  --holdout 9 10 16 17 --context-turns 2 --include-state --suffix state_ctx2
```

This creates:

```text
arjun_sft.jsonl
arjun_test.jsonl
arjun_stats.json
arjun_ctx2_sft.jsonl
arjun_ctx2_test.jsonl
arjun_ctx2_stats.json
arjun_state_ctx2_sft.jsonl
arjun_state_ctx2_test.jsonl
arjun_state_ctx2_stats.json
```

For all four personas plus combined all-persona files:

```bash
python3 parse_json_dataset.py arjun_10_conversations_evolving_tool_calls_labeled.json arjun --holdout 9 10 16 17
python3 parse_json_dataset.py arjun_10_conversations_evolving_tool_calls_labeled.json arjun --holdout 9 10 16 17 --context-turns 2 --suffix ctx2
python3 parse_json_dataset.py arjun_10_conversations_evolving_tool_calls_labeled.json arjun --holdout 9 10 16 17 --context-turns 2 --include-state --suffix state_ctx2

python3 parse_json_dataset.py dev_entrepreneur_10_conversations_evolving_tool_calls_labeled.json dev --holdout 9 10 16 17
python3 parse_json_dataset.py dev_entrepreneur_10_conversations_evolving_tool_calls_labeled.json dev --holdout 9 10 16 17 --context-turns 2 --suffix ctx2
python3 parse_json_dataset.py dev_entrepreneur_10_conversations_evolving_tool_calls_labeled.json dev --holdout 9 10 16 17 --context-turns 2 --include-state --suffix state_ctx2

python3 parse_json_dataset.py margaret_10_conversations_evolving_tool_calls_labeled.json margaret --holdout 9 10 16 17
python3 parse_json_dataset.py margaret_10_conversations_evolving_tool_calls_labeled.json margaret --holdout 9 10 16 17 --context-turns 2 --suffix ctx2
python3 parse_json_dataset.py margaret_10_conversations_evolving_tool_calls_labeled.json margaret --holdout 9 10 16 17 --context-turns 2 --include-state --suffix state_ctx2

python3 parse_json_dataset.py neel_neuroscience_10_conversations_evolving_tool_calls_labeled.json neel --holdout 9 10 16 17
python3 parse_json_dataset.py neel_neuroscience_10_conversations_evolving_tool_calls_labeled.json neel --holdout 9 10 16 17 --context-turns 2 --suffix ctx2
python3 parse_json_dataset.py neel_neuroscience_10_conversations_evolving_tool_calls_labeled.json neel --holdout 9 10 16 17 --context-turns 2 --include-state --suffix state_ctx2

python3 build_combined_persona_dataset.py
python3 build_combined_persona_dataset.py --suffix ctx2
python3 build_combined_persona_dataset.py --suffix state_ctx2
```

This creates the per-persona files plus:

```text
all_sft.jsonl
all_test.jsonl
all_stats.json
all_ctx2_sft.jsonl
all_ctx2_test.jsonl
all_ctx2_stats.json
all_state_ctx2_sft.jsonl
all_state_ctx2_test.jsonl
all_state_ctx2_stats.json
```

## Colab Training

Set the mode accordingly. To regenerate JSONL/stat files inside Colab from source persona `.json` files.

```python
REGENERATE_JSONL = True
```

In both notebooks, keep these enabled for the full experiment:

```python
RUN_CONTEXT_EXPERIMENT = True
RUN_STATE_CONTEXT_EXPERIMENT = True
```

The notebooks evaluate these rows:

```text
base_current_test
sft_current_test
ctx2_sft_context_test
state_ctx2_sft_context_test
```

### One-Persona Notebook

Open:

```text
colab_retrain_one_persona.ipynb
```

Set the persona:

```python
PERSONA = "arjun"  # or dev, margaret, neel
```

For Option A, upload:

```text
train_sft.py
train_sft_ctx2.py
train_sft_state_ctx2.py
eval.py
json_decision.py

{persona}_sft.jsonl
{persona}_test.jsonl
{persona}_stats.json
{persona}_ctx2_sft.jsonl
{persona}_ctx2_test.jsonl
{persona}_ctx2_stats.json
{persona}_state_ctx2_sft.jsonl
{persona}_state_ctx2_test.jsonl
{persona}_state_ctx2_stats.json
```

For Option B, upload:

```text
parse_json_dataset.py
train_sft.py
train_sft_ctx2.py
train_sft_state_ctx2.py
eval.py
json_decision.py
{persona source json file}
```

The one-persona notebook trains:

```text
{persona}-sft-merged
{persona}-ctx2-sft-merged
{persona}-state-ctx2-sft-merged
```

### All-Persona Notebook

Open:

```text
colab_retrain_all_personas.ipynb
```

For Option A, upload:

```text
train_sft.py
train_sft_ctx2.py
train_sft_state_ctx2.py
eval.py
json_decision.py

all_sft.jsonl
all_test.jsonl
all_stats.json
all_ctx2_sft.jsonl
all_ctx2_test.jsonl
all_ctx2_stats.json
all_state_ctx2_sft.jsonl
all_state_ctx2_test.jsonl
all_state_ctx2_stats.json
```

For Option B, upload:

```text
parse_json_dataset.py
build_combined_persona_dataset.py
train_sft.py
train_sft_ctx2.py
train_sft_state_ctx2.py
eval.py
json_decision.py

arjun_10_conversations_evolving_tool_calls_labeled.json
dev_entrepreneur_10_conversations_evolving_tool_calls_labeled.json
margaret_10_conversations_evolving_tool_calls_labeled.json
neel_neuroscience_10_conversations_evolving_tool_calls_labeled.json
```

The all-persona notebook trains:

```text
all-sft-merged
all-ctx2-sft-merged
all-state-ctx2-sft-merged
```

Use the all-persona state+context model when you want one assistant model that can work with different persona prompts.

## Best Model Choice

For local voice assistant use, prefer the state+context model:

```text
arjun-state-ctx2-sft-merged
```

or, for the all-persona model:

```text
all-state-ctx2-sft-merged
```

The live voice code now sends:

```text
Recent conversation:
...

Existing saved items:
- Reminder: ...
- Calendar: ...
- Alarm: ...
- Note: ...

Current utterance:
...
```

This matches the state+ctx2 training format.

## Convert A Colab Model To GGUF

After Colab training, save or download the merged Hugging Face model folder, for example:

```text
arjun-state-ctx2-sft-merged
```

On a machine with `llama.cpp`, convert it to GGUF:

```bash
python3 llama.cpp/convert_hf_to_gguf.py \
  ./arjun-state-ctx2-sft-merged \
  --outfile arjun-state-ctx2-sft.gguf \
  --outtype f16
```

If you want a smaller file, quantize it:

```bash
llama.cpp/build/bin/llama-quantize \
  arjun-state-ctx2-sft.gguf \
  arjun-state-ctx2-sft-q4_k_m.gguf \
  Q4_K_M
```

Either GGUF can be used with Ollama. The quantized one is smaller and usually better for local use.

## Create An Ollama Model

Install Ollama locally first:

macOS:

```bash
brew install --cask ollama
```

or download it from:

```text
https://ollama.com/download
```

Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start Ollama if it is not already running:

```bash
ollama serve
```

In a second terminal, check that Ollama is available:

```bash
ollama --version
```

Create a Modelfile, for example `Modelfile.arjun-state-ctx2`:

```text
FROM ./arjun-state-ctx2-sft-q4_k_m.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0
```

Create the Ollama model:

```bash
ollama create arjun-state-ctx2-assistant -f Modelfile.arjun-state-ctx2
```

Test it:

```bash
ollama run arjun-state-ctx2-assistant
```

## Run The Local Voice UI

The safe voice UI logs decisions and shows them in the browser. It does not call real apps.

Install local dependencies:

```bash
python3 -m venv .venv-voice
source .venv-voice/bin/activate
pip install flask flask-socketio simple-websocket faster-whisper sounddevice numpy ollama
```

Run:

```bash
PERSONA=arjun \
OLLAMA_MODEL=arjun-state-ctx2-assistant \
python voice_ui.py
```

Then open:

```text
http://127.0.0.1:5050
```

The UI writes decisions to:

```text
actions_log.txt
```

## Run The Local Tool-Calling Agent

`listening_agent.py` can call the MCP tool server. This is the path for actual reminder/calendar/alarm/note tool calls.

Install dependencies:

```bash
source .venv-voice/bin/activate
pip install openai-whisper sounddevice numpy ollama mcp
```

Run:

```bash
PERSONA=arjun \
OLLAMA_MODEL=arjun-state-ctx2-assistant \
python listening_agent.py
```

Behavior:

```text
voice_ui.py        -> logs decisions only
listening_agent.py -> calls MCP tools when available
```

Edit/delete tool calls are routed to the MCP server if the server implements them:

```text
update_reminder
delete_reminder
update_calendar_event
delete_calendar_event
update_alarm
delete_alarm
update_note
delete_note
```

The live assistant maintains lightweight in-session state. It tracks items it creates/updates/deletes during the current run and sends that state back into the model. It does not automatically read every existing item from macOS Calendar/Reminders/Notes.

## Evaluation Metrics

`eval.py` reports:

```text
json_parse_rate
tool_accuracy
importance_accuracy
exact_match_accuracy
json_exact_match_accuracy
action_precision
action_recall
action_f1
detail_exact_match_accuracy
detail_presence_accuracy
detail_token_f1
detail_exact_when_tool_correct
per_tool_precision_recall_f1
tool_confusion_matrix
importance_confusion_matrix
action_vs_no_action_confusion
weak_class_recall
```

The most useful headline metrics are:

```text
tool_accuracy
importance_accuracy
exact_match_accuracy
json_parse_rate
action_f1
detail_token_f1
```

