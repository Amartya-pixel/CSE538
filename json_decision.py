import json
from typing import Any


ALLOWED_IMPORTANCE = {"Low", "Medium", "High"}
ALLOWED_TOOLS = {
    "create_reminder",
    "update_reminder",
    "delete_reminder",
    "add_note",
    "update_note",
    "delete_note",
    "create_calendar_event",
    "update_calendar_event",
    "delete_calendar_event",
    "create_alarm",
    "update_alarm",
    "delete_alarm",
    "ask_clarification",
}

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "importance": {"type": "string", "enum": sorted(ALLOWED_IMPORTANCE)},
        "tool": {
            "anyOf": [
                {"type": "string", "enum": sorted(ALLOWED_TOOLS)},
                {"type": "null"},
            ],
        },
        "detail": {"type": "string"},
    },
    "required": ["importance", "tool", "detail"],
    "additionalProperties": False,
}

STRICT_JSON_INSTRUCTIONS = (
    "\n\nOutput contract:\n"
    "- Respond with exactly one valid JSON object and nothing else.\n"
    "- Do not use markdown, code fences, comments, or explanations.\n"
    "- Use exactly these keys: importance, tool, detail.\n"
    "- importance must be one of: Low, Medium, High.\n"
    "- tool must be one of the allowed tool names or null.\n"
    "- detail must be a string; use an empty string when tool is null.\n"
)

SAFE_DECISION = {"importance": "Low", "tool": None, "detail": ""}


def strict_system_prompt(system_prompt: str) -> str:
    if "Output contract:" in system_prompt:
        return system_prompt
    return system_prompt.rstrip() + STRICT_JSON_INSTRUCTIONS


def extract_json_object(raw: str) -> dict[str, Any] | None:
    raw = (raw or "").strip()
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i, ch in enumerate(raw[start:], start=start):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(raw[start:i + 1])
                except Exception:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def validate_decision(obj: Any) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None

    importance = obj.get("importance")
    tool = obj.get("tool")
    detail = obj.get("detail")

    if importance not in ALLOWED_IMPORTANCE:
        return None
    if tool in ("", "null"):
        tool = None
    if tool is not None and tool not in ALLOWED_TOOLS:
        return None
    if detail is None:
        detail = ""
    if not isinstance(detail, str):
        detail = str(detail)
    if tool is None:
        detail = ""

    return {"importance": importance, "tool": tool, "detail": detail}


def parse_decision(raw: str) -> dict[str, Any] | None:
    return validate_decision(extract_json_object(raw))


def _chat(ollama_module, *, model: str, messages: list[dict[str, str]],
          options: dict[str, Any], keep_alive: str | None = None,
          use_schema: bool = True):
    kwargs = {
        "model": model,
        "messages": messages,
        "options": options,
    }
    if keep_alive is not None:
        kwargs["keep_alive"] = keep_alive
    if use_schema:
        kwargs["format"] = DECISION_SCHEMA
    try:
        return ollama_module.chat(**kwargs)
    except Exception:
        if not use_schema:
            raise
        kwargs["format"] = "json"
        try:
            return ollama_module.chat(**kwargs)
        except Exception:
            kwargs.pop("format", None)
            return ollama_module.chat(**kwargs)


def decide_with_json_guardrails(
    ollama_module,
    *,
    model: str,
    system_prompt: str,
    user_text: str,
    options: dict[str, Any] | None = None,
    keep_alive: str | None = None,
) -> tuple[dict[str, Any], str, str]:
    options = {"temperature": 0.0, **(options or {})}
    messages = [
        {"role": "system", "content": strict_system_prompt(system_prompt)},
        {"role": "user", "content": user_text},
    ]
    resp = _chat(
        ollama_module, model=model, messages=messages,
        options=options, keep_alive=keep_alive, use_schema=True,
    )
    raw = resp["message"]["content"]
    decision = parse_decision(raw)
    if decision is not None:
        return decision, raw, "schema"

    repair_messages = [
        {
            "role": "system",
            "content": (
                "Convert the provided text into exactly one valid JSON object. "
                "Use this schema only: "
                '{"importance":"Low|Medium|High","tool":"allowed tool name or null",'
                '"detail":"string"}. Return JSON only.'
            ),
        },
        {"role": "user", "content": raw},
    ]
    repair = _chat(
        ollama_module, model=model, messages=repair_messages,
        options={"temperature": 0.0, "num_predict": 80},
        keep_alive=keep_alive, use_schema=True,
    )
    repaired_raw = repair["message"]["content"]
    decision = parse_decision(repaired_raw)
    if decision is not None:
        return decision, raw + "\n[repair]\n" + repaired_raw, "repair"

    return dict(SAFE_DECISION), raw + "\n[repair_failed]\n" + repaired_raw, "fallback"
