"""
MCP server exposing the four life-action tools that match Arjun's dataset:
  - create_reminder
  - create_calendar_event
  - create_alarm
  - add_note
  - update_reminder / delete_reminder
  - update_calendar_event / delete_calendar_event
  - update_alarm / delete_alarm
  - update_note / delete_note

For now actions are simulated by appending to local JSONL stores in ./store/.
Later you can wire these to real macOS Calendar / Reminders via AppleScript
(osascript -e 'tell application "Reminders" ...') without touching the MCP
contract that the model has been fine-tuned on.

Run standalone:  python arjun_mcp_server.py
Normally the agent spawns it over stdio.
"""

import datetime as dt
import json
import os

from mcp.server.fastmcp import FastMCP

HERE      = os.path.dirname(os.path.abspath(__file__))
STORE_DIR = os.path.join(HERE, "store")
os.makedirs(STORE_DIR, exist_ok=True)

mcp = FastMCP("arjun-life-actions")


def _append(store_name: str, item: dict) -> str:
    path = os.path.join(STORE_DIR, f"{store_name}.jsonl")
    record = {"timestamp": dt.datetime.now().isoformat(timespec="seconds"), **item}
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return path


@mcp.tool()
def create_reminder(text: str, when: str = "later",
                    priority: str = "Medium") -> str:
    """Save something Arjun shouldn't forget to do at a future moment.
    Example: 'Register for the Sunnyvale tournament tonight'."""
    _append("reminders", {"text": text, "when": when, "priority": priority})
    return f"Reminder saved [{priority}] for {when}: {text}"


@mcp.tool()
def update_reminder(target: str, new_text: str = "", new_when: str = "",
                    priority: str = "Medium") -> str:
    """Record that an existing reminder should be edited."""
    _append("reminder_updates", {
        "target": target, "new_text": new_text,
        "new_when": new_when, "priority": priority,
    })
    return f"Reminder update saved for {target}: {new_text or new_when}"


@mcp.tool()
def delete_reminder(target: str, reason: str = "") -> str:
    """Record that an existing reminder should be removed."""
    _append("reminder_deletes", {"target": target, "reason": reason})
    return f"Reminder delete saved: {target}"


@mcp.tool()
def create_calendar_event(title: str, time: str, location: str = "",
                          description: str = "",
                          priority: str = "Medium") -> str:
    """Reserve a real time block on Arjun's calendar.
    Example: 'Badminton practice, tomorrow 6:30-8 PM, usual gym'."""
    _append("calendar", {
        "title": title, "time": time, "location": location,
        "description": description, "priority": priority,
    })
    return f"Calendar event saved [{priority}]: {title} @ {time}"


@mcp.tool()
def update_calendar_event(target: str, new_title: str = "",
                          new_time: str = "", new_location: str = "",
                          description: str = "",
                          priority: str = "Medium") -> str:
    """Record that an existing calendar event should be edited."""
    _append("calendar_updates", {
        "target": target, "new_title": new_title, "new_time": new_time,
        "new_location": new_location, "description": description,
        "priority": priority,
    })
    return f"Calendar update saved for {target}: {new_title or new_time or new_location}"


@mcp.tool()
def delete_calendar_event(target: str, reason: str = "") -> str:
    """Record that an existing calendar event should be removed."""
    _append("calendar_deletes", {"target": target, "reason": reason})
    return f"Calendar delete saved: {target}"


@mcp.tool()
def create_alarm(time: str, message: str,
                 priority: str = "High") -> str:
    """Set an urgent prompt that fires at a specific moment.
    Example: 'Tonight 9 PM — finish tournament registration'."""
    _append("alarms", {"time": time, "message": message, "priority": priority})
    return f"Alarm saved [{priority}] @ {time}: {message}"


@mcp.tool()
def update_alarm(target: str, new_time: str = "", new_message: str = "",
                 priority: str = "High") -> str:
    """Record that an existing alarm should be edited."""
    _append("alarm_updates", {
        "target": target, "new_time": new_time,
        "new_message": new_message, "priority": priority,
    })
    return f"Alarm update saved for {target}: {new_time or new_message}"


@mcp.tool()
def delete_alarm(target: str, reason: str = "") -> str:
    """Record that an existing alarm should be removed."""
    _append("alarm_deletes", {"target": target, "reason": reason})
    return f"Alarm delete saved: {target}"


@mcp.tool()
def add_note(category: str, content: str,
             priority: str = "Medium") -> str:
    """Save reference material under a topic. Examples of categories:
    'Fitness & Badminton', 'Health & Recovery', 'Shopping Lists',
    'Work — Launch Review'."""
    _append("notes", {
        "category": category, "content": content, "priority": priority,
    })
    return f"Note saved under {category} [{priority}]: {content}"


@mcp.tool()
def update_note(target: str, new_content: str = "",
                category: str = "", priority: str = "Medium") -> str:
    """Record that an existing note should be edited."""
    _append("note_updates", {
        "target": target, "new_content": new_content,
        "category": category, "priority": priority,
    })
    return f"Note update saved for {target}: {new_content}"


@mcp.tool()
def delete_note(target: str, reason: str = "") -> str:
    """Record that an existing note should be removed."""
    _append("note_deletes", {"target": target, "reason": reason})
    return f"Note delete saved: {target}"


if __name__ == "__main__":
    mcp.run()
