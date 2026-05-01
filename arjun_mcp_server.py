"""
MCP server exposing the four life-action tools that match Arjun's dataset:
  - create_reminder
  - create_calendar_event
  - create_alarm
  - add_note

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
def create_alarm(time: str, message: str,
                 priority: str = "High") -> str:
    """Set an urgent prompt that fires at a specific moment.
    Example: 'Tonight 9 PM — finish tournament registration'."""
    _append("alarms", {"time": time, "message": message, "priority": priority})
    return f"Alarm saved [{priority}] @ {time}: {message}"


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


if __name__ == "__main__":
    mcp.run()
