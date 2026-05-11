"""
MCP server that actually creates real macOS Reminders / Calendar events /
Notes / Alarms via AppleScript. Replaces the stub `arjun_mcp_server.py` which
just appended to JSONL files.

First run: macOS will prompt for permission to control Reminders, Calendar,
and Notes. Allow each one (System Settings → Privacy & Security → Automation).
Without those grants the AppleScript calls will silently fail.

Run standalone (for debugging):  python mac_actions_mcp_server.py
Normally the listening agent / web UI spawns it over stdio.
"""

import datetime as dt
import re
import subprocess

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mac-real-actions")


# ---------- AppleScript helpers --------------------------------------------
def _osa(script: str) -> tuple[bool, str]:
    """Run AppleScript via osascript. Returns (ok, output_or_error)."""
    p = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True, text=True, timeout=10,
    )
    if p.returncode != 0:
        return False, (p.stderr or "").strip()
    return True, (p.stdout or "").strip()


def _esc(s: str) -> str:
    """Escape a string for safe inclusion in an AppleScript string literal."""
    return (s or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _applescript_date(d: dt.datetime) -> str:
    """AppleScript date literal format: 'Monday, January 1, 2026 at 10:00:00 AM'."""
    return d.strftime("%A, %B %d, %Y at %I:%M:%S %p")


def _parse_when(detail: str) -> dt.datetime:
    """Best-effort time parsing from the model's free-text detail string."""
    now = dt.datetime.now().replace(second=0, microsecond=0)
    text = (detail or "").lower()

    base = now
    if "tomorrow" in text:
        base = now + dt.timedelta(days=1)
    elif "tonight" in text:
        base = now.replace(hour=20, minute=0)
    elif "next week" in text:
        base = now + dt.timedelta(days=7)

    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?\b", text)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        ampm = (m.group(3) or "").replace(".", "")
        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        base = base.replace(hour=hour, minute=minute)
    elif base == now:
        # No time found and base is "now" — push to 1 hour from now
        base = now + dt.timedelta(hours=1)
    return base


def _first_nonempty(*values: str) -> str:
    for value in values:
        if value:
            return value
    return ""


# ---------- MCP tools ------------------------------------------------------
@mcp.tool()
def create_reminder(text: str, when: str = "later",
                    priority: str = "Medium") -> str:
    """Create a reminder in macOS Reminders.app.
    Use when the user wants to be nudged to do something later."""
    body = _esc(f"{text} (priority: {priority})")
    script = f'''
        tell application "Reminders"
            set defList to default list
            tell defList
                make new reminder with properties {{name:"{body}"}}
            end tell
        end tell
    '''
    ok, msg = _osa(script)
    if not ok:
        return f"Failed to add reminder: {msg}"
    return f"Reminder added: {text}"


@mcp.tool()
def update_reminder(target: str, new_text: str = "", new_when: str = "",
                    priority: str = "Medium") -> str:
    """Edit the first reminder whose title contains the target text."""
    target_e = _esc(target)
    new_text_e = _esc(new_text or target)
    date_clause = ""
    if new_when:
        date_clause = f', remind me date:date "{_applescript_date(_parse_when(new_when))}"'
    script = f'''
        tell application "Reminders"
            repeat with r in reminders
                if name of r contains "{target_e}" then
                    set name of r to "{new_text_e} (priority: {priority})"
                    {f"set remind me date of r to date \"{_applescript_date(_parse_when(new_when))}\"" if new_when else ""}
                    return "updated"
                end if
            end repeat
            return "not found"
        end tell
    '''
    ok, msg = _osa(script)
    if not ok:
        return f"Failed to update reminder: {msg}"
    return f"Reminder update result for {target}: {msg}"


@mcp.tool()
def delete_reminder(target: str, reason: str = "") -> str:
    """Delete the first reminder whose title contains the target text."""
    target_e = _esc(target)
    script = f'''
        tell application "Reminders"
            repeat with r in reminders
                if name of r contains "{target_e}" then
                    delete r
                    return "deleted"
                end if
            end repeat
            return "not found"
        end tell
    '''
    ok, msg = _osa(script)
    if not ok:
        return f"Failed to delete reminder: {msg}"
    return f"Reminder delete result for {target}: {msg}"


@mcp.tool()
def create_calendar_event(title: str, time: str = "tomorrow 10am",
                          location: str = "",
                          description: str = "",
                          priority: str = "Medium") -> str:
    """Create an event on macOS Calendar.app.
    Use when the user wants to reserve a real time block."""
    start = _parse_when(time)
    end = start + dt.timedelta(hours=1)
    title_e   = _esc(title)
    loc_e     = _esc(location)
    desc_e    = _esc(description or f"priority: {priority}")
    start_str = _applescript_date(start)
    end_str   = _applescript_date(end)
    script = f'''
        tell application "Calendar"
            set writableCals to (every calendar whose writable is true)
            if (count of writableCals) is 0 then return "no writable calendar"
            tell first item of writableCals
                make new event with properties {{summary:"{title_e}", start date:date "{start_str}", end date:date "{end_str}", location:"{loc_e}", description:"{desc_e}"}}
            end tell
        end tell
    '''
    ok, msg = _osa(script)
    if not ok:
        return f"Failed to add calendar event: {msg}"
    return f"Calendar event added: {title} at {start.strftime('%a %b %d, %I:%M %p')}"


@mcp.tool()
def update_calendar_event(target: str, new_title: str = "",
                          new_time: str = "", new_location: str = "",
                          description: str = "",
                          priority: str = "Medium") -> str:
    """Edit the first future calendar event whose summary contains target."""
    target_e = _esc(target)
    title_e = _esc(new_title)
    loc_e = _esc(new_location)
    desc_e = _esc(description or f"priority: {priority}")
    date_lines = ""
    if new_time:
        start = _parse_when(new_time)
        end = start + dt.timedelta(hours=1)
        date_lines = (
            f'set start date of ev to date "{_applescript_date(start)}"\n'
            f'                    set end date of ev to date "{_applescript_date(end)}"'
        )
    script = f'''
        tell application "Calendar"
            set cutoff to current date
            repeat with cal in calendars
                repeat with ev in (events of cal whose start date is greater than cutoff)
                    if summary of ev contains "{target_e}" then
                        {f'set summary of ev to "{title_e}"' if new_title else ""}
                        {date_lines}
                        {f'set location of ev to "{loc_e}"' if new_location else ""}
                        {f'set description of ev to "{desc_e}"' if description else ""}
                        return "updated"
                    end if
                end repeat
            end repeat
            return "not found"
        end tell
    '''
    ok, msg = _osa(script)
    if not ok:
        return f"Failed to update calendar event: {msg}"
    return f"Calendar update result for {target}: {msg}"


@mcp.tool()
def delete_calendar_event(target: str, reason: str = "") -> str:
    """Delete the first future calendar event whose summary contains target."""
    target_e = _esc(target)
    script = f'''
        tell application "Calendar"
            set cutoff to current date
            repeat with cal in calendars
                repeat with ev in (events of cal whose start date is greater than cutoff)
                    if summary of ev contains "{target_e}" then
                        delete ev
                        return "deleted"
                    end if
                end repeat
            end repeat
            return "not found"
        end tell
    '''
    ok, msg = _osa(script)
    if not ok:
        return f"Failed to delete calendar event: {msg}"
    return f"Calendar delete result for {target}: {msg}"


@mcp.tool()
def create_alarm(time: str = "in 1 hour", message: str = "Reminder",
                 priority: str = "High") -> str:
    """Set an urgent prompt at a specific moment.
    Implemented as a Reminder with a fixed alert time."""
    when_dt = _parse_when(time)
    body = _esc(f"[ALARM] {message}")
    when_str = _applescript_date(when_dt)
    script = f'''
        tell application "Reminders"
            set defList to default list
            tell defList
                make new reminder with properties {{name:"{body}", remind me date:date "{when_str}"}}
            end tell
        end tell
    '''
    ok, msg = _osa(script)
    if not ok:
        return f"Failed to set alarm: {msg}"
    return f"Alarm set for {when_dt.strftime('%a %b %d, %I:%M %p')}: {message}"


@mcp.tool()
def update_alarm(target: str, new_time: str = "", new_message: str = "",
                 priority: str = "High") -> str:
    """Edit an alarm implemented as a Reminders reminder with an alert time."""
    target_e = _esc(target)
    new_message_e = _esc(f"[ALARM] {new_message or target}")
    date_line = ""
    if new_time:
        date_line = f'set remind me date of r to date "{_applescript_date(_parse_when(new_time))}"'
    script = f'''
        tell application "Reminders"
            repeat with r in reminders
                if name of r contains "{target_e}" then
                    set name of r to "{new_message_e}"
                    {date_line}
                    return "updated"
                end if
            end repeat
            return "not found"
        end tell
    '''
    ok, msg = _osa(script)
    if not ok:
        return f"Failed to update alarm: {msg}"
    return f"Alarm update result for {target}: {msg}"


@mcp.tool()
def delete_alarm(target: str, reason: str = "") -> str:
    """Delete an alarm implemented as a Reminders reminder."""
    target_e = _esc(target)
    script = f'''
        tell application "Reminders"
            repeat with r in reminders
                if name of r contains "{target_e}" then
                    delete r
                    return "deleted"
                end if
            end repeat
            return "not found"
        end tell
    '''
    ok, msg = _osa(script)
    if not ok:
        return f"Failed to delete alarm: {msg}"
    return f"Alarm delete result for {target}: {msg}"


@mcp.tool()
def add_note(category: str, content: str,
             priority: str = "Medium") -> str:
    """Save reference material to macOS Notes.app under a topical title."""
    title_e = _esc(f"[{category}] {content[:60]}")
    body_e  = _esc(f"{content}\n\nPriority: {priority}\nCategory: {category}")
    script = f'''
        tell application "Notes"
            make new note with properties {{name:"{title_e}", body:"{body_e}"}}
        end tell
    '''
    ok, msg = _osa(script)
    if not ok:
        return f"Failed to add note: {msg}"
    return f"Note saved under {category}: {content[:60]}"


@mcp.tool()
def update_note(target: str, new_content: str = "",
                category: str = "", priority: str = "Medium") -> str:
    """Edit the first note whose name contains target."""
    target_e = _esc(target)
    title_e = _esc(f"[{category or 'Updated'}] {target}"[:80])
    body_e = _esc(f"{new_content}\n\nPriority: {priority}\nCategory: {category}")
    script = f'''
        tell application "Notes"
            repeat with n in notes
                if name of n contains "{target_e}" then
                    {f'set name of n to "{title_e}"' if category else ""}
                    set body of n to "{body_e}"
                    return "updated"
                end if
            end repeat
            return "not found"
        end tell
    '''
    ok, msg = _osa(script)
    if not ok:
        return f"Failed to update note: {msg}"
    return f"Note update result for {target}: {msg}"


@mcp.tool()
def delete_note(target: str, reason: str = "") -> str:
    """Delete the first note whose name contains target."""
    target_e = _esc(target)
    script = f'''
        tell application "Notes"
            repeat with n in notes
                if name of n contains "{target_e}" then
                    delete n
                    return "deleted"
                end if
            end repeat
            return "not found"
        end tell
    '''
    ok, msg = _osa(script)
    if not ok:
        return f"Failed to delete note: {msg}"
    return f"Note delete result for {target}: {msg}"


if __name__ == "__main__":
    mcp.run()
