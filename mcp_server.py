"""
Tiny MCP server that exposes two macOS actions as tools:
  - open_calculator
  - open_weather

Run standalone:  python mcp_server.py
But normally the listening_agent.py spawns it over stdio.
"""

import subprocess
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mac-actions")


@mcp.tool()
def open_calculator() -> str:
    """Open the macOS Calculator app. Use when the user wants to do math,
    asks to 'open calculator', mentions calculations, etc."""
    subprocess.run(["open", "-a", "Calculator"], check=False)
    return "Calculator is now open."


@mcp.tool()
def open_weather() -> str:
    """Open the macOS Weather app. Use when the user asks about the weather,
    forecast, temperature, rain, or says 'open weather'."""
    subprocess.run(["open", "-a", "Weather"], check=False)
    return "Opening the Weather app."


if __name__ == "__main__":
    # stdio transport — the agent will spawn this process and talk to it.
    mcp.run()
