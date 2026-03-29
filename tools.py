"""
tools.py - Mary local tool registry.

This keeps the same lightweight function-dispatch style as HAL9000 so the
assistant can stay simple and fully local.
"""

from __future__ import annotations

import datetime
import os
import platform
import random
import subprocess


def get_time(args: list) -> str:
    now = datetime.datetime.now()
    return f"It's {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d')}."


def list_files(args: list) -> str:
    path = args[0] if args else "."
    try:
        files = os.listdir(path)
    except Exception as exc:
        return f"Couldn't list files: {exc}"

    if not files:
        return f"No files found in {path}."
    return f"Files in {path}: " + ", ".join(files[:10])


def system_info(args: list) -> str:
    try:
        import psutil
    except ImportError:
        return f"Running {platform.system()} {platform.release()}."

    cpu = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory()
    ram_used = ram.used // (1024 ** 2)
    ram_total = ram.total // (1024 ** 2)
    return f"CPU at {cpu}%, RAM using {ram_used}MB of {ram_total}MB."


def open_browser(args: list) -> str:
    url = args[0] if args else "https://www.google.com"
    try:
        if platform.system() == "Windows":
            os.startfile(url)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", url])
        else:
            subprocess.Popen(["xdg-open", url])
        return f"Opening {url}."
    except Exception as exc:
        return f"Couldn't open browser: {exc}"


def tell_joke(args: list) -> str:
    jokes = [
        "I would tell you a space joke, but the launch window closed.",
        "Why did the GPU stay calm? It had enough bandwidth to process it.",
        "I asked for a lightweight model and got emotional baggage in 4-bit.",
    ]
    return random.choice(jokes)


TOOLS: dict[str, callable] = {
    "get_time": get_time,
    "list_files": list_files,
    "system_info": system_info,
    "open_browser": open_browser,
    "tell_joke": tell_joke,
}


TOOL_DESCRIPTIONS = """
- get_time: Returns the current time and date. Args: none.
- list_files [path]: Lists files in a directory. Args: optional path.
- system_info: Returns CPU and RAM usage. Args: none.
- open_browser [url]: Opens the default browser. Args: optional URL.
- tell_joke: Tells a short joke. Args: none.
""".strip()


def dispatch_tool(tool_name: str, args: list) -> str:
    tool_fn = TOOLS.get(tool_name.lower().strip())
    if tool_fn is None:
        available = ", ".join(TOOLS.keys())
        return f"Unknown tool '{tool_name}'. Available: {available}."
    return tool_fn(args)
