"""
llm.py - Mary local LLM interface backed by llama.cpp server.

This avoids the CPU-only Python binding by launching the official Windows CUDA
llama.cpp server locally and talking to it over its OpenAI-compatible HTTP API.
"""

from __future__ import annotations

import atexit
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from tools import TOOL_DESCRIPTIONS, dispatch_tool


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
SERVER_PATH = ROOT / "bin" / "llama.cpp" / "llama-server.exe"
HOST = "127.0.0.1"
PORT = 8123
BASE_URL = f"http://{HOST}:{PORT}"
CONTEXT_LENGTH = 2048
MAX_TOKENS = 256
TEMPERATURE = 0.2
TOP_P = 0.9
GPU_LAYERS = "all"
SERVER_TIMEOUT = 180
READINESS_TIMEOUT = 120
RETRYABLE_STATUS_CODES = {503}

SYSTEM_PROMPT = f"""You are Mary, a concise offline voice assistant.
Keep responses short and conversational - 1 to 3 sentences max.

You have access to the following tools:

{TOOL_DESCRIPTIONS}

Use a tool when and only when the user explicitly asks for one of those exact actions.
For time/date, file listing, system info, opening the browser, or telling a joke, you must use the matching tool.
Do not call tools for normal conversation, writing help, explanations, summaries,
or direct instruction-following.

If a tool is necessary, respond with exactly this format and nothing else:
TOOL: <tool_name>
ARGS: <arg1>, <arg2>, ...

If a tool is not necessary, answer naturally and do not mention tools."""


class MaryLLM:
    """Thin wrapper around a local llama.cpp server process."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        server_path: Path = SERVER_PATH,
        host: str = HOST,
        port: int = PORT,
    ):
        self.model_path = Path(model_path)
        self.server_path = Path(server_path)
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return

        if not self.server_path.exists():
            raise FileNotFoundError(f"llama-server.exe not found at {self.server_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"GGUF model not found at {self.model_path}")

        print(f"[LLM] Starting llama.cpp server with model: {self.model_path.name}")

        args = [
            str(self.server_path),
            "-m",
            str(self.model_path),
            "-ngl",
            GPU_LAYERS,
            "-c",
            str(CONTEXT_LENGTH),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--jinja",
            "--reasoning",
            "off",
        ]

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )

        self._wait_until_ready()
        print("[LLM] llama.cpp server is ready.")

    def _wait_until_ready(self, timeout: int = READINESS_TIMEOUT) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError("llama.cpp server exited before becoming ready.")

            try:
                request = urllib.request.Request(f"{self.base_url}/health", method="GET")
                with urllib.request.urlopen(request, timeout=5) as response:
                    if response.status == 200:
                        return
            except urllib.error.HTTPError as exc:
                if exc.code not in RETRYABLE_STATUS_CODES:
                    details = exc.read().decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"Unexpected llama.cpp readiness error {exc.code}: {details}"
                    ) from exc
            except urllib.error.URLError:
                pass

            time.sleep(1)

        raise TimeoutError("Timed out waiting for llama.cpp server to become healthy.")

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is not None:
            return

        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
    ) -> str:
        self.start()

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }

        request = urllib.request.Request(
            url=f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=SERVER_TIMEOUT) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"].strip()
            except urllib.error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                if exc.code in RETRYABLE_STATUS_CODES and attempt < 4:
                    time.sleep(1)
                    continue
                raise RuntimeError(f"llama.cpp HTTP error {exc.code}: {details}") from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"Could not reach llama.cpp server: {exc}") from exc

        raise RuntimeError("llama.cpp request failed after retries.")


_LLM_INSTANCE: MaryLLM | None = None


def _cleanup() -> None:
    if _LLM_INSTANCE is not None:
        _LLM_INSTANCE.stop()


atexit.register(_cleanup)


def load_llm(model_path: str | Path = MODEL_PATH) -> MaryLLM:
    """Load or reuse the local llama.cpp server wrapper."""
    global _LLM_INSTANCE
    if _LLM_INSTANCE is None:
        _LLM_INSTANCE = MaryLLM(model_path=Path(model_path))
    _LLM_INSTANCE.start()
    return _LLM_INSTANCE


def build_prompt(user_text: str, history: list[dict]) -> list[dict]:
    """Build the chat messages array sent to llama.cpp."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages


def _match_direct_tool_call(user_text: str) -> tuple[str, list[str]] | None:
    """Deterministic router for the small built-in local tool set."""
    text = user_text.strip()
    lowered = text.lower()

    if any(phrase in lowered for phrase in (
        "what time",
        "what's the time",
        "current time",
        "what is the date",
        "what's the date",
        "what day is it",
        "today's date",
    )):
        return "get_time", []

    if "tell me a joke" in lowered or lowered.strip() == "joke":
        return "tell_joke", []

    if any(phrase in lowered for phrase in (
        "system info",
        "system information",
        "cpu and ram",
        "ram usage",
        "memory usage",
    )):
        return "system_info", []

    if any(phrase in lowered for phrase in (
        "open browser",
        "open the browser",
        "launch browser",
    )):
        return "open_browser", []

    if any(phrase in lowered for phrase in (
        "list files",
        "show files",
        "show me files",
        "list directory",
        "show directory",
    )):
        match = re.search(r"(?:in|at)\s+(.+)$", text, flags=re.IGNORECASE)
        if match:
            return "list_files", [match.group(1).strip().strip('"')]
        return "list_files", []

    return None


def generate(
    llm: MaryLLM,
    user_text: str,
    history: list[dict],
    verbose: bool = True,
) -> tuple[str, list[dict]]:
    """Generate a response, optionally dispatching local tools."""
    direct_tool = _match_direct_tool_call(user_text)
    if direct_tool is not None:
        tool_name, args = direct_tool
        if verbose:
            print(f"[LLM] Direct tool route: {tool_name}({args})")
        response_text = dispatch_tool(tool_name, args)
    else:
        messages = build_prompt(user_text, history)
        raw_response = llm.chat(messages=messages)

        if verbose:
            print(f"[LLM] Raw response: {raw_response}")

        if raw_response.startswith("TOOL:"):
            response_text = _handle_tool_call(raw_response, verbose)
        else:
            response_text = raw_response

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": response_text})
    if len(history) > 12:
        history = history[-12:]

    return response_text, history


def _handle_tool_call(raw: str, verbose: bool) -> str:
    """Parse TOOL/ARGS lines and dispatch to tools.py."""
    try:
        lines = raw.strip().splitlines()
        tool_name = lines[0].replace("TOOL:", "").strip()
        args: list[str] = []
        if len(lines) > 1:
            args_line = lines[1].replace("ARGS:", "").strip()
            lowered = args_line.lower()
            if args_line and lowered not in {"none", "null", "nil", "n/a"}:
                args = [item.strip() for item in args_line.split(",") if item.strip()]

        if verbose:
            print(f"[LLM] Tool call detected: {tool_name}({args})")

        return dispatch_tool(tool_name, args)
    except Exception as exc:
        return f"I tried to use a tool but something went wrong: {exc}"


if __name__ == "__main__":
    llm = load_llm()
    history: list[dict] = []
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "bye"}:
            break
        response, history = generate(llm, user_input, history)
        print(f"Assistant: {response}\n")
