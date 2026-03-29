"""
tts.py - Mary text-to-speech.

Primary path:
- Piper binary + voice model if installed.

Fallback path on Windows:
- built-in SAPI voice via PowerShell.

Last resort:
- print the text only.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import sounddevice as sd
import soundfile as sf


ROOT = Path(__file__).resolve().parent
PIPER_BINARY = "piper.exe"
VOICE_MODEL = ROOT / "models" / "en_US-lessac-medium.onnx"
PIPER_LOCAL_BINARY = ROOT / "bin" / "piper" / "piper.exe"


def _resolve_piper_binary() -> str | None:
    if PIPER_LOCAL_BINARY.exists():
        return str(PIPER_LOCAL_BINARY)
    found = shutil.which(PIPER_BINARY)
    if found:
        return found
    return None


def _speak_with_piper(text: str, verbose: bool = True) -> bool:
    binary = _resolve_piper_binary()
    if not binary:
        return False
    if not VOICE_MODEL.exists():
        if verbose:
            print(f"[TTS] Piper found but voice model is missing at {VOICE_MODEL}")
        return False

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        result = subprocess.run(
            [
                binary,
                "--model",
                str(VOICE_MODEL),
                "--output_file",
                tmp_path,
            ],
            input=text.encode("utf-8"),
            capture_output=True,
        )

        if result.returncode != 0:
            if verbose:
                print(f"[TTS] Piper error: {result.stderr.decode(errors='replace')}")
            return False

        data, samplerate = sf.read(tmp_path, dtype="float32")
        sd.play(data, samplerate)
        sd.wait()
        return True
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _speak_with_windows_sapi(text: str, verbose: bool = True) -> bool:
    if platform.system() != "Windows":
        return False

    script = (
        "$text = [Console]::In.ReadToEnd(); "
        "Add-Type -AssemblyName System.Speech; "
        "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$speaker.Speak($text)"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        input=text.encode("utf-8"),
        capture_output=True,
    )
    if result.returncode != 0:
        if verbose:
            print(f"[TTS] Windows SAPI error: {result.stderr.decode(errors='replace')}")
        return False
    return True


def speak(text: str, verbose: bool = True) -> None:
    """Speak text with Piper, then Windows SAPI, then print fallback."""
    if not text.strip():
        return

    if verbose:
        print(f"[TTS] Speaking: {text}")

    if _speak_with_piper(text, verbose=verbose):
        return
    if _speak_with_windows_sapi(text, verbose=verbose):
        return

    print(f"[TTS] No TTS backend available. Response: {text}")


def speak_streaming(text: str, verbose: bool = True) -> None:
    """Speak sentence-by-sentence for lower perceived latency."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for sentence in sentences:
        if sentence.strip():
            speak(sentence.strip(), verbose=verbose)
