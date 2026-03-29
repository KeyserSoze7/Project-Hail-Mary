"""
asr.py - Mary speech-to-text.

Uses Faster-Whisper and prefers CUDA on the local RTX 3060.
Falls back to CPU if CUDA initialisation fails.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel


ROOT = Path(__file__).resolve().parent
CUDA_DLL_DIR = ROOT / "bin" / "llama.cpp"
DEFAULT_MODEL_SIZE = "small.en"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"
CPU_FALLBACK_COMPUTE_TYPE = "int8"


def _prepare_windows_cuda_runtime() -> None:
    """Expose CUDA DLLs shipped with local llama.cpp binaries to Faster-Whisper."""
    if os.name != "nt":
        return
    if not CUDA_DLL_DIR.exists():
        return

    dll_dir = str(CUDA_DLL_DIR)
    path_value = os.environ.get("PATH", "")
    if dll_dir not in path_value.split(os.pathsep):
        os.environ["PATH"] = dll_dir + os.pathsep + path_value

    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        add_dll_directory(dll_dir)


def load_asr_model(
    model_size: str = DEFAULT_MODEL_SIZE,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
) -> WhisperModel:
    """Load Faster-Whisper, preferring GPU and falling back to CPU."""
    _prepare_windows_cuda_runtime()

    print(f"[ASR] Loading Faster-Whisper '{model_size}' on {device}...")
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print(f"[ASR] Model loaded on {device}.")
        return model
    except Exception as exc:
        if device != "cuda":
            raise
        print(f"[ASR] CUDA load failed: {exc}")
        print("[ASR] Falling back to CPU int8.")
        model = WhisperModel(model_size, device="cpu", compute_type=CPU_FALLBACK_COMPUTE_TYPE)
        print("[ASR] Model loaded on cpu.")
        return model


def transcribe(model: WhisperModel, audio: np.ndarray, verbose: bool = True) -> str:
    """Transcribe a mono 16 kHz float32 numpy array into text."""
    segments, _ = model.transcribe(
        audio,
        beam_size=1,
        language="en",
        vad_filter=False,
        condition_on_previous_text=False,
    )

    text = " ".join(segment.text.strip() for segment in segments).strip()

    if verbose and text:
        print(f"[ASR] Transcript: {text}")
    elif verbose:
        print("[ASR] No speech recognised.")

    return text
