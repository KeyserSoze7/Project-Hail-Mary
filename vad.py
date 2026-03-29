"""
vad.py - Mary voice activity detection.

Uses Silero VAD through torch.hub and sounddevice for microphone capture.
Silero expects 512-sample frames at 16 kHz, so the stream runs at that granularity.
"""

from __future__ import annotations

import numpy as np
import sounddevice as sd
import torch


SAMPLE_RATE = 16000
FRAME_SAMPLES = 512
FRAME_DURATION = FRAME_SAMPLES / SAMPLE_RATE
SILENCE_THRESHOLD = 25
MIN_SPEECH_CHUNKS = 6
DEFAULT_ACTIVITY_THRESHOLD = 0.6
DEFAULT_SPEECH_THRESHOLD = 0.5


def load_vad():
    """Load the Silero VAD model."""
    print("[VAD] Loading Silero VAD model...")
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    print("[VAD] Silero VAD loaded.")
    return model


def is_speech(model, audio_chunk: np.ndarray, threshold: float = DEFAULT_SPEECH_THRESHOLD) -> bool:
    """Return True when Silero scores the chunk above the threshold."""
    tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)
    confidence = model(tensor, SAMPLE_RATE).item()
    return confidence > threshold


def wait_for_activity(
    model,
    threshold: float = DEFAULT_ACTIVITY_THRESHOLD,
    verbose: bool = True,
) -> None:
    """Block until speech activity is detected."""
    if verbose:
        print("[VAD] Waiting for activity...")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME_SAMPLES,
    ) as stream:
        while True:
            chunk, _ = stream.read(FRAME_SAMPLES)
            chunk = chunk[:, 0]
            if is_speech(model, chunk, threshold=threshold):
                return


def record_until_silence(model, verbose: bool = True) -> np.ndarray | None:
    """Record a full utterance until enough silence is detected."""
    audio_buffer: list[np.ndarray] = []
    silent_chunks = 0
    speech_started = False
    speech_chunk_count = 0

    if verbose:
        print("[VAD] Listening... (speak now)")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME_SAMPLES,
    ) as stream:
        while True:
            chunk, _ = stream.read(FRAME_SAMPLES)
            chunk = chunk[:, 0]

            speaking = is_speech(model, chunk)
            if speaking:
                if not speech_started and verbose:
                    print("[VAD] Speech detected.")
                speech_started = True
                speech_chunk_count += 1
                silent_chunks = 0
                audio_buffer.append(chunk)
            elif speech_started:
                silent_chunks += 1
                audio_buffer.append(chunk)
                if silent_chunks >= SILENCE_THRESHOLD:
                    break

    if speech_chunk_count < MIN_SPEECH_CHUNKS:
        if verbose:
            print("[VAD] Too short, ignoring.")
        return None

    utterance = np.concatenate(audio_buffer, axis=0)
    if verbose:
        print(f"[VAD] Captured {len(utterance) / SAMPLE_RATE:.1f}s of audio.")
    return utterance
