"""
main.py - Mary pipeline orchestrator.

Current status:
- text-only mode works through llama.cpp + local GGUF model
- voice mode is wired for VAD -> ASR -> LLM -> TTS
- TTS uses Piper if installed, otherwise Windows SAPI, otherwise print fallback
"""

import argparse
import sys

from asr import load_asr_model, transcribe
from llm import generate, load_llm
from tts import speak
from vad import load_vad, record_until_silence, wait_for_activity


BANNER = """
========================================
 Mary - Offline Voice Assistant
 Windows + RTX 3060 local stack
========================================
"""


def run_voice_loop(llm, asr_model, vad_model, use_tts: bool) -> None:
    """Full voice loop."""
    history = []
    print("\n[MARY] Voice mode ready. Say something.\n")

    while True:
        try:
            wait_for_activity(vad_model, verbose=False)
            audio = record_until_silence(vad_model, verbose=True)
            if audio is None:
                continue

            user_text = transcribe(asr_model, audio)
            if not user_text:
                continue

            print(f"\n  You: {user_text}")
            response, history = generate(llm, user_text, history)
            print(f"  Mary: {response}\n")

            if use_tts:
                speak(response)
        except KeyboardInterrupt:
            print("\n[MARY] Goodbye.")
            sys.exit(0)


def run_text_loop(llm, use_tts: bool) -> None:
    """Working text chat loop backed by llama.cpp server."""
    history = []
    print("\n[MARY] Text mode ready. Type your message.\n")

    while True:
        try:
            user_text = input("  You: ").strip()
            if not user_text:
                continue
            if user_text.lower() in ("exit", "quit", "bye"):
                print("[MARY] Goodbye.")
                break

            response, history = generate(llm, user_text, history)
            print(f"  Mary: {response}\n")

            if use_tts:
                speak(response)
        except KeyboardInterrupt:
            print("\n[MARY] Goodbye.")
            sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mary - Windows GPU offline assistant")
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Use keyboard input instead of microphone",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Print responses instead of speaking them",
    )
    args = parser.parse_args()

    print(BANNER)
    use_tts = not args.no_tts

    print("[MARY] Loading local LLM...\n")
    llm = load_llm()

    if args.text_only:
        run_text_loop(llm, use_tts)
        return

    print("[MARY] Loading ASR...\n")
    asr_model = load_asr_model()

    print("[MARY] Loading VAD...\n")
    vad_model = load_vad()

    run_voice_loop(llm, asr_model, vad_model, use_tts)


if __name__ == "__main__":
    main()
