# Mary

`Mary` is the Windows + GPU-oriented successor to `HAL9000`.


## Project Demo

<p align="center">
  <a href="https://youtu.be/BmJ_biVMV_Q">
    <img src="https://img.youtube.com/vi/BmJ_biVMV_Q/0.jpg" alt="Watch Demo" />
  </a>
</p>



Current target pipeline:

`Mic -> VAD -> ASR -> LLM -> TTS -> Speaker`

## Current backend choices

- `VAD`: Silero VAD through `torch.hub`
- `ASR`: Faster-Whisper, preferring CUDA on the RTX 3060
- `LLM`: local `llama.cpp` server with CUDA, using `unsloth/Llama-3.2-1B-Instruct-GGUF`
- `TTS`: Piper if installed, otherwise native Windows SAPI, otherwise print fallback
- `Tools`: lightweight local dispatcher in `tools.py`

## What works now

- `main.py --text-only --no-tts`
- local GPU-backed chat through `llama.cpp`
- basic local tools such as time lookup
- voice pipeline code is wired in `main.py`

## What still depends on local machine setup

- first `VAD` load may need the Silero model fetched through `torch.hub`
- first `ASR` load will download the Whisper model if it is not cached yet
- `Piper` is optional right now; if it is not installed, Mary will fall back to Windows SAPI speech on Windows

## Layout

```text
Mary/
|- main.py
|- vad.py
|- asr.py
|- llm.py
|- tts.py
|- tools.py
|- requirements.txt
|- models/
|  |- Llama-3.2-1B-Instruct-Q4_K_M.gguf
|  `- optional Piper voice files
`- bin/
   `- llama.cpp/
```

## Running

```powershell
# text mode
C:\aditya\Mary\.venv\Scripts\python.exe Mary\main.py --text-only --no-tts

# text mode with spoken output if a TTS backend is available
C:\aditya\Mary\.venv\Scripts\python.exe Mary\main.py --text-only

# full voice mode
C:\aditya\Mary\.venv\Scripts\python.exe Mary\main.py
```

## TTS behavior

`tts.py` uses this order:

1. `piper.exe` + a local voice model if available
2. Windows built-in SAPI voice
3. plain text fallback

If you want better offline voice quality, install Piper and place a voice model in `Mary/models/`.
