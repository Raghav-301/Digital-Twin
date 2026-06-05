"""
voice_input.py — ElevenLabs Speech-to-Text (STT) Input Module
==============================================================
Records from the default microphone in 100-ms chunks, detects end-of-speech
via an RMS silence gate, then transcribes via the ElevenLabs Scribe v1 API.

Dependencies (install once):
    pip install sounddevice soundfile requests numpy python-dotenv

Environment variable required:
    ELEVEN_LABS_API_KEY=sk_...   (add to your .env file)
"""

import io
import os

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── ElevenLabs config ─────────────────────────────────────────────────────────
_ELEVEN_LABS_API_KEY: str | None = os.getenv("ELEVEN_LABS_API_KEY")
_STT_ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-text"
_STT_MODEL    = "scribe_v1"          # ElevenLabs' flagship STT model (multilingual)

# ── Recording defaults ────────────────────────────────────────────────────────
SAMPLE_RATE       = 16_000  # Hz  — 16 kHz is optimal for STT
CHANNELS          = 1       # mono
SILENCE_THRESHOLD = 0.015   # RMS amplitude below this counts as silence
SILENCE_DURATION  = 1.8     # seconds of continuous silence → auto-stop
MAX_RECORD_SEC    = 30      # hard cap so the loop never runs forever

# ── Availability check ────────────────────────────────────────────────────────
# Imported by hybrid_retriever.py to decide whether to surface the voice-input
# option to the user at startup.
VOICE_INPUT_AVAILABLE: bool = False

try:
    import sounddevice as _sd   # noqa: F401  (imported lazily inside functions)
    import soundfile  as _sf    # noqa: F401
    import requests   as _req   # noqa: F401

    if _ELEVEN_LABS_API_KEY:
        VOICE_INPUT_AVAILABLE = True
    else:
        print(
            "[Voice Input] ELEVEN_LABS_API_KEY not found in environment.\n"
            "              Add it to your .env file to enable microphone input.\n"
        )
except ImportError as _missing:
    print(
        f"[Voice Input] Missing dependency: {_missing}\n"
        f"              Run: pip install sounddevice soundfile requests\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
def _record_until_silence(
    sample_rate:  int   = SAMPLE_RATE,
    threshold:    float = SILENCE_THRESHOLD,
    silence_dur:  float = SILENCE_DURATION,
    max_sec:      float = MAX_RECORD_SEC,
) -> np.ndarray:
    """
    Streams microphone audio in 100-ms chunks.
    Recording stops automatically after `silence_dur` consecutive seconds
    of silence (post first detected speech), or when `max_sec` is reached.

    Returns
    -------
    np.ndarray
        1-D float32 array of the captured audio samples.
        Empty array if nothing was captured.
    """
    import sounddevice as sd

    chunk_frames   = int(sample_rate * 0.10)                          # 100 ms
    max_chunks     = int(max_sec * sample_rate / chunk_frames)
    silence_chunks = int(silence_dur * sample_rate / chunk_frames)

    frames: list[np.ndarray] = []
    silent_count   = 0
    speech_started = False

    print("[Voice Input] 🎤  Listening — speak now  (stops automatically on silence)...")

    with sd.InputStream(samplerate=sample_rate, channels=CHANNELS, dtype="float32") as stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_frames)
            frames.append(chunk.copy())

            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if rms > threshold:
                speech_started = True
                silent_count   = 0
            elif speech_started:
                silent_count += 1
                if silent_count >= silence_chunks:
                    break   # user has stopped speaking — exit early

    print("[Voice Input] ✅  Recording complete.")
    return np.concatenate(frames, axis=0).flatten() if frames else np.array([], dtype=np.float32)


def _transcribe(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """
    Encodes `audio` as a 16-bit PCM WAV in memory (no temp file written to
    disk) and POSTs it to the ElevenLabs Speech-to-Text endpoint.

    Returns
    -------
    str
        Transcribed text, stripped of leading/trailing whitespace.

    Raises
    ------
    RuntimeError
        If the API key is missing.
    requests.HTTPError
        If the API returns a non-2xx status code.
    """
    if not _ELEVEN_LABS_API_KEY:
        raise RuntimeError(
            "ELEVEN_LABS_API_KEY is not set. "
            "Add it to your .env file and restart."
        )

    import soundfile as sf
    import requests

    # Encode the float32 array as a 16-bit PCM WAV held purely in RAM
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)

    response = requests.post(
        _STT_ENDPOINT,
        headers={"xi-api-key": _ELEVEN_LABS_API_KEY},
        files={"file": ("query.wav", buf, "audio/wav")},
        data={"model_id": _STT_MODEL},
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("text", "").strip()


# ── Public API ────────────────────────────────────────────────────────────────
def get_voice_query() -> str:
    """
    Full voice-input pipeline:  record mic → transcribe via ElevenLabs → return text.

    This is the only function imported by hybrid_retriever.py.
    Returns an empty string on any failure; never raises an exception.
    """
    try:
        audio = _record_until_silence()

        if audio.size == 0:
            print("[Voice Input] ⚠  No audio captured — please try again.")
            return ""

        text = _transcribe(audio)

        if text:
            print(f"[Voice Input] 📝  Transcribed: \"{text}\"")
        else:
            print("[Voice Input] ⚠  Transcription returned an empty result.")

        return text

    except Exception as exc:
        print(f"[Voice Input] ⚠  Error during voice capture: {exc}")
        return ""