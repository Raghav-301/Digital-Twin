"""
voice_output.py  ─  Voice synthesis and auto-playback helper  (v2 — fast)
Pipeline: F5TTS generate → save .wav → play → auto-delete

Speed levers vs v1
  • nfe_step 32 → 8   : 4× fewer diffusion steps          (~4× faster inference)
  • torch CPU threads  : uses all physical cores by default (~1.5-2× on multi-core)
  • synthesis timer    : prints exact time so you can tune nfe_step further

Imported by hybrid_retriever.py. Can also be run standalone for testing.
"""

import os
import time

# ── Windows: register FFmpeg shared-library DLLs BEFORE any F5TTS import ─────
# Edit this path if your FFmpeg lives elsewhere.
FFMPEG_BIN_PATH = r"C:\Program Files\ffmpeg-master-latest-win64-gpl-shared\bin"
if os.name == "nt" and os.path.exists(FFMPEG_BIN_PATH):
    os.add_dll_directory(FFMPEG_BIN_PATH)

# ── Optional dependency guard ─────────────────────────────────────────────────
VOICE_AVAILABLE = False
try:
    import torch
    import soundfile as sf
    import sounddevice as sd
    from f5_tts.api import F5TTS

    # ── CPU thread tuning ─────────────────────────────────────────────────────
    # By default PyTorch may use far fewer threads than your CPU has cores.
    # Setting this to the physical core count gives a free 1.5-2× speedup on CPU.
    _cpu_cores = os.cpu_count() or 4
    torch.set_num_threads(_cpu_cores)
    torch.set_num_interop_threads(max(1, _cpu_cores // 2))

    VOICE_AVAILABLE = True
except ImportError as _err:
    print(
        f"[Voice] Dependency missing — voice output disabled. ({_err})\n"
        f"        Fix with: pip install sounddevice soundfile f5-tts"
    )


class VoiceSpeaker:
    """
    Voice synthesis helper for the hybrid-retriever pipeline.

    On every speak() call:
      1. Runs F5TTS inference (nfe_step controls speed vs quality).
      2. Saves the result as a timestamped .wav file.
      3. Plays that file (blocking until done).
      4. Deletes the file.

    Parameters
    ----------
    reference_audio : path to the 5-10 s reference WAV clip (default: andrew_voice.wav)
    reference_text  : transcript of what is said in the reference clip
    output_dir      : directory where the temp .wav is written (default: cwd)
    nfe_step        : diffusion steps — lower = faster, higher = slightly better quality
                      8  → fast   (~2 min on CPU, ~8 s on GPU)   ← default
                      16 → medium (~4 min on CPU, ~15 s on GPU)
                      32 → original quality (~9 min on CPU, ~30 s on GPU)
    """

    def __init__(
        self,
        reference_audio: str = "andrew_voice.wav",
        reference_text: str = (
            "In this class you learn about the state of the art and also "
            "practice implementing machine learning algorithms yourself. "
            "You learn about the"
        ),
        output_dir: str = ".",
        nfe_step: int = 8,
    ):
        if not VOICE_AVAILABLE:
            raise RuntimeError(
                "F5TTS / soundfile / sounddevice not installed. "
                "Run: pip install sounddevice soundfile f5-tts"
            )
        if not os.path.exists(reference_audio):
            raise FileNotFoundError(
                f"Reference audio clip not found: {reference_audio!r}  "
                "Place andrew_voice.wav in the working directory."
            )

        self.reference_audio = reference_audio
        self.reference_text  = reference_text
        self.output_dir      = output_dir
        self.nfe_step        = nfe_step

        device = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"[Voice] Loading F5TTS on {device} (nfe_step={nfe_step}, threads={torch.get_num_threads()})…")
        self.model = F5TTS()
        print("[Voice] Voice model ready.\n")

    # ─────────────────────────────────────────────────────────────────────────
    def speak(self, text: str) -> None:
        """
        Synthesise *text* in Andrew's cloned voice, play it, then delete the file.
        Blocks until playback finishes so the next query prompt appears only after
        the user has heard the full response.
        """
        wav_path = os.path.join(
            self.output_dir,
            f"response_{int(time.time() * 1_000)}.wav",
        )

        try:
            # 1. Synthesise ───────────────────────────────────────────────────
            t_start = time.perf_counter()
            print(f"[Voice] Synthesising… (nfe_step={self.nfe_step})")
            wav, sr, _ = self.model.infer(
                ref_file=self.reference_audio,
                ref_text=self.reference_text,
                gen_text=text,
                nfe_step=self.nfe_step,     # ← key speed lever
            )
            t_synth = time.perf_counter() - t_start
            print(f"[Voice] Synthesised in {t_synth:.1f}s")

            # 2. Save .wav ────────────────────────────────────────────────────
            sf.write(wav_path, wav, sr)
            print(f"[Voice] Playing → {os.path.basename(wav_path)}")

            # 3. Play (blocking) ──────────────────────────────────────────────
            data, samplerate = sf.read(wav_path)
            sd.play(data, samplerate)
            sd.wait()

        except Exception as exc:
            print(f"[Voice] Error during synthesis / playback: {exc}")

        finally:
            # 4. Always delete the temp file ──────────────────────────────────
            if os.path.exists(wav_path):
                os.remove(wav_path)
                print("[Voice] Temp file deleted.\n")


# ── Standalone smoke-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    speaker = VoiceSpeaker(nfe_step=8)
    test_text = (
        "Hello! This is a live pipeline test. "
        "The voice file will play right now and then disappear automatically."
    )
    print(f"Test text: {test_text!r}\n")
    speaker.speak(test_text)
    print("Smoke-test complete.")