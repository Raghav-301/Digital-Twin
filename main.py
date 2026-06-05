"""
main.py (Runtime Context & Driver Script)
Sets up necessary file structures and launches the hybrid_retriever loop.
Preprocessing pipeline (pdf_to_text → research_paper_scrapper → vector_pipeline)
runs exactly once per project, guarded by a .pipeline_initialized flag file.
"""

import os
import json
import subprocess
import sys
from dotenv import load_dotenv

REQUIRED_FILES = {
    "persona.txt": "You speak exactly as Andrew Ng himself, using first-person pronouns. Always be encouraging, clear, and prioritize logical intuition over abstract definitions.",
    "Lecture_Andrew.txt": "So, let me give you a concrete example. Imagine you want to map an input A to an output B. It turns out that building networks is a bit like Lego bricks. So the key idea here is to start simple.",
    "short_term_memory.json": [],
    "long_term_memory.json": []
}

def verify_environment():
    """Ensure file state anchors are correctly initialized before booting REPL."""
    load_dotenv()   # populate os.environ from .env before any key checks
    print("Verifying runtime local file architecture...")
    for filename, default_content in REQUIRED_FILES.items():
        if not os.path.exists(filename):
            print(f"Creating missing template file: {filename}")
            if filename.endswith(".json"):
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(default_content, f, indent=2)
            else:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(default_content)

    # ── Voice reference audio check ───────────────────────────────────────────
    # andrew_voice.wav is a binary file — it cannot be auto-generated.
    # Place a 5-10 second clean WAV clip of Andrew's voice in the working directory.
    if os.path.exists("andrew_voice.wav"):
        size_kb = os.path.getsize("andrew_voice.wav") // 1024
        print(f"Voice reference audio found: andrew_voice.wav ({size_kb} KB) — voice output enabled.")
    else:
        print(
            "\n[Voice] WARNING: 'andrew_voice.wav' not found in the working directory.\n"
            "         Voice output will be disabled for this session.\n"
            "         To enable it: place a 5-10 s clean WAV clip of Andrew's voice\n"
            "         in this folder and name it 'andrew_voice.wav'.\n"
        )

    # ── ElevenLabs STT API key check ──────────────────────────────────────────
    # Required for voice input (microphone → speech-to-text) via ElevenLabs.
    eleven_key = os.environ.get("ELEVEN_LABS_API_KEY", "")
    if eleven_key:
        print("[Voice Input] ELEVEN_LABS_API_KEY found — microphone input enabled.")
    else:
        print(
            "\n[Voice Input] WARNING: 'ELEVEN_LABS_API_KEY' not set.\n"
            "               Voice input will be disabled for this session.\n"
            "               To enable it: add ELEVEN_LABS_API_KEY=sk_... to your .env file.\n"
        )

PIPELINE_FLAG = ".pipeline_initialized"
PIPELINE_STEPS = [
    ("pdf_to_text.py",            "Step 1/3 — Converting PDFs to text..."),
    ("research_paper_scrapper.py","Step 2/3 — Scraping research papers..."),
    ("vector_pipeline.py",        "Step 3/3 — Building vector index..."),
]

def run_preprocessing_pipeline():
    """
    Run pdf_to_text → research_paper_scrapper → vector_pipeline exactly once.
    On subsequent launches the flag file '.pipeline_initialized' is detected
    and the entire block is skipped, so startup stays fast.
    """
    if os.path.exists(PIPELINE_FLAG):
        print(f"[Pipeline] Flag '{PIPELINE_FLAG}' found — preprocessing already complete, skipping.\n")
        return

    print("=" * 60)
    print("[Pipeline] First-run preprocessing started.")
    print("=" * 60)

    for script, label in PIPELINE_STEPS:
        print(f"\n[Pipeline] {label}")
        if not os.path.exists(script):
            print(f"[Pipeline] WARNING: '{script}' not found — skipping this step.")
            continue
        try:
            subprocess.run([sys.executable, script], check=True)
            print(f"[Pipeline] '{script}' completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[Pipeline] ERROR: '{script}' exited with code {e.returncode}.")
            print("[Pipeline] Aborting preprocessing — fix the error and re-run.")
            sys.exit(e.returncode)

    # Write the flag file so this block is never executed again
    with open(PIPELINE_FLAG, "w", encoding="utf-8") as f:
        f.write("Preprocessing pipeline completed successfully.\n")

    print("\n[Pipeline] All preprocessing steps done.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    verify_environment()
    run_preprocessing_pipeline()
    print("\nLaunching Persistent Interactive REPL pipeline via hybrid_retriever...\n")

    # Run the retriever process, matching stdout/stderr channels dynamically
    try:
        subprocess.run([sys.executable, "hybrid_retriever.py"], check=True)
    except KeyboardInterrupt:
        print("\nDriver process stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"\nREPL exited with an unexpected error state code: {e.returncode}")