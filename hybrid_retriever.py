"""
hybrid_retriever.py  ─  OPTIMISED v3  (+Voice Output)
Semantic Search + BM25 → Merge/Deduplicate → CrossEncoder Rerank → Voice Playback

Changes vs v2
  7. voice_output.VoiceSpeaker integrated — after each text response is printed,
     F5TTS clones Andrew's voice, saves a temp .wav, plays it, then deletes it.
     If voice dependencies are missing the pipeline silently falls back to text-only.
"""

import os
import json
import numpy as np
import torch
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ── Voice output (optional — requires F5TTS, soundfile, sounddevice) ──────────
# voice_output.py handles the Windows FFmpeg DLL registration before F5TTS loads.
try:
    from voice_output import VoiceSpeaker, VOICE_AVAILABLE as _VOICE_AVAIL
except ImportError:
    _VOICE_AVAIL = False
    VoiceSpeaker  = None   # type: ignore[assignment,misc]

# ── Voice input (optional — requires sounddevice, soundfile, requests + ELEVEN_LABS_API_KEY) ─
try:
    from voice_input import get_voice_query, VOICE_INPUT_AVAILABLE as _VOICE_INPUT_AVAIL
except ImportError:
    _VOICE_INPUT_AVAIL = False
    get_voice_query    = None  # type: ignore[assignment]

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH     = "chroma_db_data"
COLLECTION_NAME = "research_papers"
EMBED_MODEL     = "all-MiniLM-L6-v2"
RERANK_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RETRIEVAL = 8       # was 12  →  fewer cross-encoder pairs = big CPU speedup
TOP_K_FINAL     = 4       # was 7   →  smaller prompt sent to Gemini
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
SEMANTIC_WEIGHT = 0.70
KEYWORD_WEIGHT  = 0.30
RRF_K           = 60
MAX_SEQ_LEN     = 256
MAX_DOC_CHARS   = 400     # was 800  →  halves cross-encoder input & Gemini prompt size

# Style guide: only the first N characters of Lecture_Andrew.txt are needed
STYLE_GUIDE_MAX_CHARS = 1_500

# ── Voice config — edit these to match your setup ────────────────────────────
VOICE_REFERENCE_AUDIO = "andrew_voice.wav"
VOICE_REFERENCE_TEXT  = (
    "In this class you learn about the state of the art and also "
    "practice implementing machine learning algorithms yourself. "
    "You learn about the"
)

# ── Gemini Configuration ──────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Please set it in your .env file or as an environment variable."
    )

# ── Core functions ────────────────────────────────────────────────────────────
def load_chroma() -> chromadb.Collection:
    return chromadb.PersistentClient(path=CHROMA_PATH).get_collection(COLLECTION_NAME)


def build_bm25_index(collection: chromadb.Collection) -> tuple[BM25Okapi, list[dict]]:
    result   = collection.get(include=["documents", "metadatas"])
    ids, docs_raw, metas = result["ids"], result["documents"], result["metadatas"]
    corpus   = [doc.lower().split() for doc in docs_raw]
    docs     = [{"id": ids[i], "document": docs_raw[i], "metadata": metas[i]}
                for i in range(len(ids))]
    return BM25Okapi(corpus), docs


def semantic_search(query_vec: list[float], collection: chromadb.Collection, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    res = collection.query(query_embeddings=[query_vec], n_results=top_k, include=["documents", "metadatas"])
    ids, docs, metas = res["ids"][0], res["documents"][0], res["metadatas"][0]
    return [{"id": ids[i], "document": docs[i], "metadata": metas[i]} for i in range(len(ids))]


def keyword_search(query_tokens: list[str], bm25_index: BM25Okapi, bm25_docs: list[dict], top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    scores  = np.array(bm25_index.get_scores(query_tokens))
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [bm25_docs[i] for i in top_idx if scores[i] > 0]


def merge_results(semantic: list[dict], keyword: list[dict], semantic_weight: float = SEMANTIC_WEIGHT, keyword_weight: float = KEYWORD_WEIGHT, k: int = RRF_K) -> list[dict]:
    scores, docs_map = {}, {}
    for rank, doc in enumerate(semantic, 1):
        scores[doc["id"]]   = scores.get(doc["id"], 0.0) + semantic_weight / (k + rank)
        docs_map[doc["id"]] = doc
    for rank, doc in enumerate(keyword, 1):
        scores[doc["id"]]   = scores.get(doc["id"], 0.0) + keyword_weight / (k + rank)
        docs_map[doc["id"]] = doc
    return [docs_map[i] for i in sorted(scores, key=scores.__getitem__, reverse=True)]


def rerank(query: str, candidates: list[dict], cross_encoder: CrossEncoder, top_k: int = TOP_K_FINAL) -> list[dict]:
    if not candidates:
        return []
    pairs  = [[query, c["document"][:MAX_DOC_CHARS]] for c in candidates]
    scores = cross_encoder.predict(pairs, batch_size=32, show_progress_bar=False)
    ranked = np.argsort(scores)[::-1][:top_k]
    return [{**candidates[i], "rerank_score": float(scores[i])} for i in ranked]


# ── HybridRetriever ───────────────────────────────────────────────────────────
class HybridRetriever:
    def __init__(self, n_workers: int = 2):
        self.collection    = load_chroma()
        self.embed_model   = SentenceTransformer(EMBED_MODEL, device=DEVICE)
        self.cross_encoder = CrossEncoder(RERANK_MODEL, device=DEVICE, max_length=MAX_SEQ_LEN)
        self.bm25_index, self.bm25_docs = build_bm25_index(self.collection)
        self._executor     = ThreadPoolExecutor(max_workers=n_workers)
        self._embed_cache: dict[str, list[float]] = {}

        self._warmup()
        print(f"HybridRetriever ready — {len(self.bm25_docs)} chunks indexed on {DEVICE}")

    def _warmup(self) -> None:
        with torch.no_grad():
            self.embed_model.encode("warmup", convert_to_numpy=True)
        self.cross_encoder.predict([["warmup", "warmup"]], show_progress_bar=False)

    def _encode(self, query: str) -> list[float]:
        if query not in self._embed_cache:
            with torch.no_grad():
                vec = self.embed_model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
            self._embed_cache[query] = vec.tolist()
        return self._embed_cache[query]

    def retrieve(self, query: str, top_k: int = TOP_K_FINAL) -> list[dict]:
        query_vec    = self._encode(query)
        query_tokens = query.lower().split()

        future_sem = self._executor.submit(semantic_search, query_vec, self.collection)
        future_kw  = self._executor.submit(keyword_search, query_tokens, self.bm25_index, self.bm25_docs)
        semantic   = future_sem.result()
        keyword    = future_kw.result()

        merged = merge_results(semantic, keyword)
        return rerank(query, merged, self.cross_encoder, top_k)

    def close(self) -> None:
        self._executor.shutdown(wait=True)


# ── Helper Utilities ─────────────────────────────────────────────────────────
def read_file(filepath: str, default_text: str = "") -> str:
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    return default_text


def read_json(filepath: str) -> list | dict:
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return [] if filepath.endswith("memory.json") else {}


def save_json(filepath: str, data: list | dict) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Interactive Pipeline Execution Engine ─────────────────────────────────────
if __name__ == "__main__":
    import time

    # ── One-time initialisation ───────────────────────────────────────────────
    ai_client = genai.Client(api_key=GEMINI_API_KEY)
    retriever = HybridRetriever()

    # ── Voice speaker setup (optional) ────────────────────────────────────────
    # Initialised once here — F5TTS model loading takes ~5-10 s on first call.
    # If voice dependencies are missing or the reference file is absent, the
    # pipeline continues in text-only mode without crashing.
    speaker = None
    if _VOICE_AVAIL:
        try:
            speaker = VoiceSpeaker(
                reference_audio=VOICE_REFERENCE_AUDIO,
                reference_text=VOICE_REFERENCE_TEXT,
                nfe_step=8,   # 32 = default quality (~9 min CPU), 8 = fast (~2 min CPU)
            )
        except (FileNotFoundError, RuntimeError) as _ve:
            print(f"[Voice] Skipping voice output — {_ve}")
            print("[Voice] Running in text-only mode.\n")
    else:
        print("[Voice] Voice module not loaded — running in text-only mode.\n")

    # ── Voice input status ────────────────────────────────────────────────────
    if _VOICE_INPUT_AVAIL:
        print("[Voice Input] ElevenLabs STT ready — type 'v' at the prompt to speak your query.\n")
    else:
        print(
            "[Voice Input] Not available — ensure ELEVEN_LABS_API_KEY is set in .env\n"
            "              and sounddevice / soundfile / requests are installed.\n"
        )

    # Load static files ONCE — not on every query.
    persona_rules = read_file("persona.txt", "You are Andrew Ng.")
    style_guide   = read_file("Lecture_Andrew.txt", "")[:STYLE_GUIDE_MAX_CHARS]
    long_term_mem = read_json("long_term_memory.json")

    # Pre-build the immutable system instruction once.
    SYSTEM_INSTRUCTION = f"""
    {persona_rules}

    STYLIZATION CONSTRAINT:
    Use the following lecture excerpt to guide your vocabulary and phrasing. Mirror this tone explicitly:
    ---
    {style_guide}
    ---

    CRITICAL WORD LIMIT:
    Your response must be an absolute maximum of 50 to 60 words. This is a strict constraint. Ensure the output fits gracefully within this window without ending mid-sentence.
    """

    _has_voice_out = bool(speaker)
    if _VOICE_INPUT_AVAIL and _has_voice_out:
        mode_label = "Voice I/O"
    elif _VOICE_INPUT_AVAIL:
        mode_label = "Voice Input + Text Output"
    elif _has_voice_out:
        mode_label = "Text Input + Voice Output"
    else:
        mode_label = "Text-only"

    voice_hint = "  (type 'v' + Enter to use your mic)" if _VOICE_INPUT_AVAIL else ""
    print(f"\nReady [{mode_label}]. Type a query and press Enter.{voice_hint} ('quit' to finalize session)\n")

    while True:
        # ── Input: text or voice ──────────────────────────────────────────────
        try:
            _prompt = "Query [T]ext / [V]oice > " if _VOICE_INPUT_AVAIL else "Query > "
            raw = input(_prompt).strip()
        except (KeyboardInterrupt, EOFError):
            raw = "quit"

        # Route to ElevenLabs STT when the user types 'v' or 'voice'
        if raw.lower() in ("v", "voice") and _VOICE_INPUT_AVAIL:
            query = get_voice_query()
            if not query:
                print("[Voice Input] Nothing transcribed — please try again.\n")
                continue
        else:
            query = raw

        if not query:
            continue

        # ── Session Termination & Compilation ────────────────────────────────
        if query.lower() == "quit":
            print("\nProcessing session termination...")
            short_term = read_json("short_term_memory.json")

            if short_term:
                summary_prompt = f"""
                Analyze the following recent conversation history and generate a structured summary.
                You must provide exactly three sections matching these names verbatim:
                - User preferences
                - Conversation summaries
                - Semantic memories

                CRITICAL WORD COUNT RULE: For each section, write a maximum of 10 to 15 words. Be extremely brief.

                Conversation History:
                {json.dumps(short_term, indent=2)}
                """

                try:
                    summary_response = ai_client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=summary_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction="You are an expert data analysis engine. Summarize accurately within word limits.",
                            temperature=0.2,
                            thinking_config=types.ThinkingConfig(thinking_budget=0)
                        )
                    )

                    long_term = read_json("long_term_memory.json")
                    if isinstance(long_term, list):
                        long_term.append({
                            "timestamp": time.time(),
                            "summary": summary_response.text.strip()
                        })
                        save_json("long_term_memory.json", long_term)
                        long_term_mem = long_term

                    print("\n=== Session Compiled ===")
                    print(summary_response.text.strip())
                except Exception as e:
                    print(f"Error compiling session metrics: {e}")

            save_json("short_term_memory.json", [])
            retriever.close()
            print("System components offline. Goodbye!")
            break

        # ── Executive Pipeline Mode (Retrieve → Ingest → Process) ────────────
        t0 = time.perf_counter()

        # 1. Gather Vector Chunks
        retrieved_results = retriever.retrieve(query)
        chunks_text = "\n".join([f"Context [{i+1}]: {r['document']}" for i, r in enumerate(retrieved_results)])

        # 2. Load short-term memory (changes every turn; must be fresh)
        short_term_mem = read_json("short_term_memory.json")

        # 3. Construct prompt payload
        prompt_payload = f"""
        Long-Term Memory / User Profile Context:
        {json.dumps(long_term_mem)}

        Short-Term Context (Recent Turn History):
        {json.dumps(short_term_mem)}

        Semantically Relevant Retrieved Chunks:
        {chunks_text}

        Current User Prompt:
        {query}
        """

        # 4. Query Engine Processing
        try:
            response = ai_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt_payload,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.3,
                    max_output_tokens=120,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            )
            output_text = response.text.strip()
            elapsed = time.perf_counter() - t0

            # 5. Print text response first so the user sees it immediately
            print(f"\nResponse ({elapsed:.2f}s) > {output_text}\n")

            # 6. ── Voice output ────────────────────────────────────────────────
            # Synthesises audio → saves temp .wav → plays → auto-deletes.
            # Runs synchronously: next prompt appears only after playback ends.
            if speaker:
                speaker.speak(output_text)

            # 7. History Maintenance
            if not isinstance(short_term_mem, list):
                short_term_mem = []
            short_term_mem.append({"user": query, "model": output_text})
            save_json("short_term_memory.json", short_term_mem)

        except Exception as e:
            print(f"\nPipeline processing failure: {e}\n")