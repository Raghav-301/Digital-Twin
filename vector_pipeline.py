"""
vector_pipeline.py
Pipeline: .txt files -> semantic chunks -> embeddings -> ChromaDB

Tracking: research.txt holds the filenames already processed.
On each run, only files NOT listed there are chunked and indexed.
New filenames are appended to research.txt after successful indexing.
"""

import os
import chromadb
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# -- Config -------------------------------------------------------------------
DATA_DIR      = "Data_text"
PROCESSED_LOG = "research.txt"

# -- Init ---------------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
chunker    = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
collection = chromadb.PersistentClient(path="chroma_db_data").get_or_create_collection("research_papers")


# -- Helpers ------------------------------------------------------------------
def load_processed_log() -> set[str]:
    """Return the set of filenames already recorded in research.txt."""
    if not os.path.exists(PROCESSED_LOG):
        return set()
    with open(PROCESSED_LOG, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def append_to_log(filename: str) -> None:
    """Append a single filename to research.txt."""
    with open(PROCESSED_LOG, "a", encoding="utf-8") as f:
        f.write(filename + "\n")


def read_file(filepath: str) -> tuple[str, str]:
    """Line 0 = title, Line 1 = separator, Line 2+ = body."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    title = lines[0].strip() if lines else os.path.basename(filepath)
    body  = "".join(lines[2:]).strip()
    return title, body


def process_file(filepath: str) -> int:
    """Chunk one file, embed, upsert into ChromaDB. Returns chunk count."""
    source = os.path.splitext(os.path.basename(filepath))[0]
    title, body = read_file(filepath)

    if not body:
        return 0

    chunks = chunker.split_text(body)
    if not chunks:
        return 0

    collection.upsert(
        ids        = [f"{source}__{i}" for i in range(len(chunks))],
        embeddings = embeddings.embed_documents(chunks),
        documents  = chunks,
        metadatas  = [{"source": source, "title": title, "chunk_index": i} for i in range(len(chunks))],
    )
    return len(chunks)


# -- Main ---------------------------------------------------------------------
if __name__ == "__main__":

    # All .txt files currently in Data_text/
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    if not all_files:
        raise SystemExit(f"No .txt files found in '{DATA_DIR}/'")

    # Filenames already processed — O(1) lookup
    processed = load_processed_log()

    # Only files whose name is not in research.txt
    new_files = [f for f in all_files if f not in processed]

    if not new_files:
        print(f"Nothing to do — all {len(all_files)} file(s) already in research.txt")
    else:
        print(f"{len(new_files)} new file(s) to index  "
              f"({len(processed)} already logged in research.txt)\n")

        total = 0
        for i, filename in enumerate(new_files, 1):
            n = process_file(os.path.join(DATA_DIR, filename))
            total += n
            append_to_log(filename)          # written immediately after success
            print(f"[{i}/{len(new_files)}] {filename} -> {n} chunks")

        print(f"\nDone! {len(new_files)} file(s) -> {total} chunk(s) added.")
        print(f"research.txt now tracks {len(processed) + len(new_files)} file(s).")