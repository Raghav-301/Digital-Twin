"""
api.py — FastAPI Backend Server for Andrew Ng Digital Twin
Serves the REST API for RAG retrieval & Gemini generation,
and hosts the custom web frontend (index.html, style.css, script.js).
"""

import os
import sys
import time
import json
import asyncio
import uvicorn
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from google import genai
from google.genai import types

# Import project backend dependencies without modifying them
from hybrid_retriever import HybridRetriever, read_file, read_json, save_json

load_dotenv()

# Global lazy service instances
retriever: Optional[HybridRetriever] = None
ai_client: Optional[genai.Client] = None
service_lock = asyncio.Lock()

# Load system instructions from workspace files
persona_rules = read_file("persona.txt", "You speak exactly as Andrew Ng himself, using first-person pronouns.")
style_guide   = read_file("Lecture_Andrew.txt", "")[:1500]

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

def get_gemini_client() -> Optional[genai.Client]:
    global ai_client
    if ai_client is None:
        key = os.getenv("GEMINI_API_KEY")
        if key and key != "your_gemini_api_key_here":
            ai_client = genai.Client(api_key=key)
    return ai_client

def get_retriever() -> HybridRetriever:
    global retriever
    if retriever is None:
        print("[API] Initializing HybridRetriever (loading ChromaDB vector store and neural reranker)...")
        retriever = HybridRetriever()
        print("[API] HybridRetriever successfully loaded!")
    return retriever


app = FastAPI(
    title="Andrew Ng Digital Twin API",
    description="REST API backing the Andrew Ng AI Digital Twin frontend.",
    version="1.0.0"
)

# Enable CORS for external frontend applications if hosted on different domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
class MessageItem(BaseModel):
    user: str
    model: str

class ChatRequest(BaseModel):
    message: str = Field(..., description="User prompt or question")
    history: Optional[List[Dict[str, str]]] = Field(default=[], description="List of previous turn dicts [{'user': '...', 'model': '...'}]")
    include_sources: Optional[bool] = Field(default=True, description="Whether to include retrieved RAG context sources in response")

class SourceChunk(BaseModel):
    id: str
    document: str
    metadata: Dict[str, Any] = {}
    rerank_score: float = 0.0

class ChatResponse(BaseModel):
    response: str
    sources: List[SourceChunk] = []
    elapsed_seconds: float
    status: str = "success"

class ResetResponse(BaseModel):
    status: str
    message: str

class SystemStatus(BaseModel):
    status: str
    retriever_ready: bool
    gemini_key_set: bool
    indexed_documents_count: int


# API Routes
@app.get("/api/status", response_model=SystemStatus)
async def check_status():
    """Health check endpoint showing backend initialization status."""
    key = os.getenv("GEMINI_API_KEY")
    key_valid = bool(key and key != "your_gemini_api_key_here")
    
    is_ready = retriever is not None
    doc_count = len(retriever.bm25_docs) if retriever else 0
    
    return SystemStatus(
        status="ok" if key_valid else "warning_missing_api_key",
        retriever_ready=is_ready,
        gemini_key_set=key_valid,
        indexed_documents_count=doc_count
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request_data: ChatRequest):
    """
    Main RAG Chat Endpoint:
    1. Retrieves semantically relevant documents using HybridRetriever
    2. Constructs prompt payload with short-term history & retrieved context
    3. Calls Gemini 2.5 Flash for response generation
    4. Updates short-term memory json
    """
    query = request_data.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    
    client = get_gemini_client()
    if not client:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured or invalid in environment variables."
        )

    t0 = time.perf_counter()

    try:
        # Retrieve context using thread pool / sync retriever
        ret = get_retriever()
        retrieved_results = ret.retrieve(query)
    except Exception as exc:
        print(f"[API Error] Retrieval failed: {exc}")
        retrieved_results = []

    chunks_text = "\n".join([f"Context [{i+1}]: {r['document']}" for i, r in enumerate(retrieved_results)])

    # Format history payload
    history_payload = []
    if request_data.history:
        history_payload = request_data.history[-3:]
    else:
        # Fallback to short_term_memory.json
        st_mem = read_json("short_term_memory.json")
        if isinstance(st_mem, list):
            history_payload = st_mem[-3:]

    # Construct prompt payload
    prompt_payload = f"""
    Short-Term Context (Recent Conversation History):
    {json.dumps(history_payload)}

    Semantically Relevant Retrieved Chunks:
    {chunks_text}

    Current User Prompt:
    {query}
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt_payload,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.3,
                max_output_tokens=120,
            )
        )
        output_text = response.text.strip() if response.text else "No response generated."
    except Exception as exc:
        print(f"[API Error] Gemini generation error: {exc}")
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(exc)}")

    elapsed = round(time.perf_counter() - t0, 2)

    # Maintain short_term_memory.json
    try:
        st_mem = read_json("short_term_memory.json")
        if not isinstance(st_mem, list):
            st_mem = []
        st_mem.append({"user": query, "model": output_text})
        save_json("short_term_memory.json", st_mem)
    except Exception as mem_err:
        print(f"[API Warning] Failed to update short_term_memory.json: {mem_err}")

    # Build sources response list
    sources_list = []
    if request_data.include_sources:
        for item in retrieved_results:
            sources_list.append(SourceChunk(
                id=str(item.get("id", "")),
                document=str(item.get("document", "")),
                metadata=item.get("metadata", {}),
                rerank_score=round(float(item.get("rerank_score", 0.0)), 4)
            ))

    return ChatResponse(
        response=output_text,
        sources=sources_list,
        elapsed_seconds=elapsed,
        status="success"
    )


@app.get("/api/history")
async def get_history():
    """Retrieve full short-term memory history."""
    short_term = read_json("short_term_memory.json")
    long_term = read_json("long_term_memory.json")
    return {
        "short_term_memory": short_term if isinstance(short_term, list) else [],
        "long_term_memory": long_term if isinstance(long_term, list) else []
    }


@app.post("/api/reset", response_model=ResetResponse)
async def reset_chat():
    """Reset short term conversation history."""
    save_json("short_term_memory.json", [])
    return ResetResponse(
        status="success",
        message="Short-term conversation memory has been cleared."
    )


# Serve Static Frontend Files (index.html, style.css, script.js)
current_dir = os.path.dirname(os.path.abspath(__file__))

@app.get("/")
async def serve_index():
    index_file = os.path.join(current_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"message": "Frontend index.html not found. Please create index.html in the project root."})

app.mount("/", StaticFiles(directory=current_dir, html=True), name="static")


if __name__ == "__main__":
    print("\n========================================================")
    print(" Andrew Ng Digital Twin -- Web Server Booting")
    print(" Access the frontend in your browser at: http://localhost:8000")
    print(" API Documentation available at: http://localhost:8000/docs")
    print("========================================================\n")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
