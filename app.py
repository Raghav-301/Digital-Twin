import os
import time
import json
import gradio as gr
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Import project retriever
from hybrid_retriever import HybridRetriever, read_file, read_json

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Lazy global instances
retriever = None
ai_client = None

def get_services():
    global retriever, ai_client
    if ai_client is None:
        key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
        if key and key != "your_gemini_api_key_here":
            ai_client = genai.Client(api_key=key)
    if retriever is None:
        print("[App] Initializing HybridRetriever (loading models and vector index)...")
        retriever = HybridRetriever()
        print("[App] HybridRetriever ready!")
    return retriever, ai_client

persona_rules = read_file("persona.txt", "You are Andrew Ng.")
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

def respond(message, history):
    key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    if not key or key == "your_gemini_api_key_here":
        return "⚠️ Error: GEMINI_API_KEY is missing. Please set your API key in the .env file!"

    try:
        ret_inst, client_inst = get_services()
    except Exception as e:
        return f"⚠️ Error initializing model/retriever: {str(e)}"

    if client_inst is None:
        return "⚠️ Error: Could not initialize Gemini API client. Please check GEMINI_API_KEY."

    # 1. Retrieve relevant chunks
    try:
        retrieved_results = ret_inst.retrieve(message)
        chunks_text = "\n".join([f"Context [{i+1}]: {r['document']}" for i, r in enumerate(retrieved_results)])
    except Exception as e:
        chunks_text = ""

    # 2. Build history payload
    formatted_history = []
    for user_msg, bot_msg in history:
        formatted_history.append({"user": user_msg, "model": bot_msg})

    prompt_payload = f"""
    Short-Term Context (Recent Conversation):
    {json.dumps(formatted_history[-3:])}

    Semantically Relevant Retrieved Chunks:
    {chunks_text}

    Current User Prompt:
    {message}
    """

    # 3. Generate response with Gemini
    try:
        response = client_inst.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt_payload,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.3,
                max_output_tokens=120,
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"Gemini API Error: {str(e)}"

# Build Gradio Chatbot interface
demo = gr.ChatInterface(
    fn=respond,
    title="🎓 Andrew Ng Digital Twin",
    description="Ask questions to the AI Digital Twin of Andrew Ng, trained on his lectures, research papers, and teaching style.",
    examples=["What is Machine Learning?", "Can you explain Gradient Descent intuitively?", "How do I start building a neural network?"],
)

if __name__ == "__main__":
    print("[App] Starting Gradio web server at http://localhost:7860...")
    demo.launch(server_name="0.0.0.0", server_port=7860)
