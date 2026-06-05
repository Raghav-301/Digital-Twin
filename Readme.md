Andrew Ng Digital Twin

An interactive AI agent that recreates Andrew Ng's teaching style and voice. The system retrieves information from his lectures and publications, generates responses in his persona using Gemini and delivers them through a cloned version of his voice.

Project Overview

The project consists of two main parts:

A preprocessing pipeline that prepares the data for retrieval.
An interactive engine that handles user queries and generates responses.
Preprocessing Pipeline

The preprocessing pipeline runs only during the first launch. Once completed, it creates a .pipeline_initialized file so the setup process is skipped in future runs.

1. PDF Conversion (pdf_to_text.py)

Converts PDFs stored in the Data_pdfs directory into text files inside Data_text. The process uses Poppler and Tesseract OCR for extraction.

2. Publication Scraping (research_paper_scrapper.py)

Retrieves Andrew Ng's publication information from DBLP and stores the paper titles and abstracts in the Data_text directory.

3. Vector Database Ingestion (vector_pipeline.py)

Processes all text files by:

Splitting them into semantic chunks
Generating embeddings using a local Hugging Face model
Storing the embeddings in ChromaDB

The script keeps track of processed files using research.txt to avoid reprocessing the same content.

Interactive Engine (hybrid_retriever.py)

After preprocessing is complete, main.py starts the interactive loop.

1. Retrieval

Uses a hybrid retrieval strategy that combines:

Semantic search through ChromaDB
Keyword search using BM25

The results are merged using Reciprocal Rank Fusion (RRF).

2. Reranking

A cross-encoder model evaluates the retrieved chunks and selects the most relevant context for the query.

3. Response Generation

The selected context, conversation history and user query are sent to Gemini.

The response is guided by:

Rules defined in persona.txt
Teaching style constraints extracted from Lecture_Andrew.txt

This helps maintain Andrew Ng's teaching style while keeping responses concise.

4. Voice Synthesis

Responses are converted into speech using F5-TTS and a reference voice sample stored in andrew_voice.wav.

5. Voice Input

If voice mode is enabled, microphone input is transcribed using the ElevenLabs Speech-to-Text API.

6. Memory Management

The system stores short-term conversation history during a session and consolidates it into long-term memory when the session ends.

Setup Instructions
External Requirements

Install the following tools and ensure they are available in your system PATH:

Poppler: Used for PDF-to-image conversion
Tesseract OCR: Used for text extraction
FFmpeg: Required for audio processing and playback
Python Environment
Create a Virtual Environment
python -m venv .venv
.venv\Scripts\activate
Install Dependencies
pip install -r requirements.txt
Configuration

Create a .env file in the project root and add the following variables:

# Required for response generation
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Required for voice input
ELEVEN_LABS_API_KEY=your_elevenlabs_api_key_here

# Optional: Override paths if tools are not in PATH
POPPLER_PATH=C:\path\to\poppler\bin
TESSERACT_PATH=C:\path\to\tesseract.exe

# Optional: Required for andrew_experience.py
LINKEDIN_COOKIE=your_linkedin_li_at_cookie_here
Voice Reference File

To enable voice synthesis, place a clean 5-10 second WAV recording of Andrew Ng's voice in the project root directory and name it:

andrew_voice.wav
Running the Application

Launch the application using:

python main.py

On the first run, the preprocessing pipeline will execute automatically. After that, the command-line interface will start and accept user queries.

Type your question and press Enter to interact with the agent.
Enter v to switch to voice input mode.
Enter quit to exit the application and save the current session memory.
