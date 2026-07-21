# Agentic RAG Voice Assistant

A locally-run chatbot web app with agentic tool calling and voice capabilities. Built for learning and portfolio purposes, with plans to expand into full Retrieval-Augmented Generation (RAG). Automatically opens in your browser when started.

## Table of Contents
- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [File Structure](#file-structure)
- [Setup / Installation](#setup--installation)
- [Challenges & Solutions](#challenges--solutions)
- [Roadmap / Future Enhancements](#roadmap--future-enhancements)
- [License](#license)

## Project Overview
- **Core Engine:** Built with FastAPI (backend), LangChain + LangGraph (agentic orchestration), and Groq API for high-speed LLM inference (using `llama-3.3-70b-versatile`).
- **Agentic Tool Use:** Features real-time web search (via DuckDuckGo) and a custom timezone-aware date/time tool. The agent autonomously decides when to use tools to answer queries.
- **Voice Interaction:** Supports Speech-to-Text (STT) and Text-to-Speech (TTS) via ElevenLabs API (currently using Groq Whisper API for STT). Integrates TEN VAD (Voice Activity Detection) as a preprocessing layer to filter out silence and background noise before transcription, combined with a custom frequency-range human-voice check.
- **Memory Management:** Session-based conversation memory (in-memory, resets on restart) with automatic summarization of older messages to prevent exceeding the LLM context window.
- **UI:** A clean, locally hosted web UI that properly renders Markdown (bullet points, bold text, etc.).
- **Open-Source Focused:** Uses 100% free/open-source tools wherever possible. ElevenLabs (TTS/STT) is currently used within its free tier, with future plans to replace it with fully self-hosted, open-source alternatives like Piper.

## Tech Stack
- **Python**
- **FastAPI / Uvicorn**
- **LangChain / LangGraph**
- **Groq API** (`llama-3.3-70b-versatile`)
- **DuckDuckGo Search**
- **TEN VAD**
- **ElevenLabs API** (TTS/STT)
- **Jinja2 Templates**
- **Vanilla JS / HTML / CSS** (Frontend)

## Features
- Real-time web search for current events/facts
- Accurate date/time lookup by timezone
- Voice input (STT) and voice output (TTS)
- Background noise filtering via VAD before transcription
- Session-based conversation memory with long-conversation summarization
- Clean Markdown-rendered chat responses

## File Structure
```text
chatbot/
├── main.py
├── config.py
├── requirements.txt
├── .env.example
├── app/
│   ├── routes.py
│   ├── agent.py
│   ├── tools.py
│   ├── chat_history.py
│   ├── voice_client.py
│   ├── vad_processor.py
│   ├── schemas.py
│   └── groq_client.py
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── chat.js
└── templates/
    └── index.html
```

## Setup / Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd chatbot
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   *Note: Git must be installed on your system to fetch TEN VAD.*
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Create a `.env` file in the root directory (use `.env.example` as a template) and add your API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key
   ELEVENLABS_API_KEY=your_elevenlabs_api_key
   ELEVENLABS_VOICE_ID=your_voice_id
   ```

5. **Run the application:**
   ```bash
   python main.py
   ```
   The browser will automatically open to `http://127.0.0.1:5000`.

## Challenges & Solutions

1. **Deprecated Groq Models Causing 400 Errors**
   - *Problem:* API returned errors when attempting to use older models (e.g., `llama3-groq-8b-8192-tool-use-preview`).
   - *Cause:* The model was decommissioned by Groq.
   - *Solution:* Switched to actively supported, production-ready models.

2. **Stale Tool Results Leaking Across Turns**
   - *Problem:* The agent would use old search results to answer unrelated new questions.
   - *Cause:* Improper state handling and lack of message history isolation in the LangGraph agent state.
   - *Solution:* Fixed message history isolation and ensured state is correctly passed/cleared between turns in the graph.

3. **Small LLM Unreliability with Tool-Calling**
   - *Problem:* `llama-3.1-8b-instant` struggled with complex tool use, generating malformed function calls, hallucinating answers instead of searching, or inverting logic (e.g., currency conversion directions).
   - *Cause:* The 8B model lacked the reasoning capacity for consistent autonomous agentic behavior.
   - *Solution:* Upgraded to a larger model (`llama-3.3-70b-versatile`), consolidated system prompts, and added a self-verification node in the agent graph to double-check output before returning it.

4. **ElevenLabs TTS Failures**
   - *Problem:* API calls to ElevenLabs failed with authentication/permission errors.
   - *Cause:* Attempted to use generic Voice Library voice IDs, which are not accessible on the free-tier API. The API requires account-specific "My Voices".
   - *Solution:* Programmatically listed account-accessible voices and updated the config to use a confirmed, accessible voice ID.

5. **Whisper STT Hallucinations on Silence**
   - *Problem:* The STT model would frequently return phrases like "Thank you" or "Bye" when fed silence or faint noise.
   - *Cause:* A known failure mode of Whisper-family models where they output common training-data phrases for low-confidence or silent audio.
   - *Solution:* Added strict pre-STT safeguards: stricter VAD acceptance thresholds, minimum speech duration checks, audio RMS energy checks, and a post-STT hallucination-phrase filter (rejecting exact matches to known hallucinations).

6. **Frontend Markdown Not Rendering**
   - *Problem:* Bullet points, bold text, and lists were displayed as raw text symbols in the chat UI.
   - *Cause:* The frontend was inserting the raw text into the DOM as text nodes rather than parsing it as HTML.
   - *Solution:* Integrated a Markdown-to-HTML renderer (e.g., `marked.js`) in the frontend chat script.

7. **Accidental Repo Restructuring**
   - *Problem:* The project was pushed to GitHub with an extra wrapping folder, breaking standard pathing.
   - *Cause:* Committing from the wrong root directory during initialization.
   - *Solution:* Moved files to the true repository root using `git mv` (or Windows `Move-Item`) and re-committed for a clean structure.

## Roadmap / Future Enhancements

- [ ] Add RAG (Retrieval-Augmented Generation) support — ingest PDFs/documents (starting with technical books/papers) into a vector database (ChromaDB) with local embeddings, and add a search_documents tool to the existing agent alongside web_search
- [ ] Replace ElevenLabs (TTS/STT) with fully open-source, self-hosted alternatives (faster-whisper for STT, Piper for TTS) for a 100% free/open-source stack
- [ ] Redesign the chat UI with a modern voice-assistant aesthetic (dark theme, animated waveform visualization reacting to live audio)
- [ ] Add an Interrupt button — allow the user to stop the bot mid-response/mid-speech
- [ ] Add a Mute button — toggle TTS audio output on/off without disabling the feature entirely
- [ ] Add a Hold/Pause button — pause the conversation/recording without ending the session
- [ ] Persistent conversation history (optional file-based or database storage across server restarts)
- [ ] Multi-document RAG source management as more PDFs/references are added over time

## License
License: MIT
