"""
app/routes.py — URL handlers, migrated to FastAPI APIRouter.

Changes vs. Flask version:
  - Blueprint replaced with APIRouter.
  - All route functions are now `async def` — required to `await` groq_client.
  - HTML page served with Jinja2Templates.TemplateResponse instead of render_template().
  - Request body validated automatically by Pydantic (ChatRequest model).
  - Response typed with ChatResponse / ErrorResponse for /docs visibility.
  - Session state: Flask's signed cookie session replaced with a UUID passed
    in the request body (session_id field). Frontend sends it back each turn.
    This is simpler, stateless on the server, and works fine for single-tab use.
"""

import uuid
import os
import io
from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import JSONResponse, Response
from fastapi.templating import Jinja2Templates

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app import chat_history
from app import agent
from app.schemas import ChatRequest, ChatResponse, ErrorResponse, TTSRequest

# ── Setup ─────────────────────────────────────────────────────
router = APIRouter()

# Templates directory — resolve relative to project root
APP_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(APP_DIR)
templates = Jinja2Templates(directory=os.path.join(ROOT_DIR, "templates"))

# System prompt — personality/behaviour, unchanged from Flask version
SYSTEM_PROMPT = (
    "You are a helpful, concise AI assistant. "
    "Answer clearly and accurately. If you don't know something, say so."
)


# ── Routes ─────────────────────────────────────────────────────

@router.get("/", include_in_schema=False)
async def index(request: Request):
    """
    Serve the main chat page.
    Starlette >= 0.36 changed TemplateResponse to require keyword arguments.
    Old API: TemplateResponse("index.html", {"request": request})  ← broken
    New API: TemplateResponse(request=request, name="index.html")  ← correct
    """
    return templates.TemplateResponse(request=request, name="index.html")


@router.post(
    "/api/chat",
    response_model=ChatResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Send a message and get an AI reply",
)
async def chat(body: ChatRequest):
    """
    Handle a chat message from the frontend.

    - Accepts: ChatRequest  { message: str, session_id: str | None }
    - Returns: ChatResponse { reply: str, session_id: str }
      or       ErrorResponse{ error: str } on failure (HTTP 503)

    session_id flow:
      1. First request: frontend sends session_id=null → server creates one → returns it.
      2. Subsequent requests: frontend sends back the same session_id → history is preserved.
    """
    # Create session ID on first turn; subsequent turns reuse existing one
    session_id = body.session_id or str(uuid.uuid4())

    # Persist user turn
    await chat_history.add_message(session_id, role="user", content=body.message)

    # Call Agent (async — non-blocking)
    try:
        history = chat_history.get_history(session_id)
        
        # Convert dict history to LangChain BaseMessage objects
        langchain_messages = []
        for msg in history:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
            elif msg.get("is_summary"):
                langchain_messages.append(SystemMessage(content=msg["content"]))

        reply = await agent.run_agent(langchain_messages, system_prompt=SYSTEM_PROMPT)
    except RuntimeError as exc:
        # Roll back the user message so history stays consistent
        chat_history.get_history(session_id).pop()
        return JSONResponse(
            status_code=503,
            content={"error": str(exc)},
        )

    # Persist assistant turn
    await chat_history.add_message(session_id, role="assistant", content=reply)

    return ChatResponse(reply=reply, session_id=session_id)


@router.post("/api/tts", summary="Generate TTS audio from text")
async def tts(body: TTSRequest):
    """
    Generate audio for testing using ElevenLabs TTS.
    """
    from app import voice_client
    try:
        audio_bytes = await voice_client.generate_tts(body.text)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"error": str(exc)},
        )

@router.post("/api/stt", summary="Transcribe audio to text")
async def stt(audio: UploadFile = File(...)):
    """
    Transcribe uploaded audio to text.

    Pipeline:
      1. Read raw audio bytes from the browser upload.
      2. Run TEN VAD + frequency-range check (vad_processor.trim_to_speech)
         to strip silence and background noise BEFORE sending to the STT provider.
         This preprocessing step improves transcription accuracy and avoids
         wasting API quota on empty or noise-only recordings.
      3. Pass the cleaned audio to voice_client.generate_stt (ElevenLabs STT).
         The STT call itself is unchanged — only its input is cleaner.
    """
    from app import voice_client
    from app import vad_processor

    try:
        # Read the full audio upload into memory once
        raw_bytes = await audio.read()

        # --- VAD preprocessing step ---
        cleaned_bytes = vad_processor.trim_to_speech(raw_bytes)

        if cleaned_bytes is None:
            # No human speech detected — skip STT call entirely
            return {"text": "", "vad": "no_speech"}

        # --- STT call — unchanged; receives cleaned audio instead of raw ---
        text = await voice_client.generate_stt(io.BytesIO(cleaned_bytes))
        return {"text": text}

    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"error": str(exc)},
        )

