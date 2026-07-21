"""
app/schemas.py — NEW in FastAPI migration.

Pydantic models replace Flask's manual request.get_json() + ad-hoc dicts.
Benefits:
  - Automatic request validation (FastAPI returns 422 on bad input).
  - Self-documenting: models appear in /docs (Swagger UI) automatically.
  - Typed response contracts — easier to extend later (e.g. add token counts).
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Body expected by POST /api/chat."""
    message: str = Field(..., min_length=1, description="User message text")
    # session_id is optional here — routes.py creates one from cookie if absent
    session_id: str | None = Field(
        default=None,
        description="Session identifier (created server-side if not supplied)"
    )


class ChatResponse(BaseModel):
    """Successful response from POST /api/chat."""
    reply: str = Field(..., description="Assistant reply text")
    session_id: str = Field(..., description="Session ID for this conversation")


class ErrorResponse(BaseModel):
    """Error response from POST /api/chat."""
    error: str = Field(..., description="Human-readable error message")

class TTSRequest(BaseModel):
    """Body expected by POST /api/tts."""
    text: str = Field(..., min_length=1, description="Text to synthesize")
