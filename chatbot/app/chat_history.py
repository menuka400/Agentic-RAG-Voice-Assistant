"""
app/chat_history.py — In-memory conversation history manager.

Unchanged from Flask version — fully framework-agnostic.
Design goals:
  - Routes never touch raw lists; they call methods here.
  - To swap for file/DB storage later: only change this file, not routes.py.
  - Each session gets its own history keyed by session_id (UUID string).

Session IDs are now passed explicitly in the request body (see schemas.py),
rather than via Flask's signed cookie session.
"""

from typing import List, Dict
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
import config

logger = logging.getLogger(__name__)

# Module-level store: { session_id: [{"role": ..., "content": ...}, ...] }
_histories: Dict[str, List[Dict[str, str]]] = {}


def get_history(session_id: str) -> List[Dict[str, str]]:
    """Return the message list for this session (creates empty list if new)."""
    if session_id not in _histories:
        _histories[session_id] = []
    return _histories[session_id]


async def add_message(session_id: str, role: str, content: str) -> None:
    """Append one message to this session's history and trim if necessary.

    Args:
        session_id: Unique identifier for the user session.
        role: 'user' or 'assistant' (Groq API convention).
        content: The message text.
    """
    history = get_history(session_id)
    history.append({"role": role, "content": content})
    
    # Check if we need to summarize older messages to prevent the context window from blowing up
    await _trim_history_if_needed(session_id)


async def _trim_history_if_needed(session_id: str) -> None:
    """
    Summarize older messages if the history exceeds the configured threshold.
    This preserves the short-term context (recent messages) verbatim, while older 
    context is compressed into a single summary message using the LLM.
    """
    history = get_history(session_id)
    max_recent = getattr(config, "MAX_RECENT_MESSAGES", 12)
    # We trigger summarization if the history is larger than max_recent + a small buffer (e.g. 2).
    # This avoids calling the LLM to summarize every single turn once we hit the limit.
    trigger_threshold = max_recent + 2
    
    if len(history) <= trigger_threshold:
        return

    # Keep the most recent `max_recent` messages verbatim
    recent_messages = history[-max_recent:]
    # The messages to summarize are everything before the recent window
    older_messages = history[:-max_recent]
    
    try:
        llm = ChatGroq(
            model=config.GROQ_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0,
        )
        
        # Format the older messages for the summarization prompt
        transcript = ""
        for msg in older_messages:
            msg_role = "Previous Summary" if msg.get("is_summary") else msg.get("role", "user").capitalize()
            transcript += f"{msg_role}: {msg.get('content')}\n\n"
            
        summary_prompt = (
            "You are a helpful AI assistant tasked with summarizing a conversation. "
            "Below is a transcript of older messages from an ongoing conversation, which may include a previous summary. "
            "Please provide a single, concise summary of all the key information, facts, and context from these messages. "
            "The goal is to retain important details that the AI might need later in the conversation, while saving space. "
            "Do not include the recent messages, just summarize what is provided below.\n\n"
            f"TRANSCRIPT:\n{transcript}"
        )
        
        # We use ainvoke so this doesn't block the FastAPI event loop
        response = await llm.ainvoke([SystemMessage(content=summary_prompt)])
        
        # Replace the older messages with a single summary message
        summary_message = {
            "role": "system", 
            "content": f"Summary of earlier conversation:\n{response.content}",
            "is_summary": True
        }
        
        # Update the session history
        _histories[session_id] = [summary_message] + recent_messages
        logger.info(f"Summarized {len(older_messages)} older messages for session {session_id}.")
        
    except Exception as e:
        # If summarization fails (e.g. API error or rate limit), log it and skip this turn.
        # We will try again on the next turn when the threshold check is hit.
        logger.warning(f"Failed to summarize chat history: {e}")


def clear_history(session_id: str) -> None:
    """Wipe history for a session (useful for 'New Chat' feature later)."""
    _histories.pop(session_id, None)
