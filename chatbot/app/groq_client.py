"""
app/groq_client.py — Groq API logic, now async.

Changes vs. Flask version:
  - Uses groq.AsyncGroq instead of groq.Groq so route handlers can await it
    without blocking Uvicorn's event loop (which would negate async benefits).
  - Function signature unchanged — routes.py just adds `await`.
  - Still fully decoupled from routing: to add RAG or tool-calling later,
    only this file changes.
"""

from typing import List, Dict, Optional
from groq import AsyncGroq, APIError, AuthenticationError, RateLimitError
import config


def _get_client() -> AsyncGroq:
    """Create the async Groq client, validating the API key is present."""
    if not config.GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )
    return AsyncGroq(api_key=config.GROQ_API_KEY)


async def send_message(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
) -> str:
    """Send a conversation to the Groq API and return the assistant reply.

    Now async — must be awaited by the caller.
    Using AsyncGroq means the HTTP call is non-blocking: Uvicorn can serve
    other requests while waiting for Groq's response.

    Args:
        messages: Full conversation history in Groq format:
                  [{"role": "user"|"assistant", "content": "..."}, ...]
        system_prompt: Optional system instruction prepended to the conversation.

    Returns:
        The assistant's reply text.

    Raises:
        RuntimeError: Wraps Groq API errors with a user-friendly message.
    """
    try:
        client = _get_client()

        # Prepend system message if provided
        full_messages: List[Dict[str, str]] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        # `await` here — AsyncGroq's create() is a coroutine
        response = await client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=full_messages,
            temperature=0.7,
            max_tokens=1024,
        )

        return response.choices[0].message.content

    except AuthenticationError:
        raise RuntimeError(
            "Invalid Groq API key. Check your .env file."
        )
    except RateLimitError:
        raise RuntimeError(
            "Groq rate limit reached. Please wait a moment and try again."
        )
    except ValueError as exc:
        raise RuntimeError(str(exc))
    except APIError as exc:
        raise RuntimeError(f"Groq API error: {exc}")
    except Exception as exc:
        raise RuntimeError(f"Unexpected error: {exc}")
