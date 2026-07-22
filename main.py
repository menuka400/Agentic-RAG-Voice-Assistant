"""
main.py — Entry point (FastAPI + Uvicorn).

Fix vs. previous version:
  - Replaced uvicorn.run("app:create_app", factory=True) with creating the
    FastAPI instance here and passing it directly to uvicorn.run().
    The string-based factory requires Uvicorn's import machinery to resolve
    "app" as a package, which breaks when main.py is run from the project root
    instead of from inside chatbot/. Passing the object directly is simpler
    and avoids any sys.path / cwd dependency.
"""

import threading
import webbrowser
import time
import sys
import os

# Ensure chatbot/ is on sys.path so `app` and `config` imports resolve
# regardless of which directory the user runs main.py from.
sys.path.insert(0, os.path.dirname(__file__))

import uvicorn
import config
import ingest

print("[chatbot] Checking RAG PDF ingest...")
try:
    ingest.main()
except Exception as e:
    print(f"[chatbot] Ingest failed: {e}")

from app import create_app

# Build the FastAPI application instance once at startup
app = create_app()


def open_browser():
    """Wait briefly then open the chat UI in the default browser."""
    time.sleep(1.2)  # Give Uvicorn a moment to bind the port
    webbrowser.open(f"http://{config.HOST}:{config.PORT}")


if __name__ == "__main__":

    threading.Thread(target=open_browser, daemon=True).start()

    print(f"[chatbot] Server starting at http://{config.HOST}:{config.PORT}")
    print(f"[chatbot] API docs at       http://{config.HOST}:{config.PORT}/docs")

    # Pass the app object directly — no import-string ambiguity
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        # NOTE: reload=True requires a string import path, not an object,
        # so we keep reload disabled here. Enable manually if needed:
        # uvicorn chatbot.app:create_app --factory --reload
    )
