"""
app/__init__.py — FastAPI application factory.

Changes vs. Flask version:
  - FastAPI() replaces Flask().
  - StaticFiles mount replaces Flask's automatic /static serving.
  - Jinja2Templates replaces Flask's render_template (set up here, used in routes).
  - APIRouter (from routes.py) replaces Blueprint.
  - No secret_key needed — session state is managed differently (see routes.py).
"""

import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Path helpers — resolve paths relative to project root (parent of app/)
APP_DIR = os.path.dirname(__file__)             # .../chatbot/app/
ROOT_DIR = os.path.dirname(APP_DIR)            # .../chatbot/


def create_app() -> FastAPI:
    """Instantiate and configure the FastAPI application."""

    app = FastAPI(
        title="AI Chatbot",
        description="Local chatbot powered by Groq llama-3.1-8b-instant.",
        version="1.0.0",
        # Docs available at /docs (Swagger UI) and /redoc — enabled by default
    )

    # Mount static files — equivalent to Flask's automatic /static/<filename>
    app.mount(
        "/static",
        StaticFiles(directory=os.path.join(ROOT_DIR, "static")),
        name="static",
    )

    # Register all routes via the APIRouter
    from app.routes import router
    app.include_router(router)

    return app
