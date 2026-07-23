"""
config.py — unchanged from Flask version.
Loads environment variables and exposes app-wide constants.
"""

import os
from dotenv import load_dotenv

# Load .env file from the project root (same directory as this file)
load_dotenv()

# --- Groq ---
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama-3.3-70b-versatile"  # Upgraded from llama-3.1-8b-instant for better tool-calling reliability
GROQ_WHISPER_MODEL: str = "whisper-large-v3-turbo"

# --- VAD (Voice Activity Detection) — TEN VAD preprocessing for STT ---
# Audio chunk size in samples fed to TEN VAD (using 256 samples = 16ms hop at 16kHz).
CHUNK_SIZE: int = 256

# Sample rate that TEN VAD and Whisper STT both expect.
VAD_SAMPLE_RATE: int = 16000

# Probability threshold above which a chunk is considered to contain speech.
# TEN VAD's default threshold is 0.5.
VAD_THRESHOLD: float = 0.5

# Seconds of consecutive non-speech before we consider a gap a silence break.
SILENCE_DURATION: float = 2.0

# Human voice fundamental frequency range (Hz).
# Used to filter out non-vocal noise (fans, keyboard clicks, etc.).
# (Increased MAX_HUMAN_FREQ to 3000.0 to account for speech formants)
MIN_HUMAN_FREQ: float = 50.0
MAX_HUMAN_FREQ: float = 3000.0

# --- STT Safeguards (Hallucination/Noise Filters) ---
# Minimum proportion of frames in a recording that must contain speech.
# Helps reject recordings that are mostly silence with a tiny blip of noise.
MIN_SPEECH_FRAME_RATIO: float = 0.15

# Minimum duration (seconds) of the cleaned audio to be considered valid speech.
MIN_SPEECH_DURATION_SECONDS: float = 0.3

# Minimum RMS energy required in the trimmed audio to be passed to STT.
# Rejects faint background noises that technically pass VAD but aren't speech.
MIN_RMS_ENERGY: float = 0.005

# Whisper-family models frequently hallucinate common phrases on silence/noise.
# If the STT output perfectly matches one of these (case-insensitive, ignoring punctuation),
# or is just repetitions of these, it is rejected.
HALLUCINATION_PHRASES: list[str] = [
    "thank you",
    "thanks for watching",
    "bye",
    "please subscribe",
    "you",
    "thank you.",
    "bye."
]

# --- ElevenLabs (Voice) ---
ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "Rachel")  # Default to Rachel for testing

# --- Server ---
HOST: str = "127.0.0.1"
PORT: int = 5000
DEBUG: bool = False  # Set True during development for Uvicorn auto-reload

# --- Chat History Memory Management ---
# Number of recent messages to keep verbatim before summarizing older ones.
MAX_RECENT_MESSAGES: int = 12
