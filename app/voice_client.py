"""
app/voice_client.py — TTS/STT client for testing and prototyping.

NOTE: This uses ElevenLabs for TTS and Groq's Whisper API for STT.
It is isolated from routes.py and agent.py so it can be swapped out later.
"""
import io
import logging
import config
from elevenlabs.client import ElevenLabs
from groq import AsyncGroq

logger = logging.getLogger(__name__)

def _get_elevenlabs_client() -> ElevenLabs:
    """Create the ElevenLabs client for TTS, validating the API key is present."""
    if not config.ELEVENLABS_API_KEY:
        raise ValueError("ELEVENLABS_API_KEY is not set. Add it to your .env file.")
    return ElevenLabs(api_key=config.ELEVENLABS_API_KEY)

def _get_groq_client() -> AsyncGroq:
    """Create the Groq client for STT."""
    if not config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")
    return AsyncGroq(api_key=config.GROQ_API_KEY)

async def generate_tts(text: str) -> bytes:
    """Generate TTS audio (MP3) from text using ElevenLabs."""
    try:
        client = _get_elevenlabs_client()
        # Generate audio using the configured voice ID
        audio_generator = client.text_to_speech.convert(
            voice_id=config.ELEVENLABS_VOICE_ID,
            output_format="mp3_44100_128",
            text=text,
            model_id="eleven_flash_v2"
        )
        # The generator yields chunks of bytes
        audio_bytes = b"".join(list(audio_generator))
        return audio_bytes
    except ValueError as e:
        raise RuntimeError(str(e))
    except Exception as e:
        logger.error("ElevenLabs TTS call failed!", exc_info=True)
        if hasattr(e, 'status_code'):
            logger.error(f"Status Code: {e.status_code}")
        if hasattr(e, 'body'):
            logger.error(f"Response Body: {e.body}")
        raise RuntimeError("ElevenLabs TTS failed. Please check the server logs for details.")

async def generate_stt(audio_file_like: io.IOBase) -> str:
    """Transcribe audio (STT) to text using Groq's Whisper API."""
    try:
        client = _get_groq_client()
        
        # Read the audio bytes from the uploaded file
        audio_bytes = audio_file_like.read()
        
        transcription = await client.audio.transcriptions.create(
            file=("recording.webm", audio_bytes),
            model=config.GROQ_WHISPER_MODEL,
            language="en"
        )
        
        text = transcription.text.strip()
        
        # --- 4. HALLUCINATION PHRASE FILTER ---
        # Whisper-family models hallucinate common phrases on silence/noise.
        # Check if the result is composed entirely of known hallucination phrases.
        import re
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        
        remainder = clean_text
        for phrase in config.HALLUCINATION_PHRASES:
            clean_phrase = re.sub(r'[^\w\s]', '', phrase.lower()).strip()
            if clean_phrase:
                remainder = remainder.replace(clean_phrase, '')
                
        # If there's nothing left but whitespace, it was purely a hallucination
        if not remainder.strip() and clean_text.strip():
            logger.info(f"Filtered hallucinated STT phrase: '{text}'")
            return "No clear speech detected, please try again"

        return text
    except ValueError as e:
        raise RuntimeError(str(e))
    except Exception as e:
        logger.error("Groq Whisper STT call failed!", exc_info=True)
        if hasattr(e, 'status_code'):
            logger.error(f"Status Code: {e.status_code}")
        if hasattr(e, 'body'):
            logger.error(f"Response Body: {e.body}")
        raise RuntimeError("Groq STT failed. Please check the server logs for details.")
