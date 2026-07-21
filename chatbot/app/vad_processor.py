"""
app/vad_processor.py — TEN VAD + frequency-range preprocessing for STT.

PURPOSE:
  Browser MediaRecorder sends the full recorded clip (including leading/trailing
  silence, background noise, etc.) to /api/stt. This module filters that raw
  audio BEFORE it reaches the STT provider (ElevenLabs), improving transcription
  accuracy and reducing unnecessary API calls on empty recordings.

HOW IT WORKS:
  1. The uploaded audio (webm/ogg/etc.) is decoded and resampled to 16 kHz mono
     using scipy/numpy — the sample rate TEN VAD expects.
  2. Audio is processed in CHUNK_SIZE=256 sample windows.
  3. Each chunk is checked against TWO criteria:
       a) TEN VAD probability > VAD_THRESHOLD  (neural speech detector)
       b) Dominant frequency falls within MIN_HUMAN_FREQ..MAX_HUMAN_FREQ Hz
          (85–300 Hz = human voice fundamental, filters fans/AC/keyboard noise)
  4. Consecutive speech chunks are kept; silence is trimmed from head and tail.
  5. If NO speech is found anywhere → return None so the caller can respond
     with "No speech detected" instead of sending silence to ElevenLabs.

MODEL:
  Uses the open-source TEN VAD framework (from TEN-framework), replacing the
  previous Silero implementation for improved accuracy and lower resource usage.
  The model is loaded once at import time and reused for every request.
  Note: the default VAD_THRESHOLD is now 0.5, but may need tuning based on
  real-world testing in your environment.
"""

import io
import logging
import numpy as np
import scipy.io.wavfile
import scipy.signal

import config

logger = logging.getLogger(__name__)

import os
from ten_vad import TenVad

# ── Load model once at module import (shared across all requests) ─────────
logger.info(f"[vad_processor] Initializing TEN VAD (hop_size={config.CHUNK_SIZE}, threshold={config.VAD_THRESHOLD})")
# TEN VAD is initialized with hop_size matching our CHUNK_SIZE
_vad_model = TenVad(hop_size=config.CHUNK_SIZE, threshold=config.VAD_THRESHOLD)
logger.info("[vad_processor] TEN VAD model loaded OK.")


# ── Core detection helpers ────────────────────────────────────────────────

def get_dominant_frequency(chunk: np.ndarray, sample_rate: int) -> float:
    """
    Return the dominant frequency (Hz) of an audio chunk using FFT.

    Used to confirm a VAD hit is likely human speech (85–300 Hz range)
    rather than machinery hum, keyboard clicks, or broadband noise.
    """
    if len(chunk) == 0:
        return 0.0
    fft_vals = np.abs(np.fft.rfft(chunk))
    freqs    = np.fft.rfftfreq(len(chunk), d=1.0 / sample_rate)
    if fft_vals.max() == 0:
        return 0.0
    dominant = freqs[np.argmax(fft_vals)]
    return float(dominant)


def is_human_voice(chunk: np.ndarray, sample_rate: int) -> bool:
    """
    Return True if the chunk passes BOTH the TEN VAD check AND the
    human voice frequency range check.

    Args:
        chunk:       float32 numpy array, values in [-1, 1], length == CHUNK_SIZE
        sample_rate: must be 16000 for TEN VAD

    Returns:
        True  → likely human speech
        False → silence, noise, or non-vocal sound
    """
    # --- TEN VAD ---
    # Convert chunk to int16, which is typical for VAD frameworks (like WebRTC/TEN VAD)
    # If TEN VAD accepts float32 natively, this conversion can be bypassed.
    chunk_int16 = (chunk * 32767).astype(np.int16)
    
    # TEN VAD process returns probability and flag
    speech_prob, is_speech_flag = _vad_model.process(chunk_int16)

    if speech_prob < config.VAD_THRESHOLD:
        return False

    # --- Frequency range check ---
    dominant_freq = get_dominant_frequency(chunk, sample_rate)
    in_voice_range = config.MIN_HUMAN_FREQ <= dominant_freq <= config.MAX_HUMAN_FREQ
    return in_voice_range


# ── Audio loading helper ──────────────────────────────────────────────────

def _load_audio_as_float32_mono(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Decode raw audio bytes (any format readable by scipy) into a float32
    mono numpy array resampled to VAD_SAMPLE_RATE (16 kHz).

    Returns:
        (samples, original_sample_rate)

    Raises:
        ValueError if the audio cannot be decoded.
    """
    try:
        # scipy.io.wavfile works for PCM WAV.
        # Browser webm/ogg recordings need to be converted first via soundfile.
        try:
            import soundfile as sf
            buf = io.BytesIO(audio_bytes)
            samples, sr = sf.read(buf, dtype="float32", always_2d=False)
        except Exception:
            # Fallback: try raw WAV via scipy (works if browser sends wav)
            buf = io.BytesIO(audio_bytes)
            sr, samples = scipy.io.wavfile.read(buf)
            if samples.dtype != np.float32:
                samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max

        # Collapse stereo → mono
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        # Resample to 16 kHz if needed
        target_sr = config.VAD_SAMPLE_RATE
        if sr != target_sr:
            num_samples = int(len(samples) * target_sr / sr)
            samples = scipy.signal.resample(samples, num_samples)
            sr = target_sr

        return samples.astype(np.float32), sr

    except Exception as exc:
        raise ValueError(f"Could not decode audio: {exc}") from exc


# ── Main public function ──────────────────────────────────────────────────

def trim_to_speech(audio_bytes: bytes) -> bytes | None:
    """
    Run TEN VAD + frequency check on the uploaded audio and return a
    WAV-encoded bytes buffer containing only the speech segments, or None
    if no human speech was detected anywhere in the clip.

    This function is the ONLY entry point called by routes.py / voice_client.py.

    Args:
        audio_bytes: raw bytes of the uploaded audio file (webm, ogg, wav, …)

    Returns:
        bytes  → WAV-encoded cleaned audio, ready to send to STT
        None   → no human speech detected; caller should skip STT call
    """
    try:
        samples, sr = _load_audio_as_float32_mono(audio_bytes)
    except ValueError as exc:
        logger.warning(f"[vad_processor] Audio decode failed, skipping VAD: {exc}")
        # Return original bytes so STT can still attempt transcription
        return audio_bytes

    chunk_size      = config.CHUNK_SIZE
    silence_chunks  = int(config.SILENCE_DURATION * sr / chunk_size)

    speech_flags: list[bool] = []

    # Process audio in non-overlapping chunks
    for start in range(0, len(samples) - chunk_size + 1, chunk_size):
        chunk = samples[start : start + chunk_size]
        speech_flags.append(is_human_voice(chunk, sr))

    if not any(speech_flags):
        logger.info("[vad_processor] No human speech detected in recording.")
        return None

    # --- 1. STRICTER VAD ACCEPTANCE: Speech Frame Ratio Check ---
    # Require that at least a minimum proportion of frames contain speech.
    # This prevents short blips of noise in a mostly silent recording from passing VAD.
    speech_ratio = sum(speech_flags) / len(speech_flags)
    if speech_ratio < config.MIN_SPEECH_FRAME_RATIO:
        logger.info(f"[vad_processor] Speech frame ratio {speech_ratio:.2f} < {config.MIN_SPEECH_FRAME_RATIO}. Rejecting as no-speech.")
        return None

    # --- Trim leading silence ---
    first_speech = next(i for i, v in enumerate(speech_flags) if v)

    # --- Trim trailing silence (allow up to SILENCE_DURATION gap at end) ---
    last_speech = first_speech
    consecutive_silence = 0
    for i in range(first_speech, len(speech_flags)):
        if speech_flags[i]:
            last_speech = i
            consecutive_silence = 0
        else:
            consecutive_silence += 1

    # Convert chunk indices back to sample indices (add a half-second of padding)
    pad_chunks  = max(1, int(0.5 * sr / chunk_size))
    start_sample = max(0, (first_speech - pad_chunks) * chunk_size)
    end_sample   = min(len(samples), (last_speech + 1 + pad_chunks) * chunk_size)

    cleaned = samples[start_sample:end_sample]

    if len(cleaned) == 0:
        return None

    # --- 2. MINIMUM DURATION CHECK ---
    # Ensure the cleaned audio is long enough to contain meaningful speech.
    duration_sec = len(cleaned) / sr
    if duration_sec < config.MIN_SPEECH_DURATION_SECONDS:
        logger.info(f"[vad_processor] Trimmed audio duration {duration_sec:.2f}s < {config.MIN_SPEECH_DURATION_SECONDS}s. Rejecting.")
        return None

    # --- 3. AUDIO ENERGY CHECK ---
    # Ensure the audio actually has some volume/energy, avoiding faint background noise 
    # that technically triggered VAD but isn't clear speech.
    rms_energy = np.sqrt(np.mean(cleaned**2))
    if rms_energy < config.MIN_RMS_ENERGY:
        logger.info(f"[vad_processor] RMS energy {rms_energy:.5f} < {config.MIN_RMS_ENERGY}. Rejecting as faint noise.")
        return None

    n_original_chunks = len(speech_flags)
    n_kept_chunks     = (end_sample - start_sample) // chunk_size
    logger.info(
        f"[vad_processor] VAD trimmed {n_original_chunks} → {n_kept_chunks} chunks "
        f"({100*n_kept_chunks/max(n_original_chunks,1):.0f}% kept)"
    )

    # Encode the cleaned audio back to in-memory WAV (16 kHz, float32)
    out_buf = io.BytesIO()
    scipy.io.wavfile.write(out_buf, sr, cleaned)
    out_buf.seek(0)
    return out_buf.read()
