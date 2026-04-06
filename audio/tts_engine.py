"""
ISL-Multilingual Bridge — Text-to-Speech Engine
Converts translated text to audio using gTTS (online) or pyttsx3 (offline fallback).
"""

import io
import logging
from typing import Optional

from config.settings import TTS_BACKEND, LANGUAGE_CODES

logger = logging.getLogger("isl-bridge.audio")


# Language name → gTTS language code mapping
GTTS_LANGUAGE_MAP = {
    "Hindi": "hi",
    "Tamil": "ta",
    "Bengali": "bn",
    "Telugu": "te",
    "Marathi": "mr",
    "English": "en",
}


class TTSEngine:
    """
    Factory-based Text-to-Speech engine.
    Supports gTTS (online, Google) and pyttsx3 (offline) backends.
    """

    def __init__(self, backend: str = TTS_BACKEND):
        """
        Args:
            backend: "gtts" (online) or "pyttsx3" (offline)
        """
        self.backend_name = backend.lower()
        self.backend = None

        if self.backend_name == "gtts":
            self.backend = gTTSBackend()
        elif self.backend_name == "pyttsx3":
            self.backend = Pyttsx3Backend()
        else:
            logger.warning(f"Unknown TTS backend '{backend}', defaulting to gTTS")
            self.backend = gTTSBackend()

        logger.info(f"TTSEngine initialized (backend={self.backend_name})")

    def synthesize(self, text: str, language: str = "English") -> Optional[bytes]:
        """
        Convert text to audio bytes.

        Args:
            text: Text to synthesize
            language: Language name (e.g., "Hindi", "Tamil")

        Returns:
            Audio bytes (MP3 format) or None on failure
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS")
            return None

        lang_code = get_lang_code(language)
        logger.info(f"Synthesizing ({language}/{lang_code}): '{text[:50]}...'")

        try:
            audio_bytes = self.backend.synthesize(text, lang_code)
            if audio_bytes:
                logger.info(f"Audio synthesized: {len(audio_bytes)} bytes")
            return audio_bytes
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")

            # Try fallback backend
            if self.backend_name == "gtts":
                logger.info("Falling back to pyttsx3...")
                try:
                    fallback = Pyttsx3Backend()
                    return fallback.synthesize(text, lang_code)
                except Exception as e2:
                    logger.error(f"Fallback TTS also failed: {e2}")

            return None

    def switch_backend(self, backend: str):
        """Switch TTS backend at runtime."""
        self.backend_name = backend.lower()
        if self.backend_name == "gtts":
            self.backend = gTTSBackend()
        elif self.backend_name == "pyttsx3":
            self.backend = Pyttsx3Backend()
        logger.info(f"TTS backend switched to: {self.backend_name}")


class gTTSBackend:
    """Google Text-to-Speech backend (online, high quality)."""

    def synthesize(self, text: str, lang_code: str = "en") -> Optional[bytes]:
        """
        Synthesize speech using Google TTS.

        Args:
            text: Text to speak
            lang_code: BCP-47 language code

        Returns:
            MP3 audio bytes
        """
        try:
            from gtts import gTTS

            tts = gTTS(text=text, lang=lang_code, slow=False)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            return audio_buffer.read()
        except ImportError:
            logger.error("gTTS not installed. Run: pip install gTTS")
            return None
        except Exception as e:
            logger.error(f"gTTS synthesis error: {e}")
            raise


class Pyttsx3Backend:
    """Offline Text-to-Speech backend using pyttsx3."""

    def __init__(self):
        self.engine = None
        try:
            import pyttsx3

            self.engine = pyttsx3.init()
            # Configure voice properties
            self.engine.setProperty("rate", 150)  # Words per minute
            self.engine.setProperty("volume", 0.9)
        except ImportError:
            logger.warning("pyttsx3 not installed. Run: pip install pyttsx3")
        except Exception as e:
            logger.warning(f"pyttsx3 initialization failed: {e}")

    def synthesize(self, text: str, lang_code: str = "en") -> Optional[bytes]:
        """
        Synthesize speech using pyttsx3 (saves to temp file, reads bytes).

        Note: pyttsx3 has limited language support compared to gTTS.
        It primarily supports English on most systems.

        Args:
            text: Text to speak
            lang_code: Language code (limited support)

        Returns:
            WAV audio bytes or None
        """
        if self.engine is None:
            return None

        try:
            import tempfile
            import os

            # Save to temp WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            self.engine.save_to_file(text, tmp_path)
            self.engine.runAndWait()

            # Read the audio bytes
            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()

            # Clean up
            os.unlink(tmp_path)
            return audio_bytes

        except Exception as e:
            logger.error(f"pyttsx3 synthesis error: {e}")
            return None


def get_lang_code(language_name: str) -> str:
    """
    Map language display name to BCP-47 code for TTS.

    Args:
        language_name: Language name (e.g., "Hindi", "Tamil")

    Returns:
        BCP-47 language code (e.g., "hi", "ta")
    """
    return GTTS_LANGUAGE_MAP.get(language_name, LANGUAGE_CODES.get(language_name, "en"))
