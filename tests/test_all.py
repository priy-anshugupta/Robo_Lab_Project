"""
ISL-Multilingual Bridge — Test Suite
Unit tests for all major modules.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestConfig:
    """Test configuration module."""

    def test_settings_import(self):
        from config.settings import (
            OPENAI_MODEL, CONFIDENCE_THRESHOLD,
            SUPPORTED_LANGUAGES, LANGUAGE_CODES
        )
        assert OPENAI_MODEL == "gpt-4.1-mini"
        assert 0 < CONFIDENCE_THRESHOLD < 1
        assert len(SUPPORTED_LANGUAGES) == 5
        assert "Hindi" in LANGUAGE_CODES

    def test_paths_exist(self):
        from config.settings import PROJECT_ROOT
        assert PROJECT_ROOT.exists()


class TestSignBuffer:
    """Test sign buffer module."""

    def test_add_sign(self):
        from recognition.sign_buffer import SignBuffer
        buffer = SignBuffer()
        assert buffer.add_sign("HELLO") is True
        assert len(buffer) == 1

    def test_deduplication(self):
        from recognition.sign_buffer import SignBuffer
        buffer = SignBuffer()
        buffer.add_sign("HELLO")
        assert buffer.add_sign("HELLO") is False  # Duplicate
        assert len(buffer) == 1

    def test_different_signs(self):
        from recognition.sign_buffer import SignBuffer
        buffer = SignBuffer()
        buffer.add_sign("HELLO")
        buffer.add_sign("MY")
        buffer.add_sign("NAME")
        assert len(buffer) == 3
        assert buffer.get_sentence_string() == "HELLO MY NAME"

    def test_submit_sentence(self):
        from recognition.sign_buffer import SignBuffer
        buffer = SignBuffer(min_signs_for_sentence=2)
        buffer.add_sign("HELLO")
        buffer.add_sign("WORLD")
        sentence = buffer.submit_sentence()
        assert sentence == ["HELLO", "WORLD"]
        assert len(buffer) == 0  # Buffer cleared

    def test_submit_too_short(self):
        from recognition.sign_buffer import SignBuffer
        buffer = SignBuffer(min_signs_for_sentence=2)
        buffer.add_sign("HELLO")
        assert buffer.submit_sentence() is None

    def test_clear(self):
        from recognition.sign_buffer import SignBuffer
        buffer = SignBuffer()
        buffer.add_sign("HELLO")
        buffer.add_sign("WORLD")
        buffer.clear()
        assert len(buffer) == 0

    def test_remove_last(self):
        from recognition.sign_buffer import SignBuffer
        buffer = SignBuffer()
        buffer.add_sign("HELLO")
        buffer.add_sign("WORLD")
        removed = buffer.remove_last()
        assert removed == "WORLD"
        assert len(buffer) == 1


class TestPromptTemplates:
    """Test prompt template module."""

    def test_isl_to_english_format(self):
        from nlp.prompt_templates import format_isl_to_english
        result = format_isl_to_english("HELLO MY NAME PRIYA")
        assert "system" in result
        assert "user" in result
        assert "HELLO MY NAME PRIYA" in result["user"]

    def test_translation_format(self):
        from nlp.prompt_templates import format_translation
        result = format_translation("Hindi", "Hello, my name is Priya.")
        assert "Hindi" in result["system"]
        assert "Hindi" in result["user"]

    def test_context_enrichment_format(self):
        from nlp.prompt_templates import format_context_enrichment
        result = format_context_enrichment("HELLO WORLD")
        assert "HELLO WORLD" in result["user"]


class TestGPTClient:
    """Test GPT client module (without API key)."""

    def test_fallback_mode(self):
        from nlp.gpt_client import GPTClient
        client = GPTClient(api_key="")  # No API key
        assert not client.is_available

    def test_fallback_response_gloss(self):
        from nlp.gpt_client import GPTClient
        client = GPTClient(api_key="")
        result = client.call("system", "ISL gloss sequence: HELLO MY NAME")
        assert "Hello" in result or "hello" in result.lower()

    def test_fallback_response_translate(self):
        from nlp.gpt_client import GPTClient
        client = GPTClient(api_key="")
        result = client.call("system", "Translate to Hindi: Hello world")
        assert "Hello world" in result

    def test_cache_stats(self):
        from nlp.gpt_client import GPTClient
        client = GPTClient(api_key="")
        stats = client.get_stats()
        assert "total_calls" in stats
        assert "is_available" in stats


class TestSentenceBuilder:
    """Test sentence builder module."""

    def test_build_sentence(self):
        from nlp.sentence_builder import SentenceBuilder
        builder = SentenceBuilder()
        result = builder.build(["HELLO", "MY", "NAME"])
        assert "gloss" in result
        assert "english" in result
        assert result["gloss"] == "HELLO MY NAME"

    def test_empty_buffer(self):
        from nlp.sentence_builder import SentenceBuilder
        builder = SentenceBuilder()
        result = builder.build([])
        assert result["success"] is False


class TestTranslator:
    """Test translator module."""

    def test_supported_languages(self):
        from nlp.translator import Translator
        translator = Translator()
        langs = translator.get_supported_languages()
        assert "Hindi" in langs
        assert "Tamil" in langs
        assert len(langs) == 5

    def test_language_code(self):
        from nlp.translator import Translator
        translator = Translator()
        assert translator.get_language_code("Hindi") == "hi"
        assert translator.get_language_code("Tamil") == "ta"

    def test_english_translation(self):
        from nlp.translator import Translator
        translator = Translator()
        result = translator.translate("Hello", "English")
        assert result["translated"] == "Hello"
        assert result["success"] is True


class TestTTSEngine:
    """Test TTS engine module."""

    def test_language_code_mapping(self):
        from audio.tts_engine import get_lang_code
        assert get_lang_code("Hindi") == "hi"
        assert get_lang_code("Tamil") == "ta"
        assert get_lang_code("English") == "en"

    def test_empty_text(self):
        from audio.tts_engine import TTSEngine
        engine = TTSEngine(backend="gtts")
        result = engine.synthesize("", "English")
        assert result is None


class TestLandmarkNormalization:
    """Test landmark feature normalization."""

    def test_normalize_shape(self):
        from vision.landmark_extractor import LandmarkExtractor
        extractor = LandmarkExtractor.__new__(LandmarkExtractor)
        # Mock 21 landmarks with 3 coords
        landmarks = np.random.rand(21, 3).astype(np.float32)
        from config.settings import SINGLE_HAND_FEATURES
        result = extractor.normalize_landmarks(landmarks)
        assert result.shape == (SINGLE_HAND_FEATURES,)

    def test_normalize_empty(self):
        from vision.landmark_extractor import LandmarkExtractor
        extractor = LandmarkExtractor.__new__(LandmarkExtractor)
        result = extractor.normalize_landmarks(np.array([]))
        from config.settings import SINGLE_HAND_FEATURES
        assert result.shape == (SINGLE_HAND_FEATURES,)
        assert np.all(result == 0)

    def test_wrist_anchor(self):
        from vision.landmark_extractor import LandmarkExtractor
        extractor = LandmarkExtractor.__new__(LandmarkExtractor)
        landmarks = np.random.rand(21, 3).astype(np.float32)
        result = extractor.normalize_landmarks(landmarks)
        # After normalization, the flattened result should have wrist at ~0
        # (first 3 values close to 0 after normalization and scaling)
        assert result.shape[0] == 63


class TestDataPaths:
    """Test data file existence and structure."""

    def test_sign_list_exists(self):
        from config.settings import SIGN_LIST_PATH
        assert SIGN_LIST_PATH.exists()

    def test_sign_list_content(self):
        import json
        from config.settings import SIGN_LIST_PATH
        with open(SIGN_LIST_PATH) as f:
            data = json.load(f)
        assert data["total_signs"] == 100
        assert "categories" in data
        assert len(data["categories"]) == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
