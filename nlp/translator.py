"""
ISL-Multilingual Bridge — Translator
Translates English sentences to target Indian languages using GPT-4.1 Mini.
"""

import logging
from typing import Optional

from config.settings import SUPPORTED_LANGUAGES, LANGUAGE_CODES
from nlp.gpt_client import GPTClient
from nlp.prompt_templates import format_translation

logger = logging.getLogger("isl-bridge.nlp.translator")


class Translator:
    """
    Translates English text to Indian languages via GPT-4.1 Mini.
    Supports: Hindi, Tamil, Bengali, Telugu, Marathi.
    """

    def __init__(self, gpt_client: Optional[GPTClient] = None):
        """
        Args:
            gpt_client: Shared GPTClient instance (creates new one if None)
        """
        self.gpt = gpt_client or GPTClient()
        self.supported_languages = SUPPORTED_LANGUAGES
        self.language_codes = LANGUAGE_CODES

        logger.info(f"Translator initialized (languages: {', '.join(self.supported_languages)})")

    def translate(self, english_text: str, target_language: str) -> dict:
        """
        Translate English text to the target language.

        Args:
            english_text: English sentence to translate
            target_language: Target language name (e.g., "Hindi")

        Returns:
            dict with keys:
                - 'english': str — Original English text
                - 'translated': str — Translated text
                - 'language': str — Target language name
                - 'language_code': str — BCP-47 language code
                - 'success': bool — Whether translation was successful
        """
        if not english_text:
            return {
                "english": "",
                "translated": "",
                "language": target_language,
                "language_code": self.language_codes.get(target_language, "en"),
                "success": False,
            }

        # Validate language
        if target_language not in self.supported_languages and target_language != "English":
            logger.warning(f"Unsupported language: {target_language}. Using Hindi.")
            target_language = "Hindi"

        # If target is English, no translation needed
        if target_language == "English":
            return {
                "english": english_text,
                "translated": english_text,
                "language": "English",
                "language_code": "en",
                "success": True,
            }

        logger.info(f"Translating to {target_language}: '{english_text[:50]}...'")

        # Format prompt
        prompts = format_translation(target_language, english_text)

        # Call GPT for translation
        translated = self.gpt.call(
            system_prompt=prompts["system"],
            user_message=prompts["user"],
            max_tokens=200,
            temperature=0.3,
        )

        lang_code = self.language_codes.get(target_language, "en")

        result = {
            "english": english_text,
            "translated": translated,
            "language": target_language,
            "language_code": lang_code,
            "success": bool(translated and translated != english_text),
        }

        logger.info(f"Translation ({target_language}): '{translated[:50]}...'")
        return result

    def translate_all(self, english_text: str) -> dict:
        """
        Translate English text to ALL supported languages.

        Args:
            english_text: English sentence to translate

        Returns:
            dict mapping language names to translation results
        """
        results = {}
        for lang in self.supported_languages:
            results[lang] = self.translate(english_text, lang)
        return results

    def get_language_code(self, language_name: str) -> str:
        """
        Map display name to BCP-47 language code.

        Args:
            language_name: Language display name (e.g., "Hindi")

        Returns:
            BCP-47 code (e.g., "hi")
        """
        return self.language_codes.get(language_name, "en")

    def get_supported_languages(self) -> list:
        """Get list of supported language names."""
        return self.supported_languages.copy()
