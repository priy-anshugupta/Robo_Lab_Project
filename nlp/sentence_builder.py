"""
ISL-Multilingual Bridge — Sentence Builder
Converts ISL gloss word stream into grammatical English sentences
using GPT-4.1 Mini.
"""

import logging
from typing import List, Optional

from nlp.gpt_client import GPTClient
from nlp.prompt_templates import format_isl_to_english, format_context_enrichment

logger = logging.getLogger("isl-bridge.nlp.sentence")


class SentenceBuilder:
    """
    Converts ISL gloss sequences to grammatical English sentences.
    Uses GPT-4.1 Mini for grammar correction and context understanding.
    """

    def __init__(self, gpt_client: Optional[GPTClient] = None):
        """
        Args:
            gpt_client: Shared GPTClient instance (creates new one if None)
        """
        self.gpt = gpt_client or GPTClient()
        self.history: List[dict] = []

        logger.info("SentenceBuilder initialized")

    def build(self, sign_buffer: List[str], use_context: bool = False) -> dict:
        """
        Convert an ISL gloss sequence to a grammatical English sentence.

        Args:
            sign_buffer: List of ISL sign labels (e.g., ["HELLO", "MY", "NAME", "PRIYA"])
            use_context: If True, uses context enrichment prompt for noisy input

        Returns:
            dict with keys:
                - 'gloss': str — Original gloss string
                - 'english': str — Grammatical English sentence
                - 'success': bool — Whether conversion was successful
        """
        if not sign_buffer:
            return {"gloss": "", "english": "", "success": False}

        gloss = " ".join(s.upper() for s in sign_buffer)
        logger.info(f"Building sentence from gloss: {gloss}")

        # Choose prompt based on context mode
        if use_context:
            prompts = format_context_enrichment(gloss)
        else:
            prompts = format_isl_to_english(gloss)

        # Call GPT
        english = self.gpt.call(
            system_prompt=prompts["system"],
            user_message=prompts["user"],
            max_tokens=150,
            temperature=0.3,
        )

        result = {
            "gloss": gloss,
            "english": english,
            "success": bool(english and english != gloss),
        }

        # Store in history
        self.history.append(result)
        logger.info(f"Sentence built: '{gloss}' → '{english}'")

        return result

    def build_from_string(self, gloss_string: str) -> dict:
        """
        Convenience method to build from a gloss string.

        Args:
            gloss_string: Space-separated ISL gloss (e.g., "HELLO MY NAME PRIYA")

        Returns:
            Same as build()
        """
        signs = gloss_string.strip().split()
        return self.build(signs)

    def get_history(self) -> List[dict]:
        """Get the history of all built sentences."""
        return self.history.copy()

    def clear_history(self):
        """Clear the sentence history."""
        self.history.clear()
