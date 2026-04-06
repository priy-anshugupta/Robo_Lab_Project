"""
ISL-Multilingual Bridge — GPT Client
Wrapper for OpenAI API calls with retry logic, caching, and error handling.
"""

import time
import logging
from typing import Optional

from config.settings import OPENAI_API_KEY, OPENAI_MODEL

logger = logging.getLogger("isl-bridge.nlp.gpt")


class GPTClient:
    """
    Wrapper for OpenAI GPT-4.1 Mini API with:
    - Retry with exponential backoff
    - Session-level response caching
    - Error handling and fallback
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = OPENAI_MODEL,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        """
        Args:
            api_key: OpenAI API key (falls back to env variable)
            model: Model name (default: gpt-4.1-mini)
            max_retries: Maximum retry attempts on failure
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.client = None
        self.is_available = False

        # Session-level response cache
        self._cache: dict = {}
        self._cache_hits: int = 0
        self._total_calls: int = 0

        self._initialize_client()

    def _initialize_client(self):
        """Initialize the OpenAI client."""
        if not self.api_key or self.api_key == "sk-your-api-key-here":
            logger.warning(
                "OpenAI API key not configured. GPT features will use fallback mode. "
                "Set OPENAI_API_KEY in your .env file."
            )
            self.is_available = False
            return

        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key, timeout=self.timeout)
            self.is_available = True
            logger.info(f"GPT client initialized (model={self.model})")
        except ImportError:
            logger.warning("OpenAI package not installed. Run: pip install openai")
            self.is_available = False
        except Exception as e:
            logger.error(f"Failed to initialize GPT client: {e}")
            self.is_available = False

    def call(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 300,
        temperature: float = 0.3,
        use_cache: bool = True,
    ) -> str:
        """
        Make a GPT API call with retry logic and caching.

        Args:
            system_prompt: System message for the model
            user_message: User message / query
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (lower = more deterministic)
            use_cache: Whether to use/store cached responses

        Returns:
            Model response text. Returns fallback message on failure.
        """
        self._total_calls += 1

        # Check cache first
        if use_cache:
            cache_key = (system_prompt[:100], user_message, self.model)
            if cache_key in self._cache:
                self._cache_hits += 1
                logger.debug(f"Cache hit ({self._cache_hits}/{self._total_calls} total)")
                return self._cache[cache_key]

        # Fallback if API not available
        if not self.is_available:
            return self._fallback_response(user_message)

        # Retry with exponential backoff
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                result = response.choices[0].message.content.strip()

                # Cache the result
                if use_cache:
                    self._cache[cache_key] = result

                logger.debug(f"GPT call successful (attempt {attempt + 1})")
                return result

            except Exception as e:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(
                    f"GPT call failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)

        logger.error("All GPT retry attempts failed. Returning fallback.")
        return self._fallback_response(user_message)

    def _fallback_response(self, user_message: str) -> str:
        """
        Generate a basic fallback response when API is unavailable.
        Attempts simple word-level processing without LLM.

        Args:
            user_message: The original user message

        Returns:
            Best-effort fallback text
        """
        # If it's a gloss sequence, just capitalize and add punctuation
        if "ISL gloss sequence:" in user_message:
            gloss = user_message.replace("ISL gloss sequence:", "").strip()
            words = gloss.split()
            if words:
                sentence = " ".join(w.capitalize() for w in words) + "."
                return sentence

        # If it's a translation request, return the original text
        if "Translate to" in user_message:
            # Extract the English text
            parts = user_message.split(":")
            if len(parts) > 1:
                return parts[-1].strip() + " (translation unavailable)"

        return user_message

    def clear_cache(self):
        """Clear the response cache."""
        self._cache.clear()
        self._cache_hits = 0
        logger.info("GPT cache cleared")

    def get_stats(self) -> dict:
        """Get client usage statistics."""
        return {
            "total_calls": self._total_calls,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache),
            "is_available": self.is_available,
            "model": self.model,
        }
