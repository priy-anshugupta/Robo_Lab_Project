"""
ISL-Multilingual Bridge — Sign Buffer Module
Accumulates recognized signs into a sentence buffer with deduplication
and sentence trigger logic (pause detection or manual submit).
"""

import time
import logging
from collections import deque
from typing import Optional, List

from config.settings import SENTENCE_PAUSE_TIMEOUT

logger = logging.getLogger("isl-bridge.recognition.buffer")


class SignBuffer:
    """
    Accumulates recognized ISL signs into a sentence buffer.
    Provides deduplication, pause-based sentence triggering,
    and buffer management.
    """

    def __init__(
        self,
        max_length: int = 20,
        pause_timeout: float = SENTENCE_PAUSE_TIMEOUT,
        min_signs_for_sentence: int = 2,
    ):
        """
        Args:
            max_length: Maximum number of signs in the buffer
            pause_timeout: Seconds of no input to trigger sentence completion
            min_signs_for_sentence: Minimum signs needed to form a sentence
        """
        self.max_length = max_length
        self.pause_timeout = pause_timeout
        self.min_signs_for_sentence = min_signs_for_sentence

        self.buffer: deque = deque(maxlen=max_length)
        self.last_sign_time: Optional[float] = None
        self.last_added_sign: Optional[str] = None
        self.consecutive_count: int = 0
        self.total_signs_processed: int = 0

        logger.info(
            f"SignBuffer initialized (max={max_length}, "
            f"pause={pause_timeout}s, min_signs={min_signs_for_sentence})"
        )

    def add_sign(self, sign: str, confidence: float = 1.0) -> bool:
        """
        Add a recognized sign to the buffer with deduplication.

        Args:
            sign: Recognized sign label (e.g., "HELLO")
            confidence: Prediction confidence (for logging)

        Returns:
            True if the sign was added (not a duplicate)
        """
        current_time = time.time()

        # Deduplication: collapse consecutive identical signs
        if sign == self.last_added_sign:
            self.consecutive_count += 1
            self.last_sign_time = current_time
            # Don't add duplicate but update timestamp
            return False

        # Add new sign to buffer
        self.buffer.append(sign.upper())
        self.last_added_sign = sign
        self.last_sign_time = current_time
        self.consecutive_count = 1
        self.total_signs_processed += 1

        logger.debug(f"Sign added: {sign} (conf={confidence:.2f}, buffer_len={len(self.buffer)})")
        return True

    def should_trigger_sentence(self) -> bool:
        """
        Check if a sentence should be triggered based on pause detection.

        Returns:
            True if pause timeout exceeded and buffer has enough signs
        """
        if len(self.buffer) < self.min_signs_for_sentence:
            return False

        if self.last_sign_time is None:
            return False

        elapsed = time.time() - self.last_sign_time
        return elapsed >= self.pause_timeout

    def get_sentence_gloss(self) -> List[str]:
        """
        Get the current buffer contents as a list of sign labels (gloss).

        Returns:
            List of sign labels forming the ISL gloss
        """
        return list(self.buffer)

    def get_sentence_string(self) -> str:
        """
        Get the current buffer contents as a space-separated string.

        Returns:
            Space-separated ISL gloss string (e.g., "HELLO MY NAME PRIYA")
        """
        return " ".join(self.buffer)

    def submit_sentence(self) -> Optional[List[str]]:
        """
        Manually submit the current buffer as a complete sentence.
        Clears the buffer after submission.

        Returns:
            List of sign labels if buffer has enough signs, None otherwise
        """
        if len(self.buffer) < self.min_signs_for_sentence:
            logger.warning(
                f"Buffer too short for sentence ({len(self.buffer)}/{self.min_signs_for_sentence})"
            )
            return None

        sentence = list(self.buffer)
        logger.info(f"Sentence submitted: {' '.join(sentence)}")
        self.clear()
        return sentence

    def auto_submit_if_ready(self) -> Optional[List[str]]:
        """
        Automatically submit if pause timeout is exceeded.

        Returns:
            List of sign labels if sentence triggered, None otherwise
        """
        if self.should_trigger_sentence():
            return self.submit_sentence()
        return None

    def clear(self):
        """Clear the buffer and reset state."""
        self.buffer.clear()
        self.last_sign_time = None
        self.last_added_sign = None
        self.consecutive_count = 0
        logger.debug("SignBuffer cleared")

    def remove_last(self) -> Optional[str]:
        """
        Remove the last sign from the buffer (undo).

        Returns:
            The removed sign, or None if buffer was empty
        """
        if self.buffer:
            removed = self.buffer.pop()
            self.last_added_sign = self.buffer[-1] if self.buffer else None
            logger.debug(f"Last sign removed: {removed}")
            return removed
        return None

    def get_status(self) -> dict:
        """
        Get current buffer status for UI display.

        Returns:
            dict with buffer info
        """
        time_since_last = None
        if self.last_sign_time:
            time_since_last = time.time() - self.last_sign_time

        return {
            "signs": list(self.buffer),
            "count": len(self.buffer),
            "max_length": self.max_length,
            "last_sign": self.last_added_sign,
            "time_since_last_sign": time_since_last,
            "is_sentence_ready": self.should_trigger_sentence(),
            "total_processed": self.total_signs_processed,
        }

    def __len__(self) -> int:
        return len(self.buffer)

    def __repr__(self) -> str:
        return f"SignBuffer({list(self.buffer)})"
